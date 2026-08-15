#!/usr/bin/env python3
"""Publish offline continuation archives as separate W&B runs.

The original run is left untouched.  Each continuation is uploaded under a new
run ID, linked to its original run through explicit config fields, and placed in
the same requested W&B group.  The command is dry-run by default.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import wandb

try:
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore
except ImportError as exc:  # pragma: no cover - depends on the active env
    raise SystemExit(
        "W&B internals are unavailable. Run this from the marl2grid environment."
    ) from exc


DEFAULT_ENTITY = "corentin-plumet-epfl"
DEFAULT_PROJECT = "Grid2Op"
OFFLINE_DIR_RE = re.compile(
    r"^offline-run-(?P<timestamp>\d{8}_\d{6})-(?P<run_name>.+)$"
)
JOB_ID_RE = re.compile(r"-(?P<job_id>\d+)$")


@dataclass(frozen=True)
class PublicationProfile:
    name: str
    group: str
    archive_run_re: re.Pattern[str]
    expected_archive_names: tuple[str, ...]
    archive_prefix: str
    base_prefix: str
    tags: tuple[str, ...]

    def base_lookup_name(self, archive_run_name: str) -> str:
        if not archive_run_name.startswith(self.archive_prefix):
            raise ValueError(
                f"Run {archive_run_name!r} does not start with the profile prefix "
                f"{self.archive_prefix!r}."
            )
        return self.base_prefix + archive_run_name[len(self.archive_prefix) :]


def _nls_names() -> tuple[str, ...]:
    return tuple(
        f"cas_hl_NLS_{pool}_{features}_{head}_s0"
        for pool in ("mean", "tmean")
        for features in ("f0", "f1")
        for head in ("a0h0", "a0h1")
    )


def _s3dw_resumed_names() -> tuple[str, ...]:
    return tuple(
        [
            f"nl_s3dw_bus_n0_none_e0n0v0_mp{depth}_h{width}_s0"
            for depth in (1, 2)
            for width in (16, 32, 64, 128)
        ]
        + ["nl_s3dw_bus_n0_none_e0n0v0_mp3_h32_s0"]
    )


PUBLICATION_PROFILES = {
    "nls_cas_hl": PublicationProfile(
        name="nls_cas_hl",
        group="NLS_cas_hl",
        archive_run_re=re.compile(
            r"^cas_hl_NLS_(?:mean|tmean)_f[01]_a0h[01]_s0$"
        ),
        expected_archive_names=_nls_names(),
        archive_prefix="cas_hl_NLS_",
        base_prefix="cas_hl_NLS_",
        tags=("NLS_cas_hl",),
    ),
    "nl_s3dw": PublicationProfile(
        name="nl_s3dw",
        group="gs_s3dw",
        archive_run_re=re.compile(
            r"^nl_s3dw_bus_n0_none_e0n0v0_"
            r"(?:mp[12]_h(?:16|32|64|128)|mp3_h32)_s0$"
        ),
        expected_archive_names=_s3dw_resumed_names(),
        # The no-leakage configs and continuation archives use ``nl_`` while
        # the original cloud sweep predates that rename and uses ``gs_``.
        archive_prefix="nl_s3dw_",
        base_prefix="gs_s3dw_",
        tags=("gs_s3dw", "no-leakage"),
    ),
}


@dataclass(frozen=True)
class Archive:
    run_name: str
    run_file: Path
    offline_dir: Path
    timestamp: str
    job_id: str
    min_step: int
    max_step: int
    history_rows: int

    @property
    def suffix(self) -> str:
        return self.job_id or self.timestamp.replace("_", "")


def _target_run_id(archive: Archive, profile: PublicationProfile) -> str:
    return f"{profile.base_lookup_name(archive.run_name)}-cont-{archive.suffix}"


def _target_display_name(archive: Archive, profile: PublicationProfile) -> str:
    return (
        f"{profile.base_lookup_name(archive.run_name)} "
        f"[continuation {archive.suffix}]"
    )


def _parse_json(value: str) -> Any:
    try:
        return json.loads(value)
    except Exception:
        return value


def _item_values(items: Any) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for item in items:
        key_parts = list(item.nested_key) or ([item.key] if item.key else [])
        if not key_parts:
            continue
        key = "/".join(str(part).strip("/") for part in key_parts)
        values[key] = _parse_json(item.value_json)
    return values


def _training_step(values: dict[str, Any]) -> int | None:
    for key in ("charts/global_step", "global_step"):
        if key not in values:
            continue
        try:
            return int(values[key])
        except (TypeError, ValueError):
            continue
    return None


def _history_item_steps(record: Any) -> tuple[int | None, int | None]:
    """Return (training step, W&B internal step) from one history record."""
    values = _item_values(record.history.item)

    training_step = _training_step(values)

    internal_step: int | None = None
    if "_step" in values:
        try:
            internal_step = int(values["_step"])
        except (TypeError, ValueError):
            pass
    if internal_step is None and record.history.step.num:
        internal_step = int(record.history.step.num)
    return training_step, internal_step


def history_range(path: Path) -> tuple[int, int, int]:
    training_steps: list[int] = []
    internal_steps: list[int] = []
    summary_steps: list[int] = []
    rows = 0
    store = DataStore()
    store.open_for_scan(str(path))
    try:
        while True:
            data = store.scan_data()
            if data is None:
                break
            record = wandb_internal_pb2.Record()
            record.ParseFromString(data)
            record_type = record.WhichOneof("record_type")
            if record_type == "summary":
                summary_step = _training_step(_item_values(record.summary.update))
                if summary_step is not None:
                    summary_steps.append(summary_step)
                continue
            if record_type == "run":
                summary_step = _training_step(
                    _item_values(record.run.summary.update)
                )
                if summary_step is not None:
                    summary_steps.append(summary_step)
                continue
            if record_type != "history":
                continue
            training_step, internal_step = _history_item_steps(record)
            if training_step is not None:
                training_steps.append(training_step)
            if internal_step is not None:
                internal_steps.append(internal_step)
            if training_step is not None or internal_step is not None:
                rows += 1
    finally:
        store.close()

    # Resumed offline runs can restart W&B's internal counter even though the
    # checkpoint's training clock continues. Prefer the explicitly logged
    # training clock. Some W&B archives retain it only in the final summary;
    # in that case the constant offset to the internal sequence is recoverable
    # from the two final values.
    if training_steps:
        steps = training_steps
    elif summary_steps and internal_steps:
        offset = max(summary_steps) - max(internal_steps)
        steps = [step + offset for step in internal_steps]
    elif summary_steps:
        steps = summary_steps
    else:
        steps = internal_steps
    if not steps:
        raise ValueError(f"No stepped history records found in {path}")
    return min(steps), max(steps), rows


def _job_id(path: Path, stop: Path) -> str:
    for parent in [path.parent, *path.parents]:
        match = JOB_ID_RE.search(parent.name)
        if match:
            return match.group("job_id")
        if parent == stop:
            break
    return ""


def discover_archives(
    roots: list[Path],
    *,
    run_name_re: re.Pattern[str],
    min_start_step: int,
    selected_job_ids: set[str],
) -> list[Archive]:
    archives: list[Archive] = []
    seen_files: set[Path] = set()
    for root in roots:
        root = root.expanduser().resolve()
        if not root.exists():
            print(f"WARNING: archive root does not exist: {root}", file=sys.stderr)
            continue
        for offline_dir in sorted(root.rglob("offline-run-*")):
            if not offline_dir.is_dir():
                continue
            match = OFFLINE_DIR_RE.match(offline_dir.name)
            if not match:
                continue
            run_name = match.group("run_name")
            if not run_name_re.fullmatch(run_name):
                continue
            candidates = sorted(offline_dir.glob("run-*.wandb"))
            if not candidates:
                candidates = sorted(offline_dir.glob("run-*.wandb.synced"))
            if len(candidates) != 1:
                print(
                    f"WARNING: expected one run file in {offline_dir}, found "
                    f"{len(candidates)}; skipping.",
                    file=sys.stderr,
                )
                continue
            run_file = candidates[0].resolve()
            if run_file in seen_files:
                continue
            seen_files.add(run_file)
            job_id = _job_id(offline_dir, root)
            if selected_job_ids and job_id not in selected_job_ids:
                continue
            try:
                min_step, max_step, rows = history_range(run_file)
            except Exception as exc:
                print(f"WARNING: cannot inspect {run_file}: {exc}", file=sys.stderr)
                continue
            if min_step < min_start_step:
                print(
                    f"SKIP non-continuation archive {run_name}: first step "
                    f"{min_step:,} < {min_start_step:,} ({offline_dir})"
                )
                continue
            archives.append(
                Archive(
                    run_name=run_name,
                    run_file=run_file,
                    offline_dir=offline_dir,
                    timestamp=match.group("timestamp"),
                    job_id=job_id,
                    min_step=min_step,
                    max_step=max_step,
                    history_rows=rows,
                )
            )
    return sorted(archives, key=lambda item: (item.run_name, item.min_step, item.suffix))


def _base_run(api: wandb.Api, entity: str, project: str, run_name: str):
    try:
        return api.run(f"{entity}/{project}/{run_name}")
    except Exception:
        matches = list(
            api.runs(
                f"{entity}/{project}",
                filters={"display_name": run_name},
            )
        )
        matches = [run for run in matches if not run.config.get("is_continuation")]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one original cloud run named {run_name}, found {len(matches)}"
            )
        return matches[0]


def _set_metadata(
    run: Any,
    *,
    group: str,
    name: str | None = None,
    tags: list[str] | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    if name is not None:
        run.name = name
    run.group = group
    if tags:
        run.tags = sorted(set(list(run.tags or []) + tags))
    if config:
        # Public-API config is a mutable mapping; mutating it in place works
        # across W&B versions that expose no property setter.
        run.config.update(config)
    run.update()


def _semantic_sync_error(output: str) -> bool:
    lower = output.lower()
    return any(
        marker in lower
        for marker in (
            "wandb: error",
            "error while calling w&b api",
            "previously created and deleted",
            "sync failed",
        )
    )


def _continuation_config(archive: Archive, base: Any) -> dict[str, Any]:
    return {
        "is_continuation": True,
        "continuation_of_run_id": str(base.id),
        "continuation_of_run_name": str(base.name),
        "continuation_archive_path": str(archive.offline_dir),
        "continuation_archive_timestamp": archive.timestamp,
        "continuation_job_id": archive.job_id or None,
        "continuation_start_step": archive.min_step,
        "continuation_end_step": archive.max_step,
        "continuation_history_rows": archive.history_rows,
    }


def _create_target_run(
    archive: Archive,
    base: Any,
    *,
    target_id: str,
    display_name: str,
    entity: str,
    project: str,
    group: str,
    tags: list[str],
) -> None:
    run = wandb.init(
        entity=entity,
        project=project,
        id=target_id,
        name=display_name,
        group=group,
        job_type="continuation",
        tags=[*tags, "continuation"],
        config=_continuation_config(archive, base),
        resume="allow",
        reinit=True,
    )
    run.finish()


def _cloud_run_with_retry(
    api: wandb.Api,
    path: str,
    *,
    attempts: int = 12,
    delay: float = 2.0,
):
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            api.flush()
            return api.run(path)
        except Exception as exc:  # newly created runs can take a moment to appear
            last_error = exc
            time.sleep(delay)
    raise RuntimeError(f"New W&B run did not become visible: {last_error}")


def parse_args() -> argparse.Namespace:
    task_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=sorted(PUBLICATION_PROFILES),
        default="nls_cas_hl",
        help=(
            "Continuation family to publish. The profile fixes the accepted "
            "archive names, original-run name mapping, default group, and tags."
        ),
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        action="append",
        default=None,
        help=(
            "Root searched recursively for offline-run-* folders. Repeatable. "
            "Default: <repository>/outputs."
        ),
    )
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument(
        "--group",
        default=None,
        help="Override the W&B group selected by --profile.",
    )
    parser.add_argument(
        "--job-id",
        action="append",
        default=[],
        help="Optional Slurm job ID filter. Repeatable.",
    )
    parser.add_argument(
        "--min-start-step",
        type=int,
        default=1_000_000,
        help="Archives beginning below this step are treated as normal runs and skipped.",
    )
    parser.add_argument("--api-timeout", type=int, default=120)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Create/group/upload runs. Without this flag, only show the plan.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-upload a continuation already marked continuation_sync_completed.",
    )
    args = parser.parse_args()
    if args.archive_root is None:
        args.archive_root = [task_dir.parent / "outputs"]
    if args.group is None:
        args.group = PUBLICATION_PROFILES[args.profile].group
    return args


def main() -> int:
    args = parse_args()
    profile = PUBLICATION_PROFILES[args.profile]
    archives = discover_archives(
        args.archive_root,
        run_name_re=profile.archive_run_re,
        min_start_step=args.min_start_step,
        selected_job_ids=set(args.job_id),
    )
    if not archives:
        raise SystemExit("No continuation archives matched. Nothing to publish.")

    duplicates: dict[str, int] = {}
    for archive in archives:
        target_id = _target_run_id(archive, profile)
        duplicates[target_id] = duplicates.get(target_id, 0) + 1
    repeated = [run_id for run_id, count in duplicates.items() if count > 1]
    if repeated:
        raise SystemExit(f"Duplicate generated continuation IDs: {repeated}")

    found_names = {archive.run_name for archive in archives}
    missing_names = sorted(set(profile.expected_archive_names) - found_names)

    print("========== Continuation publication plan ==========")
    print(f"Profile: {profile.name}")
    print(f"Entity/project: {args.entity}/{args.project}")
    print(f"Group: {args.group}")
    print(f"Mode: {'EXECUTE' if args.execute else 'DRY-RUN'}")
    print(f"Archives: {len(archives)}")
    if missing_names:
        print(
            f"WARNING: {len(missing_names)} expected archive(s) were not found: "
            + ", ".join(missing_names)
        )
    for archive in archives:
        target_id = _target_run_id(archive, profile)
        display_name = _target_display_name(archive, profile)
        print(
            f"- {archive.run_name}: {archive.min_step:,}..{archive.max_step:,} "
            f"({archive.history_rows:,} records)\n"
            f"    source: {archive.run_file}\n"
            f"    target: {target_id} / {display_name}"
        )
    print("=======================================================")

    api = wandb.Api(timeout=args.api_timeout)
    bases: dict[str, Any] = {}
    failures = 0
    for archive_run_name in profile.expected_archive_names:
        base_lookup_name = profile.base_lookup_name(archive_run_name)
        try:
            base = _base_run(api, args.entity, args.project, base_lookup_name)
            bases[archive_run_name] = base
            print(
                f"BASE {archive_run_name} -> {base.name}: "
                f"id={base.id}, group={base.group or '<none>'} "
                f"-> {args.group}"
            )
            if args.execute:
                _set_metadata(
                    base,
                    group=args.group,
                    tags=[*profile.tags, "base-segment"],
                )
        except Exception as exc:
            print(
                f"FAILED base lookup/grouping for {archive_run_name} "
                f"({base_lookup_name}): {exc}",
                file=sys.stderr,
            )
            failures += 1

    for archive in archives:
        base = bases.get(archive.run_name)
        if base is None:
            print(f"SKIP {archive.run_name}: original run unavailable", file=sys.stderr)
            failures += 1
            continue
        target_id = _target_run_id(archive, profile)
        display_name = _target_display_name(archive, profile)
        target_path = f"{args.entity}/{args.project}/{target_id}"
        existing = None
        try:
            existing = api.run(target_path)
        except Exception:
            pass
        if existing is not None and existing.config.get("continuation_sync_completed") and not args.force:
            print(f"SKIP {target_id}: already marked as completely synced")
            continue
        if not args.execute:
            print(f"DRY-RUN publish {archive.run_name} -> {target_id}")
            continue

        try:
            _create_target_run(
                archive,
                base,
                target_id=target_id,
                display_name=display_name,
                entity=args.entity,
                project=args.project,
                group=args.group,
                tags=list(profile.tags),
            )
            command = [
                "wandb",
                "sync",
                "--append",
                "--include-synced",
                "--no-mark-synced",
                "--entity",
                args.entity,
                "--project",
                args.project,
                "--id",
                target_id,
                str(archive.run_file),
            ]
            print(f"RUN {' '.join(command)}")
            result = subprocess.run(
                command,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            output = result.stdout or ""
            if output:
                print(output, end="" if output.endswith("\n") else "\n")
            if result.returncode != 0 or _semantic_sync_error(output):
                raise RuntimeError(f"wandb sync failed with exit code {result.returncode}")

            target = _cloud_run_with_retry(api, target_path)
            final_config = _continuation_config(archive, base)
            final_config["continuation_sync_completed"] = True
            _set_metadata(
                target,
                group=args.group,
                name=display_name,
                tags=[*profile.tags, "continuation"],
                config=final_config,
            )
            print(f"DONE https://wandb.ai/{args.entity}/{args.project}/runs/{target_id}")
        except Exception as exc:
            print(f"FAILED {target_id}: {exc}", file=sys.stderr)
            failures += 1

    if not args.execute:
        print("Dry run only. Re-run with --execute after checking the archive list.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
