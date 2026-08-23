"""Shared loader for the bus14 -> WCCI transfer evaluations.

`shared_wcci_transfer_digest.ipynb` builds this table inline; the two
question-driven notebooks next to it import it instead, so all three read the
same numbers from the same files. The loading, the artifact verification and
the cohort split are copied from the digest unchanged — what this module adds
is the factor decomposition the transfer study was actually designed around
(`configs/no_leakage_config/W_wcci_cas_hl_shared_transfer/README.md`):

    input condition   NL  raw features, absolute voltage angle
                      NLS physical scaling, edge-difference angle
    candidate pooling mean / typed-mean over the touched-node embeddings
    action descriptor f0 / f1, static candidate features fed to the scorer
    idle action       a0h0 shared scorer / a0h1 dedicated do-nothing head

and the adaptation route, split by fine-tuning protocol rather than lumped:
aggressive warm-start retraining at mk256 versus the conservative annealed
protocol at mk64 (`CONSERVATIVE_FINETUNING_MK64.md`).
"""

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd

TARGET_ENV = "bus36_wcci_nomaint"
N_EPISODES = 50
# Display order for the reduced action spaces. mk32 has a greedy ceiling but no
# transfer evaluations yet; notebooks derive the caps they plot from the runs
# themselves, so a new cap appears automatically once its results land.
MK_ORDER = [32, 64, 128, 256, 512, 1024]


def caps_in(frame):
    """The reduced action spaces actually present in `frame`, in display order."""
    present = {int(m) for m in frame["mk"].dropna().unique()}
    return [m for m in MK_ORDER if m in present] + sorted(present - set(MK_ORDER))

SCALING_LABELS = {"NL": "raw inputs", "NLS": "physical scaling"}
POOL_LABELS = {"mean": "mean pool", "tmean": "typed-mean pool"}
DESCRIPTOR_LABELS = {0: "descriptor off", 1: "descriptor on"}
IDLE_LABELS = {0: "shared idle scorer", 1: "dedicated idle head"}
HEURISTIC_LABELS = {
    "none": "none",
    "rho_threshold": "global rho",
    "local_rho_threshold": "local rho",
}
ROUTE_ORDER = [
    "bus14 zero-shot",
    "bus14 zero-shot + AIB",
    "MAPPO fine-tune (aggressive)",
    "MAPPO fine-tune (conservative)",
    "BC on greedy labels",
    "MAPPO scratch",
]
ROUTE_COLORS = {
    "bus14 zero-shot": "#4C78A8",
    "bus14 zero-shot + AIB": "#72B7B2",
    "MAPPO fine-tune (aggressive)": "#F58518",
    "MAPPO fine-tune (conservative)": "#E4A11B",
    "BC on greedy labels": "#B279A2",
    "MAPPO scratch": "#54A24B",
}
SCALING_COLORS = {"raw inputs": "#4C78A8", "physical scaling": "#F58518"}
GATE_COLORS = {"ungated": "#9EC5E8", "gated": "#E45756"}

# Checkpoints are saved under a selection prefix; the campaign name follows it.
SELECTION_RE = re.compile(r"^(?P<selection>best_test|final)_(?P<rest>.+)$")
CONSERVATIVE_RE = re.compile(r"^ft\d+c")
AIB_RE = re.compile(r"^aibcas")
AIB_TARGET_RE = re.compile(r"_d(\d{3})_")
SCRATCH_RE = re.compile(r"^sc\d+c")
VARIANT_RE = re.compile(r"_(?P<pool>tmean|typed_mean|mean)_f(?P<features>[01])_a0h(?P<head>[01])")
SHORT_VARIANT_RE = re.compile(r"_(?P<pool>tm|m)f(?P<features>[01])h(?P<head>[01])(?:_|$)")
MK_RE = re.compile(r"_mk(\d+)")


def find_task_dir(start=None):
    """Locate `Topology_Task` from this file, falling back to the cwd."""
    candidates = [Path(__file__).resolve()] if start is None else []
    candidates.append(Path(start or Path.cwd()).resolve())
    for anchor in candidates:
        for candidate in [anchor, *anchor.parents]:
            direct = candidate if candidate.name == "Topology_Task" else candidate / "Topology_Task"
            if (direct / "main.py").is_file():
                return direct
    raise FileNotFoundError("Could not locate Topology_Task")


def load_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


class Bunch(dict):
    """Attribute access over a dict, so notebooks can write `data.runs`."""

    __getattr__ = dict.__getitem__


def _local_path(task_dir, path_string):
    """Map a cluster-side artifact path onto this checkout."""
    path = Path(path_string)
    if path.is_file():
        return path.resolve()
    if path.parts and path.parts[0] == "outputs":
        return task_dir / path
    if "Topology_Task" in path.parts:
        index = path.parts.index("Topology_Task")
        return task_dir.joinpath(*path.parts[index + 1:])
    return task_dir / path


def _parse_variant(*sources):
    for text in sources:
        match = VARIANT_RE.search(text)
        if match:
            pool = "tmean" if match.group("pool") in {"tmean", "typed_mean"} else "mean"
            return pool, int(match.group("features")), int(match.group("head"))
    for text in sources:
        match = SHORT_VARIANT_RE.search(text)
        if match:
            pool = "tmean" if match.group("pool") == "tm" else "mean"
            return pool, int(match.group("features")), int(match.group("head"))
    return "unknown", -1, -1


def split_selection(checkpoint_stem):
    """Separate the checkpoint-selection prefix from the campaign name.

    A run writes several checkpoints — `best_test_<label>`, `final_<label>` and
    the rolling `<label>` — and which one was evaluated is a factor in its own
    right, not part of the model's identity. Keeping them apart also stops the
    prefix from hiding the campaign from the route classifier.
    """
    match = SELECTION_RE.match(checkpoint_stem)
    if match:
        return match.group("selection").replace("_", " "), match.group("rest")
    return "last", checkpoint_stem


def _classify_route(checkpoint, checkpoint_stem, haystack):
    """Adaptation route, with the two fine-tuning protocols kept apart.

    `ft<cap>c_*` is the conservative protocol: annealed actor learning rate,
    more conservative PPO updates, less exploration, shorter budget. The earlier
    `ft_*` / `trcas_*` mk256 runs restarted the actor at 1e-4 with the whole
    network unfrozen, which the config note itself calls aggressive warm-start
    retraining rather than fine-tuning. Averaging the two would hide the only
    "what kind of fine-tuning" contrast the data contains. The cap is read from
    the evaluation, not from the campaign prefix, so `ft32c` and `ft64c` are the
    same route at different action spaces.
    """
    _, label = split_selection(checkpoint_stem)
    if "dangerous_graph_bc" in haystack or label.startswith("bcg"):
        detail = "direct from bus14" if "direct_bus14" in label else "on top of MAPPO fine-tune"
        return "BC on greedy labels", detail
    if AIB_RE.match(label):
        # Same shared-candidate actor, trained on bus14 with the adaptive
        # intervention budget active. It transfers zero-shot like the plain
        # cas_hl arm, but it is a different training recipe and must not join
        # that arm's factorial.
        target = AIB_TARGET_RE.search(label)
        budget = f"intervention budget {int(target.group(1)) / 100:.2f}" if target else "adaptive intervention budget"
        return "bus14 zero-shot + AIB", budget
    if SCRATCH_RE.match(label):
        return "MAPPO scratch", "random init"
    if CONSERVATIVE_RE.match(label):
        return "MAPPO fine-tune (conservative)", "annealed lr, short budget"
    if "finetune" in checkpoint.lower() or label.startswith(("ft", "trcas")):
        return "MAPPO fine-tune (aggressive)", "actor lr restarted at 1e-4, fully unfrozen"
    return "bus14 zero-shot", "no target-grid gradient"


def _classify_result(task_dir, result_path):
    payload = load_json(result_path)
    checkpoint = str(payload.get("checkpoint", ""))
    checkpoint_stem = Path(checkpoint).stem
    stem = result_path.stem

    pool, features, head = _parse_variant(checkpoint_stem, stem)
    family = "NLS" if "_NLS_" in f"{checkpoint_stem}_{stem}" else (
        "NL" if "_NL_" in f"{checkpoint_stem}_{stem}" else "other"
    )
    haystack = f"{checkpoint.lower()}/{result_path.parent.name.lower()}/{stem.lower()}"
    route, detail = _classify_route(checkpoint, checkpoint_stem, haystack)

    heuristic_raw = str(payload.get("eval_action_heuristic", "none") or "none")
    heuristic = HEURISTIC_LABELS.get(heuristic_raw, heuristic_raw)
    rho = float(payload.get("eval_action_rho_threshold", np.nan))
    if heuristic == "none":
        rho = np.nan

    mk_match = (
        MK_RE.search(str(payload.get("target_reduced_action_space", "") or ""))
        or MK_RE.search(result_path.parent.name)
        or MK_RE.search(stem)
    )
    artifact = (payload.get("action_artifacts") or {}).get("episode_summary_csv")
    artifact_path = _local_path(task_dir, artifact) if artifact else None

    return {
        "key": f"{result_path.parent.name}/{stem}",
        "group": result_path.parent.name,
        "run": stem,
        "route": route,
        "route_detail": detail,
        "family": family,
        "scaling": SCALING_LABELS.get(family, family),
        "pool": pool,
        "pooling": POOL_LABELS.get(pool, pool),
        "features": features,
        "descriptor": DESCRIPTOR_LABELS.get(features, "?"),
        "head": head,
        "idle": IDLE_LABELS.get(head, "?"),
        "variant": f"{family}_{pool}_f{features}_a0h{head}",
        "cell": f"{pool}_f{features}_a0h{head}",
        "mk": int(mk_match.group(1)) if mk_match else np.nan,
        "heuristic": heuristic,
        "gate": "ungated" if heuristic == "none" else "gated",
        "rho": rho,
        "checkpoint": checkpoint,
        "checkpoint_stem": checkpoint_stem,
        "selection": split_selection(checkpoint_stem)[0],
        "checkpoint_step": payload.get("checkpoint_global_step"),
        "physical_scaling_flag": payload.get("gnn_physical_scaling_effective"),
        "reported_overall_pct": payload.get("survival_percent"),
        "target_env": payload.get("target_env_id"),
        "eval_episodes": payload.get("eval_episodes"),
        "episode_csv": str(artifact_path) if artifact_path else None,
        "has_episodes": bool(artifact_path and artifact_path.is_file()),
    }


def _align_greedy_to_baseline(frame, baseline, summary_path):
    """Every greedy run substitutes one do-nothing-perfect chronic for another.

    Audited in the earlier full-test notebook and benign: both the dropped and
    the added chronic have do-nothing and greedy survival 1.0, so neither the
    overall mean nor the difficult cohort moves. Any other mismatch is refused.
    """
    greedy_ids = set(frame["greedy_chronic_fingerprint"].astype(str))
    baseline_ids = set(baseline["chronic_fingerprint"].astype(str))
    if greedy_ids == baseline_ids:
        return frame
    baseline_only = baseline_ids - greedy_ids
    greedy_only = greedy_ids - baseline_ids
    if not (len(baseline_only) == len(greedy_only) == 1):
        raise ValueError(f"{summary_path}: chronic sets differ by more than one episode")
    dropped = baseline.loc[baseline["chronic_fingerprint"] == next(iter(baseline_only))].iloc[0]
    added = frame.loc[frame["greedy_chronic_fingerprint"] == next(iter(greedy_only))].iloc[0]
    benign = (
        bool(dropped["dn_full_survival"])
        and bool(added["do_nothing_full_survival"])
        and bool(added["greedy_full_survival"])
        and np.isclose(added["do_nothing_survival"], 1.0)
        and np.isclose(added["greedy_survival"], 1.0)
    )
    if not benign:
        raise ValueError(f"{summary_path}: unaudited chronic substitution")
    frame = frame.copy()
    swap = frame["greedy_chronic_fingerprint"] == added["greedy_chronic_fingerprint"]
    frame.loc[swap, "greedy_chronic_fingerprint"] = dropped["chronic_fingerprint"]
    frame.loc[swap, "greedy_chronic_name"] = dropped["chronic_name"]
    return frame


def load(task_dir=None):
    """Load every comparable WCCI evaluation, the do-nothing floor and the greedy ceiling."""
    task_dir = Path(task_dir) if task_dir else find_task_dir()
    wcci_root = task_dir / "outputs" / "full_test_eval" / "shared" / "wcci"
    bus14_root = task_dir / "outputs" / "full_test_eval" / "shared" / "bus14"
    greedy_root = task_dir / "outputs" / "greedy_wcci_nomaint_50"
    do_nothing_path = (
        task_dir / "outputs" / "do_nothing_eval" / "wcci_test_nomaint_50"
        / "do_nothing_summary.json"
    )

    # --- do-nothing floor and the difficult / easy cohort split ---------------
    baseline = pd.DataFrame(load_json(do_nothing_path)["episodes"])[[
        "chronic_fingerprint", "chronic_name", "survival", "steps", "max_steps",
        "full_survival",
    ]].rename(columns={
        "survival": "dn_survival", "steps": "dn_steps", "full_survival": "dn_full_survival",
    })
    assert len(baseline) == N_EPISODES and baseline["chronic_fingerprint"].is_unique
    baseline["is_hard"] = ~baseline["dn_full_survival"]
    hard_fingerprints = set(baseline.loc[baseline["is_hard"], "chronic_fingerprint"])
    dn_overall = 100 * baseline["dn_survival"].mean()
    dn_hard = 100 * baseline.loc[baseline["is_hard"], "dn_survival"].mean()

    # --- greedy ceiling, one row per reduced action space ---------------------
    greedy_rows, greedy_frames = [], []
    for summary_path in sorted(greedy_root.glob("*/greedy_vs_do_nothing_summary.json")):
        payload = load_json(summary_path)
        if payload.get("env_id") != TARGET_ENV:
            continue
        cap_match = re.search(r"_mk(\d+)\.json$", str(payload.get("reduced_action_space", "")))
        if cap_match is None:
            continue
        frame = pd.DataFrame(payload["episodes"])
        if len(frame) != N_EPISODES:
            raise ValueError(f"{summary_path} does not hold {N_EPISODES} episodes")
        frame = _align_greedy_to_baseline(frame, baseline, summary_path)
        if not hard_fingerprints <= set(frame["greedy_chronic_fingerprint"]):
            raise ValueError(f"{summary_path} is missing difficult chronics")
        frame = frame.assign(
            mk=int(cap_match.group(1)),
            is_hard=frame["greedy_chronic_fingerprint"].isin(hard_fingerprints),
        )
        greedy_frames.append(frame)
        hard = frame.loc[frame["is_hard"]]
        greedy_rows.append({
            "mk": int(cap_match.group(1)),
            "greedy_overall_pct": 100 * frame["greedy_survival"].mean(),
            "greedy_hard_pct": 100 * hard["greedy_survival"].mean(),
            "greedy_hard_rescues": int(hard["greedy_full_survival"].sum()),
            "greedy_easy_kept": int(frame.loc[~frame["is_hard"], "greedy_full_survival"].sum()),
        })
    greedy_ceiling = pd.DataFrame(greedy_rows).sort_values("mk").reset_index(drop=True)
    greedy_episodes = pd.concat(greedy_frames, ignore_index=True)

    # --- inventory of comparable evaluations ---------------------------------
    inventory = pd.DataFrame(
        [_classify_result(task_dir, p) for p in sorted(wcci_root.rglob("*.json"))]
    )
    comparable = (
        (inventory["target_env"] == TARGET_ENV) & (inventory["eval_episodes"] == N_EPISODES)
    )
    inventory = inventory.loc[comparable].reset_index(drop=True)
    inventory["route"] = pd.Categorical(inventory["route"], categories=ROUTE_ORDER, ordered=True)

    # --- per-episode artifacts, verified against each run's own JSON ----------
    # A result may point at another evaluation's artifact directory; loading it
    # blindly would silently attribute the wrong episodes.
    episode_frames, rejected = [], []
    for row in inventory.loc[inventory["has_episodes"]].itertuples():
        frame = pd.read_csv(row.episode_csv).merge(
            baseline, on="chronic_fingerprint", validate="one_to_one", suffixes=("", "_baseline"),
        )
        if len(frame) != N_EPISODES:
            raise ValueError(f"{row.key}: expected {N_EPISODES} episodes, got {len(frame)}")
        if not np.isclose(100 * frame["survival"].mean(), row.reported_overall_pct, rtol=0, atol=1e-6):
            rejected.append({"key": row.key, "episode_csv": row.episode_csv})
            continue
        frame["key"] = row.key
        episode_frames.append(frame)
    inventory["has_episodes"] = inventory["key"].isin(
        {frame["key"].iloc[0] for frame in episode_frames}
    )

    episodes = pd.concat(episode_frames, ignore_index=True)
    episodes["is_hard"] = episodes["chronic_fingerprint"].isin(hard_fingerprints)
    episodes["full_survival"] = np.isclose(episodes["survival"], 1.0)
    episodes["delta_pp"] = 100 * (episodes["survival"] - episodes["dn_survival"])
    episodes["delta_steps"] = episodes["steps"] - episodes["dn_steps"]

    def summarise(group):
        hard = group.loc[group["is_hard"]]
        easy = group.loc[~group["is_hard"]]
        return pd.Series({
            "overall_pct": 100 * group["survival"].mean(),
            "hard_pct": 100 * hard["survival"].mean(),
            "hard_rescues": int(hard["full_survival"].sum()),
            "hard_wins": int((hard["delta_pp"] > 1e-9).sum()),
            "hard_ties": int((hard["delta_pp"].abs() <= 1e-9).sum()),
            "hard_losses": int((hard["delta_pp"] < -1e-9).sum()),
            "hard_steps": hard["steps"].mean(),
            "hard_dn_steps": hard["dn_steps"].mean(),
            "easy_pct": 100 * easy["survival"].mean(),
            "easy_kept": int(easy["full_survival"].sum()),
        })

    runs = inventory.merge(
        episodes.groupby("key", sort=False).apply(summarise, include_groups=False).reset_index(),
        on="key", how="left",
    )
    checked = runs.dropna(subset=["overall_pct"])
    assert np.allclose(checked["overall_pct"], checked["reported_overall_pct"])
    runs["overall_pct"] = runs["overall_pct"].fillna(runs["reported_overall_pct"])

    # --- headroom capture against the greedy oracle at the same action space --
    greedy_overall = greedy_ceiling.set_index("mk")["greedy_overall_pct"].to_dict()
    greedy_hard = greedy_ceiling.set_index("mk")["greedy_hard_pct"].to_dict()
    runs["greedy_hard_pct"] = runs["mk"].map(greedy_hard)
    runs["capture_hard_pct"] = 100 * (
        (runs["hard_pct"] - dn_hard) / (runs["greedy_hard_pct"] - dn_hard)
    )
    runs["capture_overall_pct"] = 100 * (
        (runs["overall_pct"] - dn_overall) / (runs["mk"].map(greedy_overall) - dn_overall)
    )
    runs["delta_hard_pp"] = runs["hard_pct"] - dn_hard
    runs["delta_overall_pp"] = runs["overall_pct"] - dn_overall
    runs["hard_step_ratio"] = runs["hard_steps"] / runs["hard_dn_steps"]
    runs["collapsed"] = runs["hard_pct"] < 0.5

    # --- the paired bus14 source-grid evaluations ----------------------------
    bus14_rows = []
    for result_path in sorted(bus14_root.rglob("*.json")):
        payload = load_json(result_path)
        stem = Path(payload.get("checkpoint", result_path.stem)).stem
        pool, features, head = _parse_variant(stem, result_path.stem)
        family = "NLS" if "_NLS_" in stem else ("NL" if "_NL_" in stem else "other")
        bus14_rows.append({
            "variant": f"{family}_{pool}_f{features}_a0h{head}",
            "bus14_group": result_path.parent.name,
            "bus14_pct": payload.get("survival_percent"),
        })
    bus14 = pd.DataFrame(bus14_rows)
    # The WCCI zero-shot arm transfers exactly these two bus14 batches.
    bus14 = (
        bus14.loc[bus14["bus14_group"].isin({"NL_cas_hl_shared", "NLS_cas_hl_izar_shared"})]
        .drop_duplicates(subset="variant")
        .reset_index(drop=True)
    )

    return Bunch(
        task_dir=task_dir,
        export_dir=task_dir / "outputs" / "analysis",
        baseline=baseline,
        episodes=episodes,
        runs=runs,
        bus14=bus14,
        greedy_ceiling=greedy_ceiling,
        greedy_episodes=greedy_episodes,
        greedy_hard=greedy_hard,
        greedy_overall=greedy_overall,
        rejected_artifacts=pd.DataFrame(rejected),
        n_hard=len(hard_fingerprints),
        n_easy=N_EPISODES - len(hard_fingerprints),
        n_episodes=N_EPISODES,
        dn_overall=dn_overall,
        dn_hard=dn_hard,
    )


def matched_pairs(runs, factor, levels, within, value="hard_pct"):
    """Wide table of `value` for the two levels of `factor`, over cells sharing `within`.

    Only cells where both levels were evaluated survive, so the comparison is
    paired rather than a difference of unbalanced group means.
    """
    scoped = runs.dropna(subset=[value]).copy()
    # `rho` is NaN for every ungated run, and a NaN anywhere in the index would
    # make pandas drop the whole cell — which would silently delete the ungated
    # half of the comparison. Fill it with a sentinel instead.
    for column in within:
        if scoped[column].isna().any():
            scoped[column] = scoped[column].astype(object).where(scoped[column].notna(), "—")
    wide = scoped.pivot_table(index=list(within), columns=factor, values=value,
                              aggfunc="max", observed=True)
    wide = wide.reindex(columns=list(levels)).dropna()
    wide["delta"] = wide[levels[1]] - wide[levels[0]]
    # The cell's own factor values stay addressable as columns, so a caller can
    # split the pairs by gate, cap or architecture without re-parsing a label.
    return wide.sort_values("delta").reset_index()


def paired_bootstrap(deltas, n_boot=20_000, seed=20260820):
    """Percentile CI for the mean of a paired per-chronic delta.

    Resamples chronics only: it does not cover training-seed noise, nor the
    optimism from having selected caps and thresholds on this same set.
    """
    rng = np.random.default_rng(seed)
    deltas = np.asarray(deltas, dtype=float)
    draws = deltas[rng.integers(0, len(deltas), size=(n_boot, len(deltas)))].mean(axis=1)
    return deltas.mean(), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


# --- the from-scratch architecture screen (X_wcci_scratch_arch) -------------
# These encoders were trained on WCCI directly, with an MLP head over the
# reduced action list rather than the shared candidate scorer, so they are a
# separate arm from everything `load()` returns and are kept in their own table.
WSC_SCHEMA_LABELS = {
    "bus_e0n0v0": "busbar, no augmentation",
    "bus_e1n0v0": "busbar + same-substation edges",
    "bus_e0n1v0": "busbar + substation summary nodes",
    "het_ga2b_lb2a": "disaggregated, gen a2b / load b2a",
    "het_gbi_lbi": "disaggregated, both bidirectional",
}
# bus14 full-test survival of the byte-identical encoder, from the corrected
# screens (`nl_s2em`, `nl_hmdem`). `het_gbi_lbi` has no corrected bus14 rerun;
# the pre-correction `gs_hmd` number is not comparable and is deliberately
# omitted rather than substituted.
WSC_BUS14_SOURCE = {
    "bus_e0n0v0": ("nl_s2em_bus_n0_none_e0n0v0_s0", 95.45),
    "bus_e1n0v0": ("nl_s2em_bus_n0_none_e1n0v0_s0", 97.19),
    "het_ga2b_lb2a": ("nl_hmdem_hetero_n0_none_ga2b_lb2a_s0", 99.12),
}
WSC_RE = re.compile(r"wsc_(?P<schema>bus_e\d n\d v\d |het_[a-z0-9_]+?)_mk(?P<mk>\d+)_s(?P<seed>\d+)$".replace(" ", ""))


def load_wsc(reference):
    """Load the from-scratch architecture screen against `reference` from `load()`.

    Applies the same artifact verification: a result whose episode CSV does not
    reproduce the survival in its own JSON is rejected. That matters here — the
    best-test and final checkpoints of each run were evaluated into the *same*
    action-log directory, so one of each pair is mis-attributed on disk.
    """
    task_dir = Path(reference["task_dir"])
    root = task_dir / "outputs" / "full_test_eval" / "wsc"
    baseline = reference["baseline"]
    hard_fingerprints = set(baseline.loc[baseline["is_hard"], "chronic_fingerprint"])

    rows, frames, rejected = [], [], []
    for path in sorted(root.glob("*.json")):
        payload = load_json(path)
        if payload.get("target_env_id") != TARGET_ENV or payload.get("eval_episodes") != N_EPISODES:
            continue
        stem = Path(str(payload.get("checkpoint", ""))).stem
        match = WSC_RE.search(stem)
        if match is None:
            continue
        artifact = (payload.get("action_artifacts") or {}).get("episode_summary_csv")
        artifact_path = _local_path(task_dir, artifact) if artifact else None
        row = {
            "key": path.stem,
            "run": stem,
            "checkpoint_kind": "best test" if stem.startswith("best_test_") else "final",
            "schema": match.group("schema"),
            "schema_label": WSC_SCHEMA_LABELS.get(match.group("schema"), match.group("schema")),
            "mk": int(match.group("mk")),
            "checkpoint_step": payload.get("checkpoint_global_step"),
            "reported_overall_pct": payload.get("survival_percent"),
            "episode_csv": str(artifact_path) if artifact_path else None,
        }
        if not (artifact_path and artifact_path.is_file()):
            rejected.append({**row, "reason": "artifact missing"})
            continue
        frame = pd.read_csv(artifact_path).merge(
            baseline, on="chronic_fingerprint", validate="one_to_one", suffixes=("", "_baseline"),
        )
        if len(frame) != N_EPISODES:
            rejected.append({**row, "reason": f"{len(frame)} episodes"})
            continue
        if not np.isclose(100 * frame["survival"].mean(), row["reported_overall_pct"],
                          rtol=0, atol=1e-6):
            rejected.append({**row, "reason": "artifact does not reproduce its own JSON"})
            continue
        frame["key"] = row["key"]
        frames.append(frame)
        rows.append(row)

    runs = pd.DataFrame(rows)
    episodes = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not episodes.empty:
        episodes["is_hard"] = episodes["chronic_fingerprint"].isin(hard_fingerprints)
        episodes["full_survival"] = np.isclose(episodes["survival"], 1.0)
        episodes["delta_pp"] = 100 * (episodes["survival"] - episodes["dn_survival"])

        def summarise(group):
            hard = group.loc[group["is_hard"]]
            easy = group.loc[~group["is_hard"]]
            return pd.Series({
                "overall_pct": 100 * group["survival"].mean(),
                "hard_pct": 100 * hard["survival"].mean(),
                "hard_rescues": int(hard["full_survival"].sum()),
                "hard_wins": int((hard["delta_pp"] > 1e-9).sum()),
                "hard_losses": int((hard["delta_pp"] < -1e-9).sum()),
                "easy_kept": int(easy["full_survival"].sum()),
            })

        runs = runs.merge(
            episodes.groupby("key", sort=False).apply(summarise, include_groups=False).reset_index(),
            on="key", how="left",
        )
        runs["capture_hard_pct"] = 100 * (
            (runs["hard_pct"] - reference["dn_hard"])
            / (runs["mk"].map(reference["greedy_hard"]) - reference["dn_hard"])
        )
        runs["bus14_pct"] = runs["schema"].map(
            {k: v[1] for k, v in WSC_BUS14_SOURCE.items()}
        )
        runs["collapsed"] = runs["hard_pct"] < 0.5
    return Bunch(runs=runs, episodes=episodes, rejected=pd.DataFrame(rejected))
