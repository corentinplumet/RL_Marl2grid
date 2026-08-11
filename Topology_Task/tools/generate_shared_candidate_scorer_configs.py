#!/usr/bin/env python3
"""Generate the paired NL/NLS bus14 shared-candidate-scorer sweep."""

from __future__ import annotations

from itertools import product
from pathlib import Path


TASK_DIR = Path(__file__).resolve().parents[1]
CONFIG_ROOT = TASK_DIR / "configs" / "gnn_action_scoring"
OUTPUT_DIR = CONFIG_ROOT / "shared_scorer_connected"
BASE_CONFIGS = {
    "NL": CONFIG_ROOT / "no_leak" / "cas_hl_NL_mean_f0_a0h0_s0.toml",
    "NLS": (
        CONFIG_ROOT
        / "no_leak_scaled"
        / "cas_hl_NLS_mean_f0_a0h0_s0.toml"
    ),
}


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one occurrence of {old!r}, found {count}.")
    return text.replace(old, new, 1)


def build_config(
    preprocessing: str,
    pool_name: str,
    pool_mode: str,
    use_features: bool,
    use_do_nothing_head: bool,
) -> tuple[str, str]:
    source_name = f"cas_hl_{preprocessing}_mean_f0_a0h0_s0"
    name = (
        f"cas_hl_{preprocessing}_shared_{pool_name}_"
        f"f{int(use_features)}_a0h{int(use_do_nothing_head)}_s0"
    )
    text = BASE_CONFIGS[preprocessing].read_text()
    text = text.replace(source_name, name)
    text = replace_once(
        text,
        "# Generated graph-architecture screening configuration.\n"
        "# Keep non-screened settings identical so the comparisons remain paired.",
        "# Shared-candidate-scorer bus14 run.\n"
        "# NL and NLS differ only in their input preprocessing.",
    )
    text = text.replace(
        'run_dir = "outputs/jed-{config_name}-{job_id}"',
        'run_dir = "outputs/izar-{config_name}-{job_id}"',
    )
    text = text.replace('MAX_TIME_LIMIT_MINUTES = "10080"', 'MAX_TIME_LIMIT_MINUTES = "4260"')
    text = text.replace("time_limit = 10080", "time_limit = 4260")
    text = replace_once(
        text,
        'candidate_action_pool = "mean"',
        f'candidate_action_pool = "{pool_mode}"',
    )
    text = replace_once(
        text,
        "candidate_action_use_features = false",
        f"candidate_action_use_features = {str(use_features).lower()}",
    )
    text = replace_once(
        text,
        "candidate_action_do_nothing_head = false",
        "candidate_action_do_nothing_head = "
        f"{str(use_do_nothing_head).lower()}",
    )
    text = replace_once(
        text,
        "share_actor_gnn = true",
        "share_actor_gnn = true\nshare_candidate_scorer = true",
    )
    text = replace_once(
        text,
        'gnn_graph_type = "heterogeneous_line"',
        'gnn_graph_type = "heterogeneous_line"\n'
        "# Retain the explicit connected indicator: 13 node features.\n"
        "gnn_include_legacy_connected_feature = true",
    )
    return name, text


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    expected = set()
    for preprocessing, pool, use_features, use_do_nothing_head in product(
        ("NL", "NLS"),
        (("mean", "mean"), ("tmean", "typed_mean")),
        (False, True),
        (False, True),
    ):
        pool_name, pool_mode = pool
        name, text = build_config(
            preprocessing,
            pool_name,
            pool_mode,
            use_features,
            use_do_nothing_head,
        )
        path = OUTPUT_DIR / f"{name}.toml"
        path.write_text(text)
        expected.add(path)

    for stale in OUTPUT_DIR.glob("*.toml"):
        if stale not in expected:
            stale.unlink()
    print(f"Generated {len(expected)} configs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
