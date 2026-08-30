#!/usr/bin/env python3
"""Generate the Chapter 9 encoder-only transfer study.

Four bus14 representations x four training routes. The action heads of Screens
C and D name one action per output row, so they cannot cross grids; this study
keeps the encoder, discards the heads, and learns new WCCI heads.

    route            encoder init                trainable on WCCI
    target scratch   random                      encoder + heads   <- the existing wsc campaign
    random frozen    random                      heads only
    NL frozen        bus14 NL checkpoint         heads only
    NLS frozen       bus14 NLS retrain           heads only

Everything is derived from `X_wcci_scratch_arch/wsc_*` so the four routes are
matched by construction: same environment, same 15M budget, same optimizer,
same chronic split, same encoder geometry. Only the encoder initialisation and
whether it is frozen differ. The wsc encoders are already byte-identical to
their bus14 source configs, which is what lets a source checkpoint load.

The NLS route needs bus14 sources that do not exist yet, so this also emits the
four retrains. Scaling is never switched at deployment: a frozen encoder must
receive the input convention its weights were learned under.

    python tools/generate_encoder_transfer_bridge_configs.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parents[1]
CONFIG_ROOT = TASK_DIR / "configs" / "no_leakage_config"
WSC_DIR = CONFIG_ROOT / "X_wcci_scratch_arch"
SOURCE_OUT = CONFIG_ROOT / "G_bus14_nls_sources"
BRIDGE_OUT = CONFIG_ROOT / "H_encoder_transfer_bridge"
CAPS = (64, 256)

# representation -> (wsc config stem without the cap, bus14 source config path,
#                    bus14 source run name)
REPRESENTATIONS = {
    "e0n0v0": ("wsc_bus_e0n0v0", "C_nl_s2em_structure", "nl_s2em_bus_n0_none_e0n0v0_s0"),
    "e1n0v0": ("wsc_bus_e1n0v0", "C_nl_s2em_structure", "nl_s2em_bus_n0_none_e1n0v0_s0"),
    "gbi_lbi": ("wsc_het_gbi_lbi", "D_nl_hmdem_message_direction",
                "nl_hmdem_hetero_n0_none_gbi_lbi_s0"),
    "ga2b_lb2a": ("wsc_het_ga2b_lb2a", "D_nl_hmdem_message_direction",
                  "nl_hmdem_hetero_n0_none_ga2b_lb2a_s0"),
}
NL_CHECKPOINT = {
    "C_nl_s2em_structure": "checkpoint/no_leak/nl_s2em/best_test_{run}.tar",
    "D_nl_hmdem_message_direction": "checkpoint/no_leak/nl_hmdem/best_test_{run}.tar",
}
# The NLS retrains have not run yet, so their weights land at the flat default
# checkpoint directory rather than in a curated subfolder.
NLS_CHECKPOINT = "checkpoint/best_test_nlsrc_{repr}_s0.tar"

FREEZE_BLOCK = """
# --- Chapter 9 encoder-only transfer -----------------------------------------
# The source action heads name one action per output row and cannot cross
# grids, so they are discarded: each WCCI agent gets a freshly initialised head
# and the centralized critic is target-specific and stays trainable.
{checkpoint_line}transfer_freeze_encoder = true
transfer_action_head = false
"""


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"Expected exactly one {old!r}, found {count}.")
    return text.replace(old, new, 1)


def set_nls_inputs(text: str) -> str:
    """Corrected inputs: physical scaling with the edge-difference angle."""
    text = replace_once(text, "gnn_physical_scaling = false",
                        'gnn_angle_representation = "edge_diff"\ngnn_physical_scaling = true')
    if "gnn_running_norm = false" not in text:
        raise ValueError("Expected gnn_running_norm = false alongside physical scaling")
    return text


def rename(text: str, source: str, target: str) -> str:
    text = replace_once(text, f'name = "{source}"', f'name = "{target}"')
    return replace_once(text, f'exp_tag = "{source}"', f'exp_tag = "{target}"')


def header(title: str, lines: list[str]) -> str:
    return "".join(f"# {line}\n" for line in [title, *lines]) + "\n"


def strip_header(text: str) -> str:
    split = text.find("\n\n")
    if split < 0:
        raise ValueError("config lacks a header paragraph")
    return text[split + 2:]


def build_nls_source(key: str) -> tuple[str, str]:
    _, screen, run = REPRESENTATIONS[key]
    path = CONFIG_ROOT / screen / f"{run}.toml"
    if not path.is_file():
        raise FileNotFoundError(f"Missing bus14 source config: {path}")
    target = f"nlsrc_{key}_s0"
    text = set_nls_inputs(rename(path.read_text(encoding="utf-8"), run, target))
    return target, header(
        f"bus14 NLS retrain of {key}, seed 0.",
        [f"Identical to {run}.toml except for the corrected encoder inputs",
         "(physical scaling + edge-difference angle). Supplies the NLS frozen",
         "route of Chapter 9; a frozen encoder must be fed the input convention",
         "its weights were learned under."],
    ) + strip_header(text)


def build_bridge(key: str, cap: int, route: str) -> tuple[str, str]:
    wsc_stem, screen, run = REPRESENTATIONS[key]
    path = WSC_DIR / f"{wsc_stem}_mk{cap}_s0.toml"
    if not path.is_file():
        raise FileNotFoundError(f"Missing wsc config: {path}")
    target = f"etb{cap}_{route}_{key}_s0"
    text = rename(path.read_text(encoding="utf-8"), f"{wsc_stem}_mk{cap}_s0", target)

    if route == "nlsfz":
        text = set_nls_inputs(text)
        checkpoint = NLS_CHECKPOINT.format(**{"repr": key})
        note = "bus14 NLS retrain (run G_bus14_nls_sources first)"
    elif route == "nlfz":
        checkpoint = NL_CHECKPOINT[screen].format(run=run)
        note = f"existing bus14 NL checkpoint from {run}"
    elif route == "rndfz":
        checkpoint = ""
        note = "no checkpoint: a randomly initialised encoder is frozen, the control"
    else:
        raise ValueError(route)

    checkpoint_line = (
        f'transfer_encoder_checkpoint = "{checkpoint}"\n' if checkpoint else ""
    )
    text = text.rstrip("\n") + "\n" + FREEZE_BLOCK.format(checkpoint_line=checkpoint_line)
    return target, header(
        f"Chapter 9 encoder-only transfer: {key}, mk{cap}, route={route}.",
        [f"Encoder initialisation: {note}.",
         f"Derived from {wsc_stem}_mk{cap}_s0.toml, which is the matched",
         "target-scratch route; only the encoder init and the freeze differ."],
    ) + strip_header(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    written: list[tuple[Path, str]] = []
    for key in REPRESENTATIONS:
        target, text = build_nls_source(key)
        written.append((SOURCE_OUT / f"{target}.toml", text))
    for key in REPRESENTATIONS:
        for cap in CAPS:
            for route in ("rndfz", "nlfz", "nlsfz"):
                target, text = build_bridge(key, cap, route)
                written.append((BRIDGE_OUT / f"{target}.toml", text))

    for path, text in written:
        if not args.dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
    verb = "Would write" if args.dry_run else "Wrote"
    print(f"{verb} {len(written)} config(s):")
    for folder in (SOURCE_OUT, BRIDGE_OUT):
        names = sorted(p.name for p, _ in written if p.parent == folder)
        print(f"  {folder.relative_to(TASK_DIR)}  ({len(names)})")


if __name__ == "__main__":
    main()
