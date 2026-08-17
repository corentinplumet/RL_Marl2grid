#!/usr/bin/env python3
"""Generate the complete shared-candidate WCCI full-test analysis notebook."""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent

import nbformat as nbf


TASK_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    TASK_DIR
    / "analysis"
    / "metrics"
    / "notebooks"
    / "episodic_survival"
    / "shared_wcci_complete_full_test_analysis.ipynb"
)


def markdown(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip() + "\n")


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip() + "\n")


def build_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    }
    nb.cells = [
        markdown(
            r"""
            # Complete shared-candidate WCCI full-test analysis

            This notebook inventories every full-test result currently downloaded
            under `outputs/full_test_eval/shared` and combines all comparable
            **50-chronic `bus36_wcci_nomaint` evaluations**:

            - 80 original zero-shot runs across NL/NLS and action caps
              64/128/256/512/1024;
            - eight early WCCI fine-tuned mk256 checkpoints;
            - eight evaluation-only rho-gating runs on fine-tuned checkpoints;
            - all downloaded evaluation-only local-rho screens on original
              checkpoints, including the selective stage-2 threshold and
              architecture study;
            - the target-trained WCCI MLP comparator when available.

            The 201-episode bus14 source-grid evaluations are inventoried and
            reported separately. They are not mixed with WCCI rankings because
            the environments and chronic sets differ.

            **Important:** the same 50 WCCI test chronics have now been inspected
            repeatedly while choosing action caps and rho thresholds. Results are
            therefore exploratory model-selection evidence, not an unbiased final
            test estimate. A final claim needs threshold selection on validation
            chronics and evaluation on a fresh held-out set.
            """
        ),
        code(
            """
            from pathlib import Path
            import json
            import re

            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns
            from IPython.display import Markdown, display

            pd.set_option("display.max_columns", 100)
            pd.set_option("display.max_rows", 160)
            pd.set_option("display.float_format", lambda value: f"{value:,.3f}")
            sns.set_theme(style="whitegrid", context="notebook")


            def find_task_dir(start=Path.cwd()):
                start = Path(start).resolve()
                for candidate in [start, *start.parents]:
                    direct = candidate if candidate.name == "Topology_Task" else candidate / "Topology_Task"
                    if (direct / "main.py").is_file():
                        return direct
                raise FileNotFoundError("Could not locate Topology_Task")


            def load_json(path):
                with Path(path).open(encoding="utf-8") as handle:
                    return json.load(handle)


            def local_artifact_path(path_string):
                path = Path(path_string)
                if path.is_file():
                    return path.resolve()
                if path.parts and path.parts[0] == "outputs":
                    return TASK_DIR / path
                if "Topology_Task" in path.parts:
                    index = path.parts.index("Topology_Task")
                    return TASK_DIR.joinpath(*path.parts[index + 1 :])
                return TASK_DIR / path


            def save_figure(fig, stem):
                path = EXPORT_DIR / f"{stem}.png"
                fig.savefig(path, dpi=180, bbox_inches="tight")
                return path


            TASK_DIR = find_task_dir()
            SHARED_RESULT_ROOT = TASK_DIR / "outputs" / "full_test_eval" / "shared"
            ACTION_ROOT = TASK_DIR / "outputs" / "full_test_eval_actions" / "shared"
            DO_NOTHING_PATH = (
                TASK_DIR / "outputs" / "do_nothing_eval" / "wcci_test_nomaint_50"
                / "do_nothing_summary.json"
            )
            GREEDY_RESULT_ROOT = (
                TASK_DIR / "outputs" / "greedy_wcci_nomaint_50"
            )
            EXPORT_DIR = TASK_DIR / "outputs" / "analysis" / "shared_wcci_complete_full_test"
            EXPORT_DIR.mkdir(parents=True, exist_ok=True)

            ACTION_ORDER = [64, 128, 256, 512, 1024]
            GREEDY_ACTION_ORDER = [32, 64, 128, 256, 512, 1024]
            CAMPAIGN_ORDER = [
                "Original ungated",
                "Original rho-gated",
                "Fine-tuned ungated",
                "Fine-tuned rho-gated",
                "Target-trained MLP",
            ]
            CAMPAIGN_COLORS = {
                "Original ungated": "#4C78A8",
                "Original rho-gated": "#72B7B2",
                "Fine-tuned ungated": "#F58518",
                "Fine-tuned rho-gated": "#E45756",
                "Target-trained MLP": "#79706E",
            }

            print("Shared result directory:", SHARED_RESULT_ROOT)
            print("Analysis exports:", EXPORT_DIR)
            """
        ),
        markdown(
            """
            ## 1. Discover and classify every downloaded shared result

            Classification uses result metadata and checkpoint names rather than
            relying only on directory names. Consequently, rerunning this notebook
            will automatically include newly downloaded result groups.
            """
        ),
        code(
            """
            VARIANT_RE = re.compile(r"_(?P<pool>mean|tmean)_f(?P<features>[01])_a0h(?P<head>[01])")
            ACTION_RE = re.compile(r"_mk(?P<size>\\d+)")


            def parse_result(result_path, shared=True):
                payload = load_json(result_path)
                if "survival_percent" not in payload:
                    return None
                checkpoint_stem = Path(payload.get("checkpoint", result_path.stem)).stem
                variant_match = VARIANT_RE.search(checkpoint_stem)
                pool = variant_match.group("pool") if variant_match else "mlp"
                features = int(variant_match.group("features")) if variant_match else np.nan
                head = int(variant_match.group("head")) if variant_match else np.nan
                if "_NLS_" in checkpoint_stem:
                    family = "NLS"
                elif "_NL_" in checkpoint_stem:
                    family = "NL"
                else:
                    family = "MLP"
                training_regime = "fine-tuned" if "finetune" in checkpoint_stem else "original"
                heuristic_raw = str(payload.get("eval_action_heuristic", "none") or "none")
                heuristic = {
                    "none": "none",
                    "rho_threshold": "global rho",
                    "local_rho_threshold": "local rho",
                }.get(heuristic_raw, heuristic_raw)
                target_action_path = str(payload.get("target_reduced_action_space", "") or "")
                action_match = ACTION_RE.search(target_action_path)
                if action_match:
                    action_size = int(action_match.group("size"))
                else:
                    group_name = result_path.parent.name
                    group_match = ACTION_RE.search(group_name)
                    action_size = int(group_match.group("size")) if group_match else np.nan

                target_env = str(payload.get("target_env_id", "") or "")
                source_env = str(payload.get("source_env_id", "") or "")
                eval_episodes = int(payload.get("eval_episodes", -1))
                is_wcci = target_env == "bus36_wcci_nomaint" and eval_episodes == 50
                if family == "MLP":
                    campaign = "Target-trained MLP"
                elif training_regime == "fine-tuned" and heuristic != "none":
                    campaign = "Fine-tuned rho-gated"
                elif training_regime == "fine-tuned":
                    campaign = "Fine-tuned ungated"
                elif heuristic != "none":
                    campaign = "Original rho-gated"
                else:
                    campaign = "Original ungated"

                variant = (
                    f"{family}_{pool}_f{int(features)}_a0h{int(head)}"
                    if variant_match else family
                )
                relative = (
                    str(result_path.relative_to(SHARED_RESULT_ROOT))
                    if shared else result_path.name
                )
                return {
                    "result_key": relative,
                    "result_path": str(result_path),
                    "result_group": result_path.parent.name if shared else "root comparator",
                    "checkpoint": str(payload.get("checkpoint", "")),
                    "checkpoint_stem": checkpoint_stem,
                    "checkpoint_global_step": payload.get("checkpoint_global_step"),
                    "family": family,
                    "pool": pool,
                    "features": features,
                    "head": head,
                    "variant": variant,
                    "training_regime": training_regime,
                    "campaign": campaign,
                    "heuristic": heuristic,
                    "heuristic_raw": heuristic_raw,
                    "rho_threshold": (
                        float(payload.get("eval_action_rho_threshold", np.nan))
                        if heuristic != "none" else np.nan
                    ),
                    "action_size": action_size,
                    "source_env_id": source_env,
                    "target_env_id": target_env,
                    "transfer_mode": payload.get("transfer_mode"),
                    "eval_episodes": eval_episodes,
                    "obs_normalization": payload.get("obs_normalization"),
                    "survival_percent_reported": float(payload["survival_percent"]),
                    "is_wcci_50": is_wcci,
                    "action_artifacts": payload.get("action_artifacts") or {},
                }


            records = []
            for result_path in sorted(SHARED_RESULT_ROOT.rglob("*.json")):
                record = parse_result(result_path)
                if record is not None:
                    records.append(record)

            mlp_result = (
                TASK_DIR / "outputs" / "full_test_eval"
                / "best_test_wcci_reduced_mlp_baseline_72x576_s0_step43628544_job3086193.json"
            )
            if mlp_result.is_file():
                record = parse_result(mlp_result, shared=False)
                record["family"] = "MLP"
                record["variant"] = "target_trained_mlp"
                record["training_regime"] = "target-trained"
                record["campaign"] = "Target-trained MLP"
                record["is_wcci_50"] = True
                records.append(record)

            inventory = pd.DataFrame(records)
            assert inventory["result_key"].is_unique
            shared_inventory = inventory.loc[inventory["result_group"] != "root comparator"]
            wcci_inventory = inventory.loc[inventory["is_wcci_50"]].copy()
            source_inventory = inventory.loc[~inventory["is_wcci_50"]].copy()

            display(pd.DataFrame({
                "scope": ["All shared JSONs", "Comparable WCCI 50-chronic", "Source-grid/non-WCCI", "External comparator"],
                "results": [
                    len(shared_inventory),
                    int(shared_inventory["is_wcci_50"].sum()),
                    int((~shared_inventory["is_wcci_50"]).sum()),
                    int((inventory["result_group"] == "root comparator").sum()),
                ],
            }))
            display(
                shared_inventory.groupby(["result_group", "is_wcci_50"], as_index=False)
                .size().rename(columns={"size": "results"})
            )
            """
        ),
        markdown(
            """
            ## 2. Join all WCCI episodes to the same do-nothing baseline

            Every comparable WCCI result must have exactly 50 episode rows and a
            one-to-one fingerprint match to the baseline. This prevents accidental
            comparisons between different chronic selections.
            """
        ),
        code(
            """
            baseline_payload = load_json(DO_NOTHING_PATH)
            baseline = pd.DataFrame(baseline_payload["episodes"])[[
                "chronic_fingerprint", "chronic_name", "survival", "steps",
                "max_steps", "full_survival",
            ]].rename(columns={
                "survival": "do_nothing_survival",
                "steps": "do_nothing_steps",
                "full_survival": "do_nothing_full_survival",
            })
            assert len(baseline) == 50
            assert baseline["chronic_fingerprint"].is_unique
            baseline["cohort"] = np.where(
                baseline["do_nothing_full_survival"],
                "Do-nothing = 100%",
                "Do-nothing < 100%",
            )

            episode_frames = []
            agent_rows = []
            missing_artifacts = []
            for row in wcci_inventory.itertuples(index=False):
                artifacts = row.action_artifacts
                episode_string = artifacts.get("episode_summary_csv")
                summary_string = artifacts.get("action_summary_json")
                if not episode_string or not summary_string:
                    missing_artifacts.append((row.result_key, "artifact path absent"))
                    continue
                episode_path = local_artifact_path(episode_string)
                summary_path = local_artifact_path(summary_string)
                if not episode_path.is_file() or not summary_path.is_file():
                    missing_artifacts.append((row.result_key, str(episode_path)))
                    continue

                episodes = pd.read_csv(episode_path).merge(
                    baseline, on="chronic_fingerprint", validate="one_to_one",
                    suffixes=("", "_baseline"),
                )
                if len(episodes) != 50:
                    raise ValueError(f"Expected 50 episodes for {row.result_key}; got {len(episodes)}")
                episodes["result_key"] = row.result_key
                episodes["model_full_survival"] = np.isclose(episodes["survival"], 1.0)
                episodes["delta_vs_do_nothing_pp"] = 100 * (
                    episodes["survival"] - episodes["do_nothing_survival"]
                )
                episode_frames.append(episodes)

                action_summary = load_json(summary_path)
                for agent_id, values in sorted(action_summary["agents"].items()):
                    n_steps = int(values["n_steps"])
                    policy_nonidle = sum(
                        int(count)
                        for action_id, count in values["policy_action_counts"].items()
                        if int(action_id) != 0
                    )
                    agent_rows.append({
                        "result_key": row.result_key,
                        "agent_id": agent_id,
                        "n_steps": n_steps,
                        "executed_nonidle_rate": int(values["nonidle_count"]) / n_steps,
                        "policy_nonidle_rate": policy_nonidle / n_steps,
                        "heuristic_blocked_nonidle_rate": (
                            int(values.get("heuristic_blocked_nonidle_count", 0)) / n_steps
                        ),
                        "heuristic_force_noop_rate": (
                            int(values.get("heuristic_force_noop_count", 0)) / n_steps
                        ),
                    })

            if missing_artifacts:
                raise FileNotFoundError(
                    "Missing WCCI action artifacts:\\n" + "\\n".join(map(str, missing_artifacts))
                )

            episodes = pd.concat(episode_frames, ignore_index=True)
            agent_behavior = pd.DataFrame(agent_rows)
            assert episodes["result_key"].nunique() == len(wcci_inventory)
            assert len(episodes) == 50 * len(wcci_inventory)
            assert len(agent_behavior) == 4 * len(wcci_inventory)

            metadata_columns = [
                "result_key", "result_group", "checkpoint_stem", "checkpoint_global_step",
                "family", "pool", "features", "head", "variant", "training_regime",
                "campaign", "heuristic", "rho_threshold", "action_size",
                "source_env_id", "target_env_id", "transfer_mode", "obs_normalization",
                "survival_percent_reported",
            ]
            episodes = episodes.merge(
                wcci_inventory[metadata_columns], on="result_key", validate="many_to_one"
            )
            agent_behavior = agent_behavior.merge(
                wcci_inventory[metadata_columns], on="result_key", validate="many_to_one"
            )

            display(Markdown(
                f"Loaded **{episodes['result_key'].nunique()} WCCI evaluations × 50 chronics "
                f"= {len(episodes):,} model episodes**. The baseline defines "
                f"**{int((~baseline['do_nothing_full_survival']).sum())} difficult** and "
                f"**{int(baseline['do_nothing_full_survival'].sum())} easy** chronics."
            ))
            """
        ),
        code(
            """
            def summarize_result(group):
                hard = group.loc[group["cohort"] == "Do-nothing < 100%"]
                easy = group.loc[group["cohort"] == "Do-nothing = 100%"]
                nonrescue = hard.loc[~hard["model_full_survival"]]
                return pd.Series({
                    "overall_mean_survival_percent": 100 * group["survival"].mean(),
                    "overall_median_survival_percent": 100 * group["survival"].median(),
                    "easy_mean_survival_percent": 100 * easy["survival"].mean(),
                    "easy_preservation_rate": easy["model_full_survival"].mean(),
                    "difficult_mean_survival_percent": 100 * hard["survival"].mean(),
                    "difficult_median_survival_percent": 100 * hard["survival"].median(),
                    "difficult_mean_delta_pp": hard["delta_vs_do_nothing_pp"].mean(),
                    "difficult_paired_median_delta_pp": hard["delta_vs_do_nothing_pp"].median(),
                    "difficult_win_rate": (hard["delta_vs_do_nothing_pp"] > 1e-9).mean(),
                    "difficult_tie_rate": (hard["delta_vs_do_nothing_pp"].abs() <= 1e-9).mean(),
                    "difficult_loss_rate": (hard["delta_vs_do_nothing_pp"] < -1e-9).mean(),
                    "difficult_rescues": int(hard["model_full_survival"].sum()),
                    "difficult_nonrescue_mean_delta_pp": (
                        nonrescue["delta_vs_do_nothing_pp"].mean() if len(nonrescue) else np.nan
                    ),
                })


            run_metrics = episodes.groupby("result_key", sort=False).apply(
                summarize_result, include_groups=False
            ).reset_index()
            run_metrics = run_metrics.merge(
                wcci_inventory[metadata_columns], on="result_key", validate="one_to_one"
            )
            run_behavior = (
                agent_behavior.groupby("result_key", as_index=False)
                .agg(
                    mean_executed_nonidle_rate=("executed_nonidle_rate", "mean"),
                    max_executed_nonidle_rate=("executed_nonidle_rate", "max"),
                    mean_policy_nonidle_rate=("policy_nonidle_rate", "mean"),
                    max_policy_nonidle_rate=("policy_nonidle_rate", "max"),
                    mean_blocked_nonidle_rate=("heuristic_blocked_nonidle_rate", "mean"),
                    mean_force_noop_rate=("heuristic_force_noop_rate", "mean"),
                )
            )
            agent3 = agent_behavior.loc[agent_behavior["agent_id"] == "agent_3", [
                "result_key", "executed_nonidle_rate", "policy_nonidle_rate"
            ]].rename(columns={
                "executed_nonidle_rate": "agent3_executed_nonidle_rate",
                "policy_nonidle_rate": "agent3_policy_nonidle_rate",
            })
            run_behavior = run_behavior.merge(agent3, on="result_key", validate="one_to_one")
            run_metrics = run_metrics.merge(run_behavior, on="result_key", validate="one_to_one")
            run_metrics["overall_delta_vs_do_nothing_pp"] = (
                run_metrics["overall_mean_survival_percent"]
                - 100 * baseline["do_nothing_survival"].mean()
            )
            run_metrics["rank_overall"] = run_metrics["overall_mean_survival_percent"].rank(
                method="min", ascending=False
            ).astype(int)
            run_metrics = run_metrics.sort_values("rank_overall").reset_index(drop=True)

            assert np.allclose(
                run_metrics["overall_mean_survival_percent"],
                run_metrics["survival_percent_reported"],
            )
            """
        ),
        markdown(
            """
            ## 3. Complete inventory and source-grid context

            The source-grid table answers whether a checkpoint learned the bus14
            task. It does not measure transfer. High source performance alongside
            poor WCCI performance is evidence of transfer failure rather than
            failed source training.
            """
        ),
        code(
            """
            inventory_table = (
                wcci_inventory.groupby(
                    ["campaign", "family", "action_size", "heuristic"],
                    dropna=False, as_index=False,
                )
                .size().rename(columns={"size": "results"})
                .sort_values(["campaign", "family", "action_size", "heuristic"])
            )
            display(inventory_table)

            source_results = source_inventory[[
                "result_group", "checkpoint_stem", "family", "pool", "features", "head",
                "eval_episodes", "survival_percent_reported",
            ]].sort_values(["result_group", "survival_percent_reported"], ascending=[True, False])
            display(source_results.groupby("result_group", as_index=False).agg(
                evaluations=("checkpoint_stem", "size"),
                episodes_per_evaluation=("eval_episodes", "first"),
                mean_survival_percent=("survival_percent_reported", "mean"),
                best_survival_percent=("survival_percent_reported", "max"),
            ))
            display(source_results.groupby("result_group", as_index=False).head(3))
            """
        ),
        markdown(
            """
            ## 4. WCCI leaderboard

            Overall mean alone is insufficient. The leaderboard includes easy
            preservation, difficult median, rescue count, and the non-rescue
            difficult delta to reveal results dominated by exceptional rescues.
            """
        ),
        code(
            """
            leaderboard_columns = [
                "rank_overall", "campaign", "family", "variant", "action_size",
                "heuristic", "rho_threshold", "checkpoint_global_step",
                "overall_mean_survival_percent", "overall_delta_vs_do_nothing_pp",
                "easy_preservation_rate", "difficult_mean_survival_percent",
                "difficult_median_survival_percent", "difficult_win_rate",
                "difficult_loss_rate", "difficult_rescues",
                "difficult_nonrescue_mean_delta_pp", "mean_executed_nonidle_rate",
                "mean_policy_nonidle_rate",
            ]
            display(run_metrics[leaderboard_columns].head(20))

            campaign_summary = (
                run_metrics.groupby("campaign", as_index=False)
                .agg(
                    evaluations=("result_key", "size"),
                    mean_overall=("overall_mean_survival_percent", "mean"),
                    median_overall=("overall_mean_survival_percent", "median"),
                    best_overall=("overall_mean_survival_percent", "max"),
                    best_difficult_mean=("difficult_mean_survival_percent", "max"),
                    best_difficult_median=("difficult_median_survival_percent", "max"),
                )
            )
            campaign_summary["campaign"] = pd.Categorical(
                campaign_summary["campaign"], categories=CAMPAIGN_ORDER, ordered=True
            )
            campaign_summary = campaign_summary.sort_values("campaign")
            display(campaign_summary)

            fig, ax = plt.subplots(figsize=(11.5, 5.7))
            plot_order = [campaign for campaign in CAMPAIGN_ORDER if campaign in set(run_metrics["campaign"])]
            sns.boxplot(
                data=run_metrics, x="campaign", y="overall_mean_survival_percent",
                order=plot_order, hue="campaign", palette=CAMPAIGN_COLORS,
                legend=False, showfliers=False, ax=ax,
            )
            sns.stripplot(
                data=run_metrics, x="campaign", y="overall_mean_survival_percent",
                order=plot_order, color="#222222", alpha=0.5, size=4, jitter=0.18, ax=ax,
            )
            ax.axhline(100 * baseline["do_nothing_survival"].mean(), color="black", linestyle="--", linewidth=1.2, label="Do nothing")
            ax.set(
                xlabel="Campaign",
                ylabel="Mean survival (%)",
                title="All comparable WCCI full-test evaluations",
            )
            ax.tick_params(axis="x", rotation=18)
            ax.legend(loc="lower right")
            save_figure(fig, "all_campaign_overview")
            plt.show()
            """
        ),
        markdown(
            """
            ## 5. Original zero-shot action-space sweep

            These 80 ungated runs isolate the action cap and NL/NLS architecture
            factors before fine-tuning or intervention heuristics.
            """
        ),
        code(
            """
            zero_shot = run_metrics.loc[
                (run_metrics["campaign"] == "Original ungated")
                & run_metrics["action_size"].isin(ACTION_ORDER)
            ].copy()
            zero_shot_cap = (
                zero_shot.groupby(["family", "action_size"], as_index=False)
                .agg(
                    models=("result_key", "size"),
                    mean_overall=("overall_mean_survival_percent", "mean"),
                    median_overall=("overall_mean_survival_percent", "median"),
                    best_overall=("overall_mean_survival_percent", "max"),
                    mean_nonidle=("mean_executed_nonidle_rate", "mean"),
                )
            )
            display(zero_shot_cap)

            fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
            for family, group in zero_shot_cap.groupby("family"):
                group = group.sort_values("action_size")
                axes[0].plot(group["action_size"], group["mean_overall"], marker="o", linewidth=2, label=family)
                axes[1].plot(group["action_size"], 100 * group["mean_nonidle"], marker="o", linewidth=2, label=family)
            for ax in axes:
                ax.set_xscale("log", base=2)
                ax.set_xticks(ACTION_ORDER, ACTION_ORDER)
                ax.set_xlabel("Nominal candidate cap")
                ax.legend(title="Family")
            axes[0].axhline(100 * baseline["do_nothing_survival"].mean(), color="black", linestyle="--", linewidth=1)
            axes[0].set_ylabel("Mean survival across architecture variants (%)")
            axes[0].set_title("Performance falls as the candidate set grows")
            axes[1].set_ylabel("Mean executed non-idle rate (%)")
            axes[1].set_title("Larger sets encourage more intervention")
            fig.tight_layout()
            save_figure(fig, "zero_shot_action_cap")
            plt.show()
            """
        ),
        markdown(
            """
            ## 6. Early fine-tuning versus matched original mk256 models

            Each line connects the same architecture variant before and after
            early WCCI fine-tuning. The checkpoints were selected very early
            (roughly 0.17M–2.99M steps), so this is evidence of rapid adaptation,
            not a converged fine-tuning comparison.
            """
        ),
        code(
            """
            original_256 = run_metrics.loc[
                (run_metrics["campaign"] == "Original ungated")
                & (run_metrics["family"] == "NLS")
                & (run_metrics["action_size"] == 256),
                ["variant", "overall_mean_survival_percent", "difficult_mean_survival_percent",
                 "easy_preservation_rate", "difficult_nonrescue_mean_delta_pp"]
            ].rename(columns={
                "overall_mean_survival_percent": "original_overall",
                "difficult_mean_survival_percent": "original_difficult_mean",
                "easy_preservation_rate": "original_easy_preservation",
                "difficult_nonrescue_mean_delta_pp": "original_nonrescue_delta",
            })
            fine_ungated = run_metrics.loc[
                run_metrics["campaign"] == "Fine-tuned ungated",
                ["variant", "checkpoint_global_step", "overall_mean_survival_percent",
                 "difficult_mean_survival_percent", "easy_preservation_rate",
                 "difficult_nonrescue_mean_delta_pp"]
            ].rename(columns={
                "overall_mean_survival_percent": "finetuned_overall",
                "difficult_mean_survival_percent": "finetuned_difficult_mean",
                "easy_preservation_rate": "finetuned_easy_preservation",
                "difficult_nonrescue_mean_delta_pp": "finetuned_nonrescue_delta",
            })
            matched_finetuning = fine_ungated.merge(original_256, on="variant", validate="one_to_one")
            matched_finetuning["overall_gain_pp"] = (
                matched_finetuning["finetuned_overall"] - matched_finetuning["original_overall"]
            )
            matched_finetuning = matched_finetuning.sort_values("finetuned_overall", ascending=False)
            display(matched_finetuning)

            long_ft = matched_finetuning.melt(
                id_vars="variant", value_vars=["original_overall", "finetuned_overall"],
                var_name="regime", value_name="overall",
            )
            fig, ax = plt.subplots(figsize=(10.5, 6.0))
            xmap = {"original_overall": 0, "finetuned_overall": 1}
            for variant, group in long_ft.groupby("variant"):
                group = group.assign(x=group["regime"].map(xmap)).sort_values("x")
                ax.plot(group["x"], group["overall"], marker="o", alpha=0.75, linewidth=1.5)
                ax.text(1.03, group.iloc[-1]["overall"], variant.replace("NLS_", ""), va="center", fontsize=8)
            ax.axhline(100 * baseline["do_nothing_survival"].mean(), color="black", linestyle="--", linewidth=1)
            ax.set_xticks([0, 1], ["Original zero-shot", "Early fine-tuned"])
            ax.set_xlim(-0.1, 1.48)
            ax.set_ylabel("Mean survival (%)")
            ax.set_title("Matched mk256 architectures: early fine-tuning repairs seven of eight variants")
            save_figure(fig, "matched_early_finetuning")
            plt.show()
            """
        ),
        markdown(
            """
            ## 7. Evaluation-only rho gating on fine-tuned checkpoints

            Policy non-idle rate is measured before the heuristic override;
            executed non-idle rate is measured after it. Their separation shows
            how much intervention timing is supplied by the heuristic rather than
            learned by the actor.
            """
        ),
        code(
            """
            ft_gate = run_metrics.loc[
                run_metrics["campaign"] == "Fine-tuned rho-gated"
            ].copy().sort_values(["variant", "heuristic", "rho_threshold"])
            display(ft_gate[leaderboard_columns])

            ft_reference = run_metrics.loc[
                run_metrics["campaign"] == "Fine-tuned ungated",
                ["variant", "overall_mean_survival_percent"]
            ].set_index("variant")["overall_mean_survival_percent"]

            fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.9))
            for (variant, heuristic), group in ft_gate.groupby(["variant", "heuristic"]):
                group = group.sort_values("rho_threshold")
                label = f"{variant.replace('NLS_', '')} · {heuristic}"
                axes[0].plot(group["rho_threshold"], group["overall_mean_survival_percent"], marker="o", linewidth=2, label=label)
                axes[1].plot(group["rho_threshold"], group["difficult_nonrescue_mean_delta_pp"], marker="o", linewidth=2, label=label)
            axes[0].set_ylabel("Mean survival (%)")
            axes[0].set_title("Fine-tuned gated performance")
            axes[1].axhline(0, color="black", linewidth=1)
            axes[1].set_ylabel("Difficult non-rescue delta (pp)")
            axes[1].set_title("Positive values indicate broad benefit beyond rescues")
            for ax in axes:
                ax.set_xlabel("Rho threshold")
                ax.legend(fontsize=8)
            fig.tight_layout()
            save_figure(fig, "finetuned_rho_thresholds")
            plt.show()

            fig, ax = plt.subplots(figsize=(8.5, 6.0))
            ax.scatter(
                100 * ft_gate["mean_policy_nonidle_rate"],
                100 * ft_gate["mean_executed_nonidle_rate"],
                c=ft_gate["rho_threshold"], cmap="viridis", s=95,
                edgecolor="white", linewidth=0.8,
            )
            for row in ft_gate.itertuples():
                ax.annotate(
                    f"{row.variant.replace('NLS_', '')} {row.heuristic} {row.rho_threshold:.2f}",
                    (100 * row.mean_policy_nonidle_rate, 100 * row.mean_executed_nonidle_rate),
                    xytext=(5, 4), textcoords="offset points", fontsize=7,
                )
            ax.set(
                xlabel="Policy-requested non-idle rate (%)",
                ylabel="Executed non-idle rate after gating (%)",
                title="Rho heuristics suppress most requested interventions",
            )
            save_figure(fig, "finetuned_policy_vs_executed_actions")
            plt.show()
            """
        ),
        markdown(
            """
            ## 8. Local rho gating on original bus14 weights

            The initial mk64/mk256 threshold sweep and the later selective
            architecture screen ask which transferred policies benefit from an
            evaluation-only local safety gate. Rows with a single threshold are
            screening points; rows with several thresholds show sensitivity.
            """
        ),
        code(
            """
            original_gate = run_metrics.loc[
                run_metrics["campaign"] == "Original rho-gated"
            ].copy().sort_values(["variant", "action_size", "rho_threshold"])
            display(original_gate[leaderboard_columns])

            gated_variants = list(original_gate["variant"].drop_duplicates())
            n_gated_variants = max(len(gated_variants), 1)
            fig, axes = plt.subplots(
                n_gated_variants,
                2,
                figsize=(13.5, max(4.2, 3.7 * n_gated_variants)),
                sharex=True,
                squeeze=False,
            )
            for row_index, variant in enumerate(gated_variants):
                subset = original_gate.loc[original_gate["variant"] == variant]
                for action_size, group in subset.groupby("action_size"):
                    group = group.sort_values("rho_threshold")
                    label = f"mk{int(action_size)}"
                    axes[row_index, 0].plot(group["rho_threshold"], group["overall_mean_survival_percent"], marker="o", linewidth=2, label=label)
                    axes[row_index, 1].plot(group["rho_threshold"], group["difficult_nonrescue_mean_delta_pp"], marker="o", linewidth=2, label=label)
                axes[row_index, 0].set_ylabel("Mean survival (%)")
                axes[row_index, 0].set_title(variant.replace("NLS_", "") + " · overall")
                axes[row_index, 1].axhline(0, color="black", linewidth=1)
                axes[row_index, 1].set_ylabel("Non-rescue delta (pp)")
                axes[row_index, 1].set_title(variant.replace("NLS_", "") + " · difficult robustness")
                axes[row_index, 0].legend()
                axes[row_index, 1].legend()
            for ax in axes[-1, :]:
                ax.set_xlabel("Local rho threshold")
            fig.tight_layout()
            save_figure(fig, "original_rho_cap_sweep")
            plt.show()
            """
        ),
        markdown(
            """
            ### Stage-2 matched results and threshold sensitivity

            The first table compares every newly downloaded stage-2 gated run
            with the ungated evaluation of the same architecture and action cap.
            The second table uses every available original local-rho evaluation
            and reports how much each completed threshold curve varies. A small
            range means that this 50-chronic set cannot meaningfully distinguish
            the tested thresholds.
            """
        ),
        code(
            """
            stage2_gate = original_gate.loc[
                original_gate["result_group"] == "origin_localrho_stage2"
            ].copy()
            ungated_reference = run_metrics.loc[
                run_metrics["campaign"] == "Original ungated",
                [
                    "variant", "action_size", "overall_mean_survival_percent",
                    "easy_preservation_rate", "difficult_mean_survival_percent",
                    "difficult_median_survival_percent",
                    "difficult_nonrescue_mean_delta_pp",
                ],
            ].rename(columns={
                "overall_mean_survival_percent": "ungated_overall_mean_survival_percent",
                "easy_preservation_rate": "ungated_easy_preservation_rate",
                "difficult_mean_survival_percent": "ungated_difficult_mean_survival_percent",
                "difficult_median_survival_percent": "ungated_difficult_median_survival_percent",
                "difficult_nonrescue_mean_delta_pp": "ungated_difficult_nonrescue_delta_pp",
            })
            stage2_matched = stage2_gate.merge(
                ungated_reference,
                on=["variant", "action_size"],
                how="left",
                validate="many_to_one",
            )
            stage2_matched["gain_vs_matched_ungated_pp"] = (
                stage2_matched["overall_mean_survival_percent"]
                - stage2_matched["ungated_overall_mean_survival_percent"]
            )
            stage2_columns = [
                "variant", "action_size", "rho_threshold",
                "overall_mean_survival_percent", "gain_vs_matched_ungated_pp",
                "easy_preservation_rate", "difficult_mean_survival_percent",
                "difficult_median_survival_percent", "difficult_rescues",
                "difficult_nonrescue_mean_delta_pp", "mean_policy_nonidle_rate",
                "mean_executed_nonidle_rate",
            ]
            display(
                stage2_matched[stage2_columns]
                .sort_values(
                    ["overall_mean_survival_percent", "variant", "rho_threshold"],
                    ascending=[False, True, True],
                )
            )


            def summarize_threshold_curve(group):
                ordered = group.sort_values(
                    ["overall_mean_survival_percent", "rho_threshold"],
                    ascending=[False, True],
                )
                best = ordered.iloc[0]
                return pd.Series({
                    "thresholds_tested": group["rho_threshold"].nunique(),
                    "rho_min": group["rho_threshold"].min(),
                    "rho_max": group["rho_threshold"].max(),
                    "best_rho_on_this_test_set": best["rho_threshold"],
                    "best_overall_mean_survival_percent": best["overall_mean_survival_percent"],
                    "worst_overall_mean_survival_percent": group["overall_mean_survival_percent"].min(),
                    "overall_range_pp": (
                        group["overall_mean_survival_percent"].max()
                        - group["overall_mean_survival_percent"].min()
                    ),
                    "minimum_easy_preservation_rate": group["easy_preservation_rate"].min(),
                    "maximum_difficult_rescues": group["difficult_rescues"].max(),
                    "best_nonrescue_delta_pp": group["difficult_nonrescue_mean_delta_pp"].max(),
                })


            threshold_sensitivity = (
                original_gate.groupby(["variant", "action_size"], as_index=False)
                .apply(summarize_threshold_curve, include_groups=False)
                .reset_index(drop=True)
            )
            threshold_sensitivity = threshold_sensitivity.loc[
                threshold_sensitivity["thresholds_tested"] >= 2
            ].sort_values(
                ["best_overall_mean_survival_percent", "variant"],
                ascending=[False, True],
            )
            display(threshold_sensitivity)
            """
        ),
        markdown(
            """
            ## 9. Best systems and chronic-level anatomy

            The heatmap shows paired survival deltas on the 24 difficult
            chronics. Large isolated positive cells identify rescue-driven means;
            modest positive values across many columns indicate broader benefit.
            """
        ),
        code(
            """
            def best_key(campaign):
                return run_metrics.loc[
                    run_metrics["campaign"] == campaign
                ].sort_values(
                    ["overall_mean_survival_percent", "action_size", "result_key"],
                    ascending=[False, True, True],
                    na_position="last",
                ).iloc[0]["result_key"]


            selected_keys = {
                "Best original ungated": best_key("Original ungated"),
                "Best original gated": best_key("Original rho-gated"),
                "Best fine-tuned ungated": best_key("Fine-tuned ungated"),
                "Best fine-tuned gated": best_key("Fine-tuned rho-gated"),
            }
            if "Target-trained MLP" in set(run_metrics["campaign"]):
                selected_keys["Target-trained MLP"] = best_key("Target-trained MLP")

            selected_comparison = run_metrics.set_index("result_key").loc[
                list(selected_keys.values())
            ].reset_index()
            selected_comparison.insert(0, "selection", list(selected_keys.keys()))
            display(selected_comparison[["selection", *leaderboard_columns[1:]]])

            hard_episodes = episodes.loc[episodes["cohort"] == "Do-nothing < 100%"]
            heatmap_rows = []
            for selection, key in selected_keys.items():
                frame = hard_episodes.loc[hard_episodes["result_key"] == key, [
                    "chronic_name", "delta_vs_do_nothing_pp"
                ]].copy()
                frame["selection"] = selection
                heatmap_rows.append(frame)
            heatmap_long = pd.concat(heatmap_rows, ignore_index=True)
            chronic_order = (
                baseline.loc[~baseline["do_nothing_full_survival"]]
                .sort_values("do_nothing_survival")["chronic_name"].tolist()
            )
            heatmap = heatmap_long.pivot(index="selection", columns="chronic_name", values="delta_vs_do_nothing_pp")
            heatmap = heatmap.reindex(index=list(selected_keys), columns=chronic_order)
            fig, ax = plt.subplots(figsize=(16.0, 4.8))
            sns.heatmap(
                heatmap, cmap="RdYlGn", center=0, vmin=-20, vmax=20,
                linewidths=0.4, linecolor="white", cbar_kws={"label": "Delta vs do nothing (pp)"},
                ax=ax,
            )
            ax.set_xlabel("Difficult chronic")
            ax.set_ylabel("")
            ax.set_title("Paired difficult-chronic deltas (colors clip at ±20 pp; rescues can be larger)")
            ax.tick_params(axis="x", labelrotation=70, labelsize=8)
            save_figure(fig, "best_systems_difficult_heatmap")
            plt.show()
            """
        ),
        markdown(
            """
            ### Per-chronic profile with simulator-greedy references

            This is the detailed paired view used in Section 8 of the original
            action-space notebook. Chronics are sorted by do-nothing survival,
            so the difficult cases appear first and the 26 do-nothing-perfect
            cases appear in the shaded region on the right.

            Every available one-step simulator-greedy result from
            `outputs/greedy_wcci_nomaint_50` is overlaid after a strict check of
            the environment and all 50 chronic fingerprints. The older
            `teacher_student_greedy_eval/test_fast_*` results are intentionally
            excluded because they use `bus36_wcci` and a different chronic set.

            When configurations have exactly the same mean survival, the smaller
            action cap is selected. The newly completed threshold curve is nearly
            flat; the subtitle reports the exact numerical winner on the current
            test set, while the sensitivity table above shows whether that tiny
            difference is practically meaningful.
            """
        ),
        code(
            """
            greedy_curve_rows = []
            greedy_inventory_rows = []
            rejected_greedy_results = []
            expected_reduction_prefix = "reduced_action_space_wcci_full2048a_90_v3_mk"

            if GREEDY_RESULT_ROOT.is_dir():
                for greedy_summary_path in sorted(
                    GREEDY_RESULT_ROOT.rglob("greedy_vs_do_nothing_summary.json")
                ):
                    greedy_payload = load_json(greedy_summary_path)
                    reduced_action_space = str(
                        greedy_payload.get("reduced_action_space", "")
                    )
                    cap_match = re.search(
                        r"reduced_action_space_wcci_full2048a_90_v3_mk(\\d+)\\.json$",
                        reduced_action_space,
                    )
                    if greedy_payload.get("env_id") != "bus36_wcci_nomaint":
                        rejected_greedy_results.append({
                            "path": str(greedy_summary_path),
                            "reason": f"environment={greedy_payload.get('env_id')!r}",
                        })
                        continue
                    if cap_match is None or expected_reduction_prefix not in reduced_action_space:
                        rejected_greedy_results.append({
                            "path": str(greedy_summary_path),
                            "reason": "not a wcci_full2048a_90_v3 mk reduction",
                        })
                        continue

                    action_size = int(cap_match.group(1))
                    greedy_episode_frame = pd.DataFrame(
                        greedy_payload.get("episodes", [])
                    )
                    required_greedy_columns = {
                        "greedy_chronic_fingerprint", "greedy_chronic_name",
                        "greedy_survival", "greedy_full_survival",
                        "do_nothing_survival", "do_nothing_full_survival",
                    }
                    missing_columns = required_greedy_columns - set(greedy_episode_frame)
                    if missing_columns:
                        raise ValueError(
                            f"Missing greedy columns in {greedy_summary_path}: "
                            f"{sorted(missing_columns)}"
                        )
                    if len(greedy_episode_frame) != 50:
                        raise ValueError(
                            f"Expected 50 greedy episodes in {greedy_summary_path}; "
                            f"got {len(greedy_episode_frame)}"
                        )
                    if greedy_episode_frame["greedy_chronic_fingerprint"].duplicated().any():
                        raise ValueError(
                            f"Duplicate greedy fingerprints in {greedy_summary_path}"
                        )
                    if "same_chronic_fingerprint" in greedy_episode_frame:
                        if not greedy_episode_frame["same_chronic_fingerprint"].fillna(False).all():
                            raise ValueError(
                                f"Greedy/do-nothing chronic mismatch in {greedy_summary_path}"
                            )

                    greedy_fingerprints = set(
                        greedy_episode_frame["greedy_chronic_fingerprint"].astype(str)
                    )
                    baseline_fingerprints = set(
                        baseline["chronic_fingerprint"].astype(str)
                    )
                    alignment_status = "50 exact fingerprints"
                    aligned_substitutions = 0
                    greedy_episode_frame["source_chronic_fingerprint"] = (
                        greedy_episode_frame["greedy_chronic_fingerprint"].astype(str)
                    )
                    greedy_episode_frame["source_chronic_name"] = (
                        greedy_episode_frame["greedy_chronic_name"].astype(str)
                    )
                    if greedy_fingerprints != baseline_fingerprints:
                        baseline_only = baseline_fingerprints - greedy_fingerprints
                        greedy_only = greedy_fingerprints - baseline_fingerprints
                        common_fingerprints = baseline_fingerprints & greedy_fingerprints
                        common_check = greedy_episode_frame.loc[
                            greedy_episode_frame["greedy_chronic_fingerprint"]
                            .astype(str).isin(common_fingerprints),
                            ["greedy_chronic_fingerprint", "do_nothing_survival"],
                        ].merge(
                            baseline[["chronic_fingerprint", "do_nothing_survival"]],
                            left_on="greedy_chronic_fingerprint",
                            right_on="chronic_fingerprint",
                            suffixes=("_greedy_run", "_baseline"),
                            validate="one_to_one",
                        )
                        common_do_nothing_matches = np.allclose(
                            common_check["do_nothing_survival_greedy_run"],
                            common_check["do_nothing_survival_baseline"],
                            rtol=0,
                            atol=1e-12,
                        )
                        benign_single_full_survival_substitution = (
                            len(baseline_only) == 1
                            and len(greedy_only) == 1
                            and len(common_fingerprints) == 49
                            and common_do_nothing_matches
                            and np.isclose(
                                greedy_episode_frame["do_nothing_survival"].mean(),
                                baseline["do_nothing_survival"].mean(),
                                rtol=0,
                                atol=1e-12,
                            )
                        )
                        if benign_single_full_survival_substitution:
                            baseline_only_fingerprint = next(iter(baseline_only))
                            greedy_only_fingerprint = next(iter(greedy_only))
                            baseline_only_row = baseline.loc[
                                baseline["chronic_fingerprint"].astype(str)
                                == baseline_only_fingerprint
                            ].iloc[0]
                            greedy_only_row = greedy_episode_frame.loc[
                                greedy_episode_frame["greedy_chronic_fingerprint"]
                                .astype(str) == greedy_only_fingerprint
                            ].iloc[0]
                            benign_single_full_survival_substitution = (
                                bool(baseline_only_row["do_nothing_full_survival"])
                                and bool(greedy_only_row["do_nothing_full_survival"])
                                and bool(greedy_only_row["greedy_full_survival"])
                                and np.isclose(
                                    greedy_only_row["do_nothing_survival"], 1.0
                                )
                                and np.isclose(greedy_only_row["greedy_survival"], 1.0)
                            )
                        if not benign_single_full_survival_substitution:
                            raise ValueError(
                                f"Greedy result does not match the common 50 chronics: "
                                f"{greedy_summary_path}; "
                                f"baseline_only={sorted(baseline_only)}, "
                                f"greedy_only={sorted(greedy_only)}"
                            )

                        substitute_mask = (
                            greedy_episode_frame["greedy_chronic_fingerprint"]
                            .astype(str) == greedy_only_fingerprint
                        )
                        greedy_episode_frame.loc[
                            substitute_mask, "greedy_chronic_fingerprint"
                        ] = baseline_only_fingerprint
                        greedy_episode_frame.loc[
                            substitute_mask, "greedy_chronic_name"
                        ] = baseline_only_row["chronic_name"]
                        alignment_status = (
                            "49 exact + 1 audited full-survival substitution: "
                            f"{greedy_only_row['greedy_chronic_name']} -> "
                            f"{baseline_only_row['chronic_name']}"
                        )
                        aligned_substitutions = 1

                    greedy_episode_frame = greedy_episode_frame.rename(columns={
                        "greedy_chronic_fingerprint": "chronic_fingerprint",
                        "greedy_chronic_name": "chronic_name",
                        "greedy_survival": "survival",
                        "greedy_full_survival": "full_survival",
                    })[[
                        "chronic_fingerprint", "chronic_name", "survival",
                        "full_survival", "source_chronic_fingerprint",
                        "source_chronic_name",
                    ]]
                    greedy_episode_frame["action_size"] = action_size
                    greedy_episode_frame["summary_path"] = str(greedy_summary_path)
                    greedy_episode_frame["alignment_status"] = alignment_status
                    greedy_curve_rows.append(greedy_episode_frame)
                    greedy_inventory_rows.append({
                        "action_size": action_size,
                        "episodes": len(greedy_episode_frame),
                        "mean_survival": greedy_episode_frame["survival"].mean(),
                        "full_survival_rate": greedy_episode_frame["full_survival"].mean(),
                        "episodes_worse_than_do_nothing": int(
                            greedy_payload.get("greedy_worse_episodes", -1)
                        ),
                        "aligned_substitutions": aligned_substitutions,
                        "alignment_status": alignment_status,
                        "summary_path": str(greedy_summary_path),
                    })

            greedy_curve_columns = [
                "chronic_fingerprint", "chronic_name", "survival",
                "full_survival", "source_chronic_fingerprint",
                "source_chronic_name", "action_size", "summary_path",
                "alignment_status",
            ]
            greedy_chronic_curves = (
                pd.concat(greedy_curve_rows, ignore_index=True)
                if greedy_curve_rows
                else pd.DataFrame(columns=greedy_curve_columns)
            )
            greedy_curve_inventory = pd.DataFrame(
                greedy_inventory_rows,
                columns=[
                    "action_size", "episodes", "mean_survival",
                    "full_survival_rate", "episodes_worse_than_do_nothing",
                    "aligned_substitutions", "alignment_status", "summary_path",
                ],
            ).sort_values("action_size").reset_index(drop=True)

            if greedy_curve_inventory["action_size"].duplicated().any():
                duplicates = greedy_curve_inventory.loc[
                    greedy_curve_inventory["action_size"].duplicated(False),
                    ["action_size", "summary_path"],
                ]
                raise ValueError(
                    "Multiple comparable greedy results found for the same cap:\\n"
                    + duplicates.to_string(index=False)
                )

            found_greedy_caps = set(
                greedy_curve_inventory["action_size"].astype(int)
            )
            missing_greedy_caps = [
                cap for cap in GREEDY_ACTION_ORDER if cap not in found_greedy_caps
            ]
            if len(greedy_curve_inventory):
                display(greedy_curve_inventory)
                display(Markdown(
                    f"Loaded greedy curves for **{sorted(found_greedy_caps)}**. "
                    f"Missing locally: **{missing_greedy_caps or 'none'}**."
                ))
                if greedy_curve_inventory["aligned_substitutions"].sum():
                    display(Markdown(
                        "**Alignment audit:** each downloaded greedy result has "
                        "49 exact fingerprints and one different do-nothing-perfect "
                        "chronic. The unmatched baseline and greedy episodes both "
                        "have 100% do-nothing and greedy survival, so that single "
                        "point is aligned into the easy region and recorded in the "
                        "inventory instead of being silently accepted."
                    ))
            else:
                display(Markdown(
                    "**No comparable `bus36_wcci_nomaint` greedy result folders "
                    "are downloaded locally yet.** The plotting cell is ready and "
                    "will add the curves automatically after downloading "
                    "`outputs/greedy_wcci_nomaint_50`."
                ))

            original_candidates = run_metrics.loc[
                run_metrics["campaign"].isin(["Original ungated", "Original rho-gated"])
            ].copy()
            best_original_score = original_candidates["overall_mean_survival_percent"].max()
            best_original_row = (
                original_candidates.loc[
                    np.isclose(
                        original_candidates["overall_mean_survival_percent"],
                        best_original_score,
                        rtol=0,
                        atol=1e-12,
                    )
                ]
                .sort_values(["action_size", "result_key"], na_position="last")
                .iloc[0]
            )
            best_original_key = best_original_row["result_key"]
            best_original_episodes = episodes.loc[
                episodes["result_key"] == best_original_key
            ].copy()
            best_original_chronic_comparison = best_original_episodes[[
                "episode", "chronic_name", "chronic_fingerprint", "cohort",
                "steps", "max_steps", "survival", "do_nothing_steps",
                "do_nothing_survival", "delta_vs_do_nothing_pp",
                "model_full_survival",
            ]].sort_values(
                ["do_nothing_survival", "chronic_name"]
            ).reset_index(drop=True)

            best_original_and_greedy_comparison = (
                best_original_chronic_comparison.copy()
            )
            for action_size in sorted(found_greedy_caps):
                greedy_column = f"greedy_k{action_size}_survival"
                greedy_values = greedy_chronic_curves.loc[
                    greedy_chronic_curves["action_size"] == action_size,
                    ["chronic_fingerprint", "survival"],
                ].rename(columns={"survival": greedy_column})
                best_original_and_greedy_comparison = (
                    best_original_and_greedy_comparison.merge(
                        greedy_values,
                        on="chronic_fingerprint",
                        how="left",
                        validate="one_to_one",
                    )
                )

            fig, axis = plt.subplots(figsize=(16, 7.0))
            x = np.arange(len(best_original_chronic_comparison))
            greedy_colors = plt.cm.plasma(
                np.linspace(0.08, 0.82, max(len(found_greedy_caps), 1))
            )
            for color, action_size in zip(
                greedy_colors, sorted(found_greedy_caps)
            ):
                greedy_column = f"greedy_k{action_size}_survival"
                mean_survival = best_original_and_greedy_comparison[
                    greedy_column
                ].mean()
                axis.plot(
                    x,
                    best_original_and_greedy_comparison[greedy_column] * 100,
                    color=color,
                    linestyle="--",
                    linewidth=1.35,
                    alpha=0.82,
                    label=(
                        f"One-step greedy k={action_size} "
                        f"(mean {100 * mean_survival:.1f}%)"
                    ),
                    zorder=2,
                )
            axis.plot(
                x,
                best_original_chronic_comparison["do_nothing_survival"] * 100,
                color="#222222",
                marker="o",
                markersize=3,
                linewidth=1.9,
                label="Do nothing",
                zorder=4,
            )
            axis.plot(
                x,
                best_original_chronic_comparison["survival"] * 100,
                color=CAMPAIGN_COLORS[best_original_row["campaign"]],
                marker="o",
                markersize=3,
                linewidth=2.2,
                label=(
                    "Best original zero-shot system "
                    f"(mean {best_original_score:.1f}%)"
                ),
                zorder=5,
            )
            difficult_count = int(
                best_original_chronic_comparison["cohort"]
                .eq("Do-nothing < 100%")
                .sum()
            )
            axis.axvspan(
                -0.5, difficult_count - 0.5,
                color="#E45756", alpha=0.07,
            )
            axis.axvspan(
                difficult_count - 0.5, len(x) - 0.5,
                color="#54A24B", alpha=0.07,
            )
            axis.axvline(
                difficult_count - 0.5, color="#888888", linewidth=1
            )
            axis.text(
                (difficult_count - 1) / 2, 103,
                "do-nothing < 100%", ha="center", fontsize=10,
            )
            axis.text(
                (difficult_count + len(x) - 1) / 2, 103,
                "do-nothing = 100%", ha="center", fontsize=10,
            )
            axis.set_xlim(-0.5, len(x) - 0.5)
            axis.set_ylim(-2, 108)
            axis.set_xlabel("Test chronic (sorted by do-nothing survival)")
            axis.set_ylabel("Survival (%)")
            axis.set_title(
                "Transferred policy and one-step greedy references on every test chronic\\n"
                f"{best_original_row['variant']} · mk{int(best_original_row['action_size'])} · "
                f"local rho={best_original_row['rho_threshold']:.2f}"
            )
            axis.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.14),
                ncol=3,
                fontsize=9,
                frameon=False,
            )
            fig.tight_layout()
            save_figure(fig, "best_original_and_greedy_chronic_comparison")
            plt.show()

            display(best_original_row[[
                "campaign", "variant", "action_size", "rho_threshold",
                "overall_mean_survival_percent", "difficult_mean_survival_percent",
                "difficult_median_survival_percent", "difficult_rescues",
                "difficult_nonrescue_mean_delta_pp",
            ]].to_frame("value"))
            """
        ),
        code(
            """
            best_gated_key = selected_keys["Best fine-tuned gated"]
            best_gated_episodes = episodes.loc[episodes["result_key"] == best_gated_key]
            best_gated_hard = best_gated_episodes.loc[
                best_gated_episodes["cohort"] == "Do-nothing < 100%"
            ]
            rescue_mask = best_gated_hard["model_full_survival"]
            best_contributions = pd.DataFrame([
                {
                    "component": "Fully rescued difficult chronics",
                    "episodes": int(rescue_mask.sum()),
                    "contribution_to_50_episode_mean_pp": (
                        best_gated_hard.loc[rescue_mask, "delta_vs_do_nothing_pp"].sum() / 50
                    ),
                },
                {
                    "component": "Other difficult chronics",
                    "episodes": int((~rescue_mask).sum()),
                    "contribution_to_50_episode_mean_pp": (
                        best_gated_hard.loc[~rescue_mask, "delta_vs_do_nothing_pp"].sum() / 50
                    ),
                },
                {
                    "component": "Do-nothing-perfect chronics",
                    "episodes": 26,
                    "contribution_to_50_episode_mean_pp": (
                        best_gated_episodes.loc[
                            best_gated_episodes["cohort"] == "Do-nothing = 100%",
                            "delta_vs_do_nothing_pp",
                        ].sum() / 50
                    ),
                },
            ])
            display(best_contributions)
            display(best_gated_hard.loc[rescue_mask, [
                "chronic_name", "do_nothing_survival", "survival", "delta_vs_do_nothing_pp"
            ]].sort_values("delta_vs_do_nothing_pp", ascending=False))
            """
        ),
        markdown(
            """
            ## 10. Paired bootstrap comparisons

            Bootstrap intervals resample the 50 chronic pairs. They describe
            chronic-set uncertainty for these fixed checkpoints; they do not
            account for training-seed uncertainty or the optimistic bias from
            selecting methods on this same test set.
            """
        ),
        code(
            """
            RNG = np.random.default_rng(20260817)


            def paired_bootstrap(key_a, key_b=None, n_boot=50_000):
                a = episodes.loc[episodes["result_key"] == key_a, [
                    "chronic_fingerprint", "survival"
                ]].rename(columns={"survival": "a"})
                if key_b is None:
                    data = a.merge(
                        baseline[["chronic_fingerprint", "do_nothing_survival"]],
                        on="chronic_fingerprint", validate="one_to_one",
                    )
                    delta = 100 * (data["a"] - data["do_nothing_survival"]).to_numpy()
                else:
                    b = episodes.loc[episodes["result_key"] == key_b, [
                        "chronic_fingerprint", "survival"
                    ]].rename(columns={"survival": "b"})
                    data = a.merge(b, on="chronic_fingerprint", validate="one_to_one")
                    delta = 100 * (data["a"] - data["b"]).to_numpy()
                indices = RNG.integers(0, len(delta), size=(n_boot, len(delta)))
                bootstrap_means = delta[indices].mean(axis=1)
                return {
                    "mean_delta_pp": delta.mean(),
                    "median_delta_pp": np.median(delta),
                    "wins": int((delta > 1e-9).sum()),
                    "ties": int((np.abs(delta) <= 1e-9).sum()),
                    "losses": int((delta < -1e-9).sum()),
                    "ci95_low_pp": np.quantile(bootstrap_means, 0.025),
                    "ci95_high_pp": np.quantile(bootstrap_means, 0.975),
                }


            comparison_pairs = [
                ("Best original gated", None, "Do nothing"),
                ("Best original gated", "Best original ungated", "Best original ungated"),
                ("Best fine-tuned gated", None, "Do nothing"),
                ("Best fine-tuned gated", "Best fine-tuned ungated", "Best fine-tuned ungated"),
                ("Best fine-tuned gated", "Best original gated", "Best original gated"),
                ("Best fine-tuned gated", "Best original ungated", "Best original ungated"),
            ]
            bootstrap_rows = []
            for a_label, b_label, comparison_label in comparison_pairs:
                result = paired_bootstrap(
                    selected_keys[a_label], selected_keys[b_label] if b_label else None
                )
                result.update({"method": a_label, "comparison": comparison_label})
                bootstrap_rows.append(result)
            bootstrap_comparisons = pd.DataFrame(bootstrap_rows)[[
                "method", "comparison", "mean_delta_pp", "median_delta_pp",
                "wins", "ties", "losses", "ci95_low_pp", "ci95_high_pp",
            ]]
            display(bootstrap_comparisons)
            """
        ),
        markdown(
            """
            ## 11. Conclusions from the complete downloaded evidence

            1. **Candidate ranking and intervention timing are separate transfer
               problems.** Large candidate sets degrade ungated zero-shot models,
               while local rho gating can preserve useful actions in dangerous
               states and suppress nearly all safe-state requests.
            2. **Stage 2 confirms a robust leading original configuration.**
               `NL_tmean_f0_a0h1` at mk128 preserves all 26 do-nothing-perfect
               chronics and obtains four difficult rescues at every tested local
               rho from 0.85 to 1.00. Its overall mean spans only about 0.002 pp,
               so the thresholds are practically tied on this test set; selecting
               1.00 from the third decimal place would be over-interpretation.
            3. **The head choice matters more than the exact threshold for this
               model.** At mk128 and rho 0.90, `NL_tmean_f0_a0h1` is about 3.96 pp
               above its matched `a0h0` version. The newly screened typed-mean f1
               alternatives remain below the leading f0/h1 configuration.
            4. **Rho gating is not universally beneficial.** It strongly repairs
               some mistimed policies, but all tested stage-2 thresholds make
               `NLS_mean_f1_a0h0` worse than its matched ungated model and still
               sacrifice do-nothing-perfect chronics. Architecture and cap must
               therefore be screened before threshold tuning.
            5. **The learned actor still lacks reliable act/do-nothing
               calibration.** For the best hybrid, the policy requests actions
               frequently while the local heuristic executes only a tiny fraction.
               This motivates a learned two-stage intervention gate.
            6. **Fine-tuning is only a small numerical leader in the current
               download.** The best early fine-tuned gated result is about 0.23 pp
               above the best original gated result, on the same repeatedly
               inspected chronics. That margin is not evidence of superiority
               without seed replication and held-out evaluation.
            7. **Threshold selection must move off the test set.** Use validation
               chronics to select the cap and rho threshold, then evaluate once on
               fresh held-out chronics and repeat training over multiple seeds.
            """
        ),
        markdown(
            """
            ## 12. Export all tables and figures
            """
        ),
        code(
            """
            inventory.drop(columns=["action_artifacts"]).to_csv(
                EXPORT_DIR / "complete_result_inventory.csv", index=False
            )
            source_results.to_csv(EXPORT_DIR / "source_grid_results.csv", index=False)
            episodes.to_csv(EXPORT_DIR / "all_wcci_episode_metrics.csv", index=False)
            agent_behavior.to_csv(EXPORT_DIR / "all_wcci_agent_behavior.csv", index=False)
            run_metrics.to_csv(EXPORT_DIR / "all_wcci_run_metrics.csv", index=False)
            campaign_summary.to_csv(EXPORT_DIR / "campaign_summary.csv", index=False)
            zero_shot_cap.to_csv(EXPORT_DIR / "zero_shot_action_cap_summary.csv", index=False)
            matched_finetuning.to_csv(EXPORT_DIR / "matched_early_finetuning.csv", index=False)
            ft_gate.to_csv(EXPORT_DIR / "finetuned_rho_sweep.csv", index=False)
            original_gate.to_csv(EXPORT_DIR / "original_rho_cap_sweep.csv", index=False)
            stage2_matched.to_csv(EXPORT_DIR / "original_rho_stage2_matched.csv", index=False)
            threshold_sensitivity.to_csv(
                EXPORT_DIR / "original_rho_threshold_sensitivity.csv", index=False
            )
            selected_comparison.to_csv(EXPORT_DIR / "selected_best_systems.csv", index=False)
            best_contributions.to_csv(EXPORT_DIR / "best_system_mean_contributions.csv", index=False)
            heatmap_long.to_csv(EXPORT_DIR / "best_systems_difficult_chronic_deltas.csv", index=False)
            best_original_chronic_comparison.to_csv(
                EXPORT_DIR / "best_original_model_chronic_comparison.csv", index=False
            )
            best_original_and_greedy_comparison.to_csv(
                EXPORT_DIR / "best_original_and_greedy_chronic_comparison.csv",
                index=False,
            )
            greedy_curve_inventory.to_csv(
                EXPORT_DIR / "greedy_chronic_curve_inventory.csv", index=False
            )
            greedy_chronic_curves.to_csv(
                EXPORT_DIR / "all_greedy_chronic_curves.csv", index=False
            )
            bootstrap_comparisons.to_csv(EXPORT_DIR / "paired_bootstrap_comparisons.csv", index=False)

            best_row = run_metrics.iloc[0]
            leading_original_curve = threshold_sensitivity.loc[
                (threshold_sensitivity["variant"] == best_original_row["variant"])
                & (threshold_sensitivity["action_size"] == best_original_row["action_size"])
            ].iloc[0]
            summary_lines = [
                "# Complete shared WCCI analysis summary",
                "",
                f"- Shared result JSONs inventoried: {len(shared_inventory)}.",
                f"- Comparable shared WCCI evaluations: {int(shared_inventory['is_wcci_50'].sum())}.",
                f"- WCCI evaluations including the external comparator: {len(wcci_inventory)}.",
                f"- WCCI model episodes: {len(episodes)}.",
                f"- Best current result: {best_row['overall_mean_survival_percent']:.3f}%.",
                f"- Best campaign: {best_row['campaign']}.",
                f"- Best variant: {best_row['variant']}.",
                f"- Best action cap: mk{int(best_row['action_size'])}.",
                f"- Best heuristic: {best_row['heuristic']} at rho={best_row['rho_threshold']:.2f}.",
                f"- Easy preservation: {100 * best_row['easy_preservation_rate']:.1f}%.",
                f"- Difficult mean / median: {best_row['difficult_mean_survival_percent']:.3f}% / {best_row['difficult_median_survival_percent']:.3f}%.",
                f"- Difficult rescues: {int(best_row['difficult_rescues'])}.",
                f"- Difficult non-rescue delta: {best_row['difficult_nonrescue_mean_delta_pp']:.3f} pp.",
                "",
                f"- Original rho-gated evaluations: {len(original_gate)} (including {len(stage2_gate)} stage-2 runs).",
                f"- Best original system: {best_original_row['variant']}, mk{int(best_original_row['action_size'])}, local rho={best_original_row['rho_threshold']:.2f}.",
                f"- Best original mean survival: {best_original_row['overall_mean_survival_percent']:.3f}%.",
                f"- Leading original threshold range: {leading_original_curve['rho_min']:.2f}–{leading_original_curve['rho_max']:.2f}.",
                f"- Mean-survival spread across that range: {leading_original_curve['overall_range_pp']:.4f} pp (practical tie).",
                f"- Numerical fine-tuned lead over best original: {best_row['overall_mean_survival_percent'] - best_original_row['overall_mean_survival_percent']:.3f} pp.",
                "",
                "Caveat: caps and rho thresholds were inspected on these same 50 test chronics; use validation and a fresh held-out set for final claims.",
            ]
            (EXPORT_DIR / "complete_analysis_summary.md").write_text(
                "\\n".join(summary_lines) + "\\n", encoding="utf-8"
            )
            print("Exported complete analysis to:", EXPORT_DIR)
            """
        ),
    ]
    return nb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build_notebook(), output)
    print(output)


if __name__ == "__main__":
    main()
