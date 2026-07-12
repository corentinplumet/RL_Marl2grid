#!/usr/bin/env python3
"""Inspect transformer token schemas for the WCCI 36 environment."""

import argparse
import json
from argparse import Namespace

import numpy as np

from env.utils import MAEnvWrapper


def _shape_tree(obj):
    if isinstance(obj, dict):
        return {key: _shape_tree(value) for key, value in obj.items()}
    return list(np.asarray(obj).shape)


def _spec_summary(spec):
    return {
        "tokenizer_type": spec["tokenizer_type"],
        "n_tokens": int(spec["n_tokens"]),
        "token_feature_dim": int(spec["token_feature_dim"]),
        "n_sub": int(spec["n_sub"]),
        "n_busbar": int(spec["n_busbar"]),
        "agent_index": int(spec["agent_index"]),
        "n_sub_ids": int(len(spec["sub_ids"])),
        "n_line_ids": int(len(spec["line_ids"])),
        "n_gen_ids": int(len(spec["gen_ids"])),
        "n_load_ids": int(len(spec["load_ids"])),
        "group_token_names": list(spec["group_token_names"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="bus36_wcci")
    parser.add_argument("--tokenizer-type", default="group", choices=["group", "entity", "hybrid"])
    parser.add_argument("--tokenizer-max-feature-dim", type=int, default=128)
    parser.add_argument("--tokenizer-include-neighbors", action="store_true", default=True)
    parser.add_argument("--no-tokenizer-include-neighbors", action="store_false", dest="tokenizer_include_neighbors")
    parser.add_argument("--tokenizer-include-busbar-tokens", action="store_true")
    parser.add_argument("--reduced-action-space", default="")
    args = parser.parse_args()

    env_args = Namespace(
        env_id=args.env_id,
        n_envs=1,
        action_type="topology",
        reduced_action_space=args.reduced_action_space,
        difficulty=0,
        decentralized=True,
        n1_reward=False,
        env_config_path="scenario.json",
        norm_obs=False,
        actor_encoder="transformer",
        critic_encoder="transformer",
        tokenizer_type=args.tokenizer_type,
        tokenizer_include_neighbors=args.tokenizer_include_neighbors,
        tokenizer_include_maintenance=True,
        tokenizer_max_feature_dim=args.tokenizer_max_feature_dim,
        tokenizer_include_busbar_tokens=args.tokenizer_include_busbar_tokens,
        gnn_include_neighbors=False,
        seed=0,
        optimize_mem=True,
        split_chronics=False,
        test_chronics_pct=0.2,
        chronic_split_seed=0,
        chronic_shard_count=1,
        chronic_shard_index=0,
        use_heuristic=False,
        heuristic_type="idle",
        constraints_type=0,
        topology_reward_weight=0.0,
        line_margin_reward_weight=0.0,
    )

    env = MAEnvWrapper(env_args, idx=0)
    try:
        obs, _ = env.reset()
        payload = {
            "env_id": args.env_id,
            "tokenizer_type": args.tokenizer_type,
            "action_space": {
                agent: int(space.n) for agent, space in env.action_space.items()
            },
            "specs": {
                key: _spec_summary(value)
                for key, value in env.token_specs.items()
            },
            "observation_shapes": _shape_tree(obs),
        }
        print(json.dumps(payload, indent=2))
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

