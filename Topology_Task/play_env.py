"""Play with the bus14 MAEnvWrapper to understand obs, actions, and rewards.

Run from the Topology_Task/ directory:
    conda run -n marl2grid_iclr python play_env.py
"""

from argparse import Namespace

import numpy as np

from env.config import get_env_args
from env.utils import MAEnvWrapper


def build_args() -> Namespace:
    args = get_env_args()
    args.env_id = "bus14"
    args.n_envs = 1
    args.action_type = "topology"
    args.difficulty = 0
    args.decentralized = True
    args.norm_obs = False
    args.use_heuristic = False
    args.heuristic_type = "idle"
    args.constraints_type = 0
    args.n1_reward = False
    args.optimize_mem = True
    args.env_config_path = "scenario.json"
    args.seed = 0
    return args


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    args = build_args()
    env = MAEnvWrapper(args, idx=0)
    agents = sorted(env.observation_space.keys())
    max_steps = env.g2op_env.chronics_handler.max_episode_duration()

    section(f"Env: {args.env_id}  |  max_episode_duration = {max_steps} steps")
    print(f"Agents ({len(agents)}): {agents}")
    for a in agents:
        obs_dim = env.observation_space[a].shape[0]
        n_acts = env.action_space[a].n
        print(f"  {a}: obs dim = {obs_dim:>4}   |   action space size = {n_acts:>4}")

    section("First observation after reset (raw, unnormalized)")
    obs, _ = env.reset(seed=args.seed)
    for a in agents:
        ob = obs[a]
        print(f"  {a}: shape={ob.shape}, first 8 values = {np.array2string(ob[:8], precision=3)}")

    section("Decoding selected action indices for each agent")
    for a in agents:
        space = env._conv_action_space[a]
        n = space.n
        idxs = sorted(set([0, 1, 2, n // 2, n - 1]))
        for idx in idxs:
            act = space.from_gym(idx)
            label = " (DO-NOTHING)" if idx == 0 else ""
            print(f"\n  {a} | action #{idx}/{n - 1}{label}:")
            for line in str(act).splitlines():
                print(f"    {line}")
        print()

    section("RANDOM POLICY: 3 episodes")
    rng = np.random.default_rng(args.seed)
    for ep in range(3):
        obs, _ = env.reset(seed=args.seed + ep)
        cum = {a: 0.0 for a in agents}
        step = 0
        done = False
        while not done:
            action = {a: int(rng.integers(env.action_space[a].n)) for a in agents}
            obs, reward, dones, _, info = env.step(action)
            for a in agents:
                cum[a] += reward[a]
            if step < 3:
                action_str = ", ".join(f"{a}={action[a]}" for a in agents)
                reward_str = ", ".join(f"{a}={reward[a]:+.3f}" for a in agents)
                print(f"  ep{ep} step{step}: action=({action_str})  reward=({reward_str})  done={dones['agent_0']}")
            step += 1
            done = dones["agent_0"]
        survival = step / max_steps
        cum_str = ", ".join(f"{a}={cum[a]:+.2f}" for a in agents)
        print(f"  -> episode {ep}: survived {step}/{max_steps} steps ({survival:.1%})  |  cum reward: {cum_str}\n")

    section("DO-NOTHING POLICY (action=0 for everyone): 1 episode")
    obs, _ = env.reset(seed=args.seed + 1000)
    cum = {a: 0.0 for a in agents}
    step = 0
    done = False
    while not done:
        action = {a: 0 for a in agents}
        obs, reward, dones, _, info = env.step(action)
        for a in agents:
            cum[a] += reward[a]
        if step < 3:
            reward_str = ", ".join(f"{a}={reward[a]:+.3f}" for a in agents)
            print(f"  step{step}: reward=({reward_str})  done={dones['agent_0']}  info keys: {list(info.keys())}")
        step += 1
        done = dones["agent_0"]
    survival = step / max_steps
    cum_str = ", ".join(f"{a}={cum[a]:+.2f}" for a in agents)
    print(f"  -> do-nothing: survived {step}/{max_steps} steps ({survival:.1%})  |  cum reward: {cum_str}")


if __name__ == "__main__":
    main()
