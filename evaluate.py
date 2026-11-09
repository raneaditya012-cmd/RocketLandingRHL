"""Stage 3: evaluate a saved PPO model (or a random agent) on the ORIGINAL LunarLander-v3.

    python evaluate.py --model runs/baseline_seed0/model.zip
    python evaluate.py --random --out results/eval_random.json

Protocol: 100 episodes, seeds 1000-1099 (never used in training), deterministic actions,
unmodified environment (no reward wrapper), so scores are comparable across experiments.
"""
import argparse
import json
import os

import gymnasium as gym
import numpy as np

ENV_ID = "LunarLander-v3"
PAD_HALF_WIDTH = 0.2  # the pad spans x in [-0.2, 0.2] in observation units


def run_episode(env, policy, seed):
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)  # only matters for the random policy
    total, steps, main, side = 0.0, 0, 0, 0
    while True:
        action = policy(obs, env)
        obs, reward, terminated, truncated, _ = env.step(action)
        total += reward
        steps += 1
        main += action == 2
        side += action in (1, 3)
        if terminated or truncated:
            break
    if truncated:
        outcome = "timeout"
    elif reward == 100:       # env gives exactly +100 when the lander comes to rest
        outcome = "rest"
    else:                     # exactly -100: crash or flew off-screen
        outcome = "crash"
    return dict(seed=seed, reward=total, steps=steps, outcome=outcome,
                final_x=float(obs[0]), main_steps=int(main), side_steps=int(side))


def summarize(eps):
    r = np.array([e["reward"] for e in eps])
    n = len(eps)
    out = lambda k: sum(e["outcome"] == k for e in eps)
    return dict(
        n_episodes=n,
        mean_reward=float(r.mean()), std_reward=float(r.std()),
        sem_reward=float(r.std(ddof=1) / np.sqrt(n)),  # standard error of the mean
        median_reward=float(np.median(r)), min_reward=float(r.min()), max_reward=float(r.max()),
        success_rate=float((r >= 200).mean()),          # reward >= 200 per episode
        rest_rate=out("rest") / n, crash_rate=out("crash") / n, timeout_rate=out("timeout") / n,
        on_pad_rate=float(np.mean([e["outcome"] == "rest" and abs(e["final_x"]) < PAD_HALF_WIDTH
                                   for e in eps])),
        mean_steps=float(np.mean([e["steps"] for e in eps])),
        mean_main_steps=float(np.mean([e["main_steps"] for e in eps])),
        mean_side_steps=float(np.mean([e["side_steps"] for e in eps])),
    )


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", help="path to a saved SB3 PPO model (.zip)")
    g.add_argument("--random", action="store_true", help="evaluate a uniformly random agent")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--first-seed", type=int, default=1000)
    ap.add_argument("--stochastic", action="store_true", help="sample actions instead of taking the most likely")
    ap.add_argument("--out", default=None, help="JSON output path (default: next to the model)")
    args = ap.parse_args()

    if args.random:
        policy = lambda obs, env: env.action_space.sample()
        label = "random"
    else:
        from stable_baselines3 import PPO
        model = PPO.load(args.model, device="cpu")
        det = not args.stochastic
        policy = lambda obs, env: int(model.predict(obs, deterministic=det)[0])
        label = args.model

    out_path = args.out or os.path.join(os.path.dirname(args.model) if args.model else "results",
                                        "eval.json")
    env = gym.make(ENV_ID)  # plain env: no wrappers, so the ORIGINAL reward is measured
    eps = [run_episode(env, policy, s) for s in range(args.first_seed, args.first_seed + args.episodes)]
    env.close()

    summ = summarize(eps)
    print(f"Evaluated: {label}  ({args.episodes} episodes, seeds {args.first_seed}-"
          f"{args.first_seed + args.episodes - 1}, {'random actions' if args.random else 'stochastic' if args.stochastic else 'deterministic'})")
    print(f"mean reward   : {summ['mean_reward']:.1f} +/- {summ['sem_reward']:.1f} (s.e.m.)   std {summ['std_reward']:.1f}")
    print(f"median / min / max : {summ['median_reward']:.1f} / {summ['min_reward']:.1f} / {summ['max_reward']:.1f}")
    print(f"success (>=200)    : {summ['success_rate']:.0%}")
    print(f"rest/crash/timeout : {summ['rest_rate']:.0%} / {summ['crash_rate']:.0%} / {summ['timeout_rate']:.0%}")
    print(f"landed on pad      : {summ['on_pad_rate']:.0%}")
    print(f"mean steps {summ['mean_steps']:.0f} | main-engine steps {summ['mean_main_steps']:.0f} "
          f"| side-engine steps {summ['mean_side_steps']:.0f}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(dict(label=label, deterministic=not args.stochastic, summary=summ, episodes=eps), f, indent=1)
    print("saved ->", out_path)


if __name__ == "__main__":
    main()
