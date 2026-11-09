"""Stage 1: random-agent baseline for LunarLander-v3.

Part A: trace one episode step by step (see where reward comes from).
Part B: run 100 episodes with evaluation seeds 1000-1099 and summarize.
"""
import numpy as np
import gymnasium as gym

ACTIONS = ["noop", "left", "main", "right"]
EVAL_SEEDS = range(1000, 1100)  # reserved for evaluation; training uses seeds 0-4


def run_episode(env, seed, trace_every=0):
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)  # makes the "random" actions reproducible too
    total, steps = 0.0, 0
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        total += reward
        steps += 1
        if trace_every and (steps == 1 or steps % trace_every == 0 or terminated or truncated):
            print(f"step {steps:3d} {ACTIONS[action]:5s} r={reward:8.2f} total={total:8.1f} "
                  f"obs=[x={obs[0]:+.2f} y={obs[1]:+.2f} vx={obs[2]:+.2f} vy={obs[3]:+.2f} "
                  f"ang={obs[4]:+.2f} angv={obs[5]:+.2f} legs={int(obs[6])}{int(obs[7])}]")
        if terminated or truncated:
            # The env gives exactly -100 on a crash / leaving the screen and exactly +100 on coming to rest.
            if truncated:
                outcome = "timeout"
            elif reward == 100:
                outcome = "landed"
            else:
                outcome = "crash"
            return total, steps, outcome


if __name__ == "__main__":
    env = gym.make("LunarLander-v3")

    print("=== Part A: one traced episode (seed 0) ===")
    total, steps, outcome = run_episode(env, seed=0, trace_every=10)
    print(f"-> total reward {total:.1f}, {steps} steps, outcome: {outcome}\n")

    print("=== Part B: 100 random episodes (seeds 1000-1099) ===")
    results = [run_episode(env, seed=s) for s in EVAL_SEEDS]
    rewards = np.array([r[0] for r in results])
    lengths = np.array([r[1] for r in results])
    outcomes = [r[2] for r in results]
    print(f"mean reward : {rewards.mean():8.1f}  (std {rewards.std():.1f})")
    print(f"median      : {np.median(rewards):8.1f}")
    print(f"min / max   : {rewards.min():8.1f} / {rewards.max():.1f}")
    print(f"mean length : {lengths.mean():8.1f} steps  (min {lengths.min()}, max {lengths.max()})")
    print(f"episodes >= 200 : {(rewards >= 200).sum()} / 100")
    for o in ("crash", "landed", "timeout"):
        print(f"outcome {o:8s}: {outcomes.count(o)}")
    env.close()
