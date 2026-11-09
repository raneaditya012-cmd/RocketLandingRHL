"""Stage 5 checks for wrappers.py (run: python check_wrappers.py). Exits non-zero on failure."""
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from wrappers import HoverBonusWrapper, ShapedRewardWrapper, landing_potential

GAMMA = 0.999
model = PPO.load("runs/baseline_seed0/model.zip", device="cpu")  # to get gentle, near-ground states


def policy_random(obs, env):
    return env.action_space.sample()


def policy_model(obs, env):
    return int(model.predict(obs, deterministic=True)[0])


def run(wrapped_kwargs, policy, seed):
    """Run one episode in lock-step: wrapped env and an identical plain env, same seed, same actions."""
    plain = gym.make("LunarLander-v3")
    wrapped = ShapedRewardWrapper(gym.make("LunarLander-v3"), **wrapped_kwargs)
    o1, _ = plain.reset(seed=seed)
    o2, _ = wrapped.reset(seed=seed)
    plain.action_space.seed(seed)
    assert np.array_equal(o1, o2), "reset observations differ"
    rec = dict(obs=[o2], orig=[], shaped=[], shaping=[], info_orig=[], flags=[])
    while True:
        a = policy(o1, plain)
        o1, r1, t1, tr1, _ = plain.step(a)
        o2, r2, t2, tr2, info = wrapped.step(a)
        assert np.array_equal(o1, o2), "observations differ"
        assert (t1, tr1) == (t2, tr2), "termination flags differ"
        rec["obs"].append(o2); rec["orig"].append(r1); rec["shaped"].append(r2)
        rec["shaping"].append(info["shaping"]); rec["info_orig"].append(info["original_reward"])
        rec["flags"].append((t2, tr2))
        if t2 or tr2:
            return rec


def check_wrapper(kwargs, label):
    n_ep, max_abs_F, max_tele_err, n_term = 0, 0.0, 0.0, 0
    for i, (pol, seed) in enumerate([(policy_random, s) for s in range(2000, 2030)] +
                                    [(policy_model, s) for s in range(2100, 2110)]):
        rec = run(kwargs, pol, seed)
        k, y0 = kwargs.get("k", 50.0), kwargs.get("y0", 0.5)
        # 1. original reward preserved exactly
        assert np.allclose(rec["info_orig"], rec["orig"], atol=0, rtol=0), "original reward not preserved"
        # 2. shaped - original == gamma*Phi(s') - Phi(s), recomputed independently from observations
        T = len(rec["orig"])
        terminated = rec["flags"][-1][0]
        for t in range(T):
            phi_s = landing_potential(rec["obs"][t], k, y0)
            phi_n = 0.0 if (t == T - 1 and terminated) else landing_potential(rec["obs"][t + 1], k, y0)
            expected = GAMMA * phi_n - phi_s
            assert abs((rec["shaped"][t] - rec["orig"][t]) - expected) < 1e-9, f"shaping mismatch at step {t}"
        # 3. telescoping: sum gamma^t F_t = gamma^T Phi(s_T) - Phi(s_0)   (Phi(s_T)=0 if terminated)
        F = np.array(rec["shaping"])
        lhs = float(np.sum(GAMMA ** np.arange(T) * F))
        phi_T = 0.0 if terminated else landing_potential(rec["obs"][-1], k, y0)
        rhs = GAMMA ** T * phi_T - landing_potential(rec["obs"][0], k, y0)
        max_tele_err = max(max_tele_err, abs(lhs - rhs))
        max_abs_F = max(max_abs_F, float(np.abs(F).max()))
        n_ep += 1; n_term += terminated
    print(f"[{label}] {n_ep} episodes ({n_term} terminated): checks 1-4 passed | "
          f"max |F| = {max_abs_F:.2f} | max telescoping error = {max_tele_err:.2e}")
    assert max_tele_err < 1e-9
    return max_abs_F


# Potential is exactly 0 high above the ground
assert landing_potential(np.array([0, 1.0, 3, 3, 0, 0, 0, 0])) == 0.0
print("[potential] Phi = 0 above y0: OK")

f_default = check_wrapper({"gamma": GAMMA}, "default k=50, y0=0.5")
f_strong = check_wrapper({"gamma": GAMMA, "y0": 2.0}, "y0=2.0 (Phi(s0) != 0, stronger telescoping test)")
assert f_default > 1.0, "shaping term never exceeded 1: the test would be vacuous"

# 5. SB3's Monitor (source of ep_rew_mean) must see the ORIGINAL reward, not the shaped one.
#    Use the hover control: its bonus makes shaped and original totals very different.
venv = make_vec_env("LunarLander-v3", n_envs=2, seed=0, vec_env_cls=DummyVecEnv,
                    wrapper_class=HoverBonusWrapper, wrapper_kwargs={"bonus": 0.5})
obs = venv.reset()
orig_sum, shaped_sum, checked = np.zeros(2), np.zeros(2), 0
for _ in range(600):
    obs, rew, done, infos = venv.step(np.array([venv.action_space.sample() for _ in range(2)]))
    for i in range(2):
        orig_sum[i] += infos[i]["original_reward"]; shaped_sum[i] += rew[i]
        if done[i]:
            ep_r = infos[i]["episode"]["r"]
            assert abs(ep_r - orig_sum[i]) < 1e-2, "Monitor does not see the original reward"
            assert abs(ep_r - shaped_sum[i]) > 1.0, "test vacuous: shaped == original"
            print(f"[monitor] env {i}: Monitor episode r = {ep_r:.1f} | original sum = {orig_sum[i]:.1f} | "
                  f"shaped sum = {shaped_sum[i]:.1f}")
            orig_sum[i] = shaped_sum[i] = 0.0; checked += 1
assert checked >= 2, "no episodes finished"

# 6. hover wrapper adds exactly +bonus per step
env = HoverBonusWrapper(gym.make("LunarLander-v3"), bonus=0.5)
env.reset(seed=0)
_, r, _, _, info = env.step(0)
assert abs(r - info["original_reward"] - 0.5) < 1e-12
print("[hover] adds exactly +0.5 per step: OK")
print("\nALL CHECKS PASSED")
