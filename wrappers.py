"""Stage 5: reward wrappers for LunarLander-v3.

ShapedRewardWrapper  - potential-based soft-landing shaping (policy-invariant, see below)
HoverBonusWrapper    - deliberately BAD per-step bonus (negative control: reward hacking)

Both leave observations, termination and truncation untouched and keep the original
environment reward in info["original_reward"].
"""
import gymnasium as gym


def landing_potential(obs, k=50.0, y0=0.5):
    """Phi(s) = -k * w(y) * (vx^2 + vy^2): speed is penalised more the closer we are to the ground.

    obs[1] = height above the pad, obs[2] = vx, obs[3] = vy (all in observation units).
    w(y) is 0 at or above height y0 and rises linearly to 1 at ground level.
    """
    y, vx, vy = float(obs[1]), float(obs[2]), float(obs[3])
    w = min(1.0, max(0.0, 1.0 - y / y0))
    return -k * w * (vx * vx + vy * vy)


class ShapedRewardWrapper(gym.Wrapper):
    """Adds F = gamma * Phi(s') - Phi(s) to the reward (Ng, Harada & Russell, 1999).

    Over an episode the terms telescope, so shaping changes WHEN feedback arrives but not which
    policy is optimal, provided that (a) gamma equals the learner's gamma and (b) Phi of a
    terminal state is 0. Truncation (the 1000-step limit) is not a real terminal state, so Phi is
    kept there (SB3 bootstraps the value at time limits).
    """

    def __init__(self, env, gamma=0.999, k=50.0, y0=0.5):
        super().__init__(env)
        self.gamma, self.k, self.y0 = gamma, k, y0
        self._phi = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._phi = landing_potential(obs, self.k, self.y0)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        phi_new = 0.0 if terminated else landing_potential(obs, self.k, self.y0)
        shaping = self.gamma * phi_new - self._phi
        self._phi = phi_new
        info = dict(info, original_reward=reward, shaping=shaping)
        return obs, reward + shaping, terminated, truncated, info


class HoverBonusWrapper(gym.Wrapper):
    """NEGATIVE CONTROL. Adds a flat +bonus every step. Not potential-based, so it changes the
    optimal policy: staying airborne for 1000 steps (+500) beats landing (~+260)."""

    def __init__(self, env, bonus=0.5):
        super().__init__(env)
        self.bonus = bonus

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info, original_reward=reward, shaping=self.bonus)
        return obs, reward + self.bonus, terminated, truncated, info
