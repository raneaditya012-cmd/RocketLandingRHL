import gymnasium as gym, torch, stable_baselines3 as sb3
env = gym.make("LunarLander-v3")
obs, _ = env.reset(seed=0)
print("obs space:", env.observation_space)
print("action space:", env.action_space)
print("first obs:", obs.round(3))
steps = 0
for _ in range(200):
    obs, r, term, trunc, _ = env.step(env.action_space.sample()); steps += 1
    if term or trunc: break
print("ran", steps, "random steps OK")
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available(), "| threads:", torch.get_num_threads())
print("sb3", sb3.__version__)
