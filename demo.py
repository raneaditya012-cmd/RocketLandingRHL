"""Stage 4: watch or record a trained agent.

    python demo.py --model runs/baseline_seed0/model.zip --gif results/landing.gif
    python demo.py --model runs/baseline_seed0/model.zip --live      # browser view at localhost:8000

--gif  records N episodes into a small GIF with a text overlay (episode, step, reward).
--live hands over to live_view.py (the Linux window system here does not render pygame windows).
"""
import argparse
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # render off-screen

import gymnasium as gym
from PIL import Image, ImageDraw
from stable_baselines3 import PPO


def record(model_path, gif_path, episodes, first_seed, stride, width, fps):
    model = PPO.load(model_path, device="cpu")
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    frames, results = [], []
    for ep in range(episodes):
        seed = first_seed + ep
        obs, _ = env.reset(seed=seed)
        total, step = 0.0, 0
        while True:
            action = int(model.predict(obs, deterministic=True)[0])
            obs, r, term, trunc, _ = env.step(action)
            total += r
            step += 1
            done = term or trunc
            if step % stride == 0 or done:   # keep every `stride`-th frame (and always the last)
                img = Image.fromarray(env.render())
                h = round(img.height * width / img.width)
                img = img.resize((width, h), Image.LANCZOS)
                ImageDraw.Draw(img).text((8, 6), f"seed {seed}  step {step}  reward {total:.0f}",
                                         fill=(255, 255, 255))
                frames.append(img)
            if done:
                break
        # hold the last frame ~1 s so the viewer can see the final state
        frames.extend([frames[-1]] * fps)
        outcome = "timeout" if trunc else ("landed" if r == 100 else "crashed")
        results.append((seed, step, total, outcome))
        print(f"seed {seed}: reward {total:.1f}, {step} steps, {outcome}")
    env.close()
    os.makedirs(os.path.dirname(gif_path) or ".", exist_ok=True)
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=64) for f in frames]
    pal[0].save(gif_path, save_all=True, append_images=pal[1:], duration=round(1000 / fps),
                loop=0, optimize=True)
    print(f"saved {gif_path}  ({len(frames)} frames, {os.path.getsize(gif_path) / 1e6:.2f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gif", default=None, help="output GIF path")
    ap.add_argument("--live", action="store_true", help="stream to the browser instead")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--first-seed", type=int, default=1000)
    ap.add_argument("--stride", type=int, default=2, help="keep every Nth frame (2 = 25 fps look)")
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--fps", type=int, default=25)
    args = ap.parse_args()

    if args.live:
        os.execv(sys.executable, [sys.executable, "live_view.py", "--model", args.model])
    elif args.gif:
        record(args.model, args.gif, args.episodes, args.first_seed, args.stride, args.width, args.fps)
    else:
        ap.error("choose --gif PATH or --live")
