"""Live browser view of LunarLander (no window system needed).

Runs the simulator in a background thread, encodes each frame as a JPEG,
and streams it to the browser as MJPEG. Open http://localhost:8000

Usage:
    python live_view.py                       # random agent
    python live_view.py --model runs/x/model  # trained SB3 PPO agent (later stages)
"""
import argparse
import io
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # render off-screen

import gymnasium as gym
from PIL import Image

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>LunarLander live</title>
<style>
 body{background:#111;color:#eee;font-family:system-ui,sans-serif;margin:0;padding:16px;
      display:flex;gap:24px;flex-wrap:wrap}
 img{width:600px;max-width:100%;border:2px solid #444;border-radius:6px}
 .box{min-width:260px} h1{font-size:20px;margin:0 0 12px}
 td{padding:3px 12px 3px 0} .n{font-variant-numeric:tabular-nums;font-weight:600}
 .good{color:#6f6}.bad{color:#f77}
</style></head><body>
<div><h1 id="t">LunarLander live</h1><img src="/stream"></div>
<div class="box"><table>
<tr><td>Agent</td><td class="n" id="agent"></td></tr>
<tr><td>Episode</td><td class="n" id="ep"></td></tr>
<tr><td>Step</td><td class="n" id="step"></td></tr>
<tr><td>Reward so far</td><td class="n" id="rew"></td></tr>
<tr><td>Last reward</td><td class="n" id="last"></td></tr>
<tr><td>Mean (all eps)</td><td class="n" id="mean"></td></tr>
</table><p style="color:#999;font-size:13px">Landing = 200+ points.<br>Crash = -100 penalty.</p></div>
<script>
async function tick(){
  try{const s=await (await fetch('/stats')).json();
    const f=x=>x===null?'-':x.toFixed(1);
    agent.textContent=s.agent;ep.textContent=s.episode;step.textContent=s.step;
    rew.textContent=f(s.reward);last.textContent=f(s.last);mean.textContent=f(s.mean);
    rew.className='n '+(s.reward>=0?'good':'bad');
  }catch(e){}
  setTimeout(tick,200);}
tick();
</script></body></html>"""

state = {"jpeg": None, "agent": "random", "episode": 0, "step": 0,
         "reward": 0.0, "last": None, "mean": None}
cond = threading.Condition()


def sim_loop(model_path, fps, seed):
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    model = None
    if model_path:
        from stable_baselines3 import PPO
        model = PPO.load(model_path, device="cpu")
        state["agent"] = f"PPO ({os.path.basename(model_path)})"
    rewards = []
    ep = 0
    while True:
        obs, _ = env.reset(seed=seed + ep)
        total, step, done = 0.0, 0, False
        state.update(episode=ep + 1, step=0, reward=0.0)
        while not done:
            t0 = time.time()
            if model is None:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(action)
            total += r
            step += 1
            done = term or trunc
            buf = io.BytesIO()
            Image.fromarray(env.render()).save(buf, format="JPEG", quality=80)
            with cond:
                state.update(jpeg=buf.getvalue(), step=step, reward=total)
                cond.notify_all()
            time.sleep(max(0.0, 1.0 / fps - (time.time() - t0)))
        rewards.append(total)
        state.update(last=total, mean=sum(rewards) / len(rewards))
        ep += 1
        time.sleep(1.0)  # brief pause so the final frame stays visible


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # keep the console quiet
        pass

    def do_GET(self):
        if self.path == "/":
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/stats":
            body = json.dumps({k: v for k, v in state.items() if k != "jpeg"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            last = None
            try:
                while True:
                    with cond:
                        cond.wait_for(lambda: state["jpeg"] is not None and state["jpeg"] is not last,
                                      timeout=5)
                        frame = state["jpeg"]
                    if frame is None or frame is last:
                        continue
                    last = frame
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                     b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n")
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="path to a saved SB3 PPO model")
    ap.add_argument("--fps", type=float, default=30, help="playback speed (physics runs at 50)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    threading.Thread(target=sim_loop, args=(args.model, args.fps, args.seed), daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    server.daemon_threads = True
    print(f"Open http://localhost:{args.port}  (Ctrl+C to stop)", flush=True)
    server.serve_forever()
