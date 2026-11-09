# PLAN: PPO on LunarLander-v3

> Status: **DRAFT, awaiting approval.** No code written, nothing installed.

## 1. Goal and definition of "done"

Train a PPO agent that lands the lander in Gymnasium's `LunarLander-v3`, and compare a **baseline** (original reward) against a **shaped-reward** variant across multiple seeds. The result should be a portfolio-quality repo that I (the learner) understand end to end.

**Done means:**
- [ ] A trained baseline agent with mean reward **>= 200** over 100 evaluation episodes, on at least 4 of 5 seeds.
- [ ] A shaped-reward experiment, evaluated on the **original** reward, with a fair comparison across the same seeds and step budget.
- [ ] Reproducible: one command per script, fixed seeds, pinned `requirements.txt`.
- [ ] Learning curves with mean ± spread across seeds, plus a landing GIF.
- [ ] A README with the results (real numbers only), the lessons learned, and honest limitations.
- [ ] I can explain every file without looking at the code.

## 2. Assumptions (change any of these before approving)

| Item | Assumption |
|---|---|
| Environment | New **conda env, Python 3.11** (system Python 3.14 is too new for Box2D/PyTorch wheels) |
| Library | Stable-Baselines3 (PPO) + PyTorch (CPU build) + Gymnasium[box2d] |
| Extra algorithm | **None** (DQN is an optional stretch goal, out of scope for now) |
| Location | `C:\Users\adira\lunar-lander-ppo`, local git only |
| Demo | GIF in the README (adds `imageio`) |
| Time | About **10 hours total**: ~5-6 h hands-on in sessions of 1.5-2 h, plus ~3-4 h of unattended training |
| Seeds | **5 per condition** (0-4) |
| Hardware | Ryzen 5 5600H (6C/12T), 7.4 GB RAM, no GPU. Use ~8 parallel envs, not dozens |

## 3. File-by-file breakdown

| File | Purpose |
|---|---|
| `requirements.txt` | Pinned dependencies, so the project is reproducible |
| `wrappers.py` | `ShapedRewardWrapper`: a Gymnasium wrapper that adds a shaping term. It stores the **untouched original reward** in `info["original_reward"]` so evaluation stays fair |
| `train.py` | CLI: `--seed --timesteps --shaped --run-name`. Builds vectorized envs, trains PPO, logs metrics (TensorBoard + CSV), saves the model and checkpoints |
| `evaluate.py` | Loads a saved model, runs N episodes on the **unmodified** env, reports mean/std reward, success rate (>=200), landing rate, mean fuel use. Saves JSON |
| `demo.py` | Loads a model and renders episodes live (`human`) or records a GIF/MP4 for the README |
| `plots.py` | Reads logs and eval JSONs. Draws learning curves (mean ± std across seeds) and a final-performance comparison |
| `README.md` | Overview, results, how to run, what I learned, limitations |
| `PLAN.md` | This file |

Layout: `runs/<name>_seed<k>/` holds each run's model, logs, and eval JSON. `results/` holds plots and GIFs.

## 4. Build stages

Time estimates are rough guesses for a CPU laptop. **I'll replace them with measured numbers after Stage 1.** Every stage ends with a git commit and a pause for your questions.

### Stage 0: Environment setup (~20-40 min)
- **Build:** conda env (Python 3.11), install Gymnasium[box2d], SB3, PyTorch CPU, matplotlib, imageio, tensorboard. Init git, `.gitignore` (runs/, models, `__pycache__`), `requirements.txt`.
- **Why:** Box2D/swig install is the most likely thing to break. Fix it first.
- **Verify:** a 10-line script creates `LunarLander-v3`, prints the observation/action spaces, and runs 200 random steps. We also check `torch` imports and runs on CPU.

### Stage 1: Random-agent baseline + environment exploration (~20 min)
- **Build:** a small script (becomes the start of `evaluate.py`) that runs a random policy for 100 episodes.
- **Why:** gives the "floor" to compare against and lets us see the 8 observations and the reward in real numbers.
- **Verify:** random mean reward is strongly negative (expected roughly -150 to -200). The 8-value observation matches the table in our concept notes.

### Stage 2: Minimal PPO training, `train.py` (~30-45 min incl. a 1M-step run)
- **Build:** `train.py` with a vectorized env (8 envs), PPO with a sensible config, TensorBoard logging, a checkpoint callback, and an evaluation callback. Run one seed at 1M steps.
- **Why:** the core of the project. Also measures real wall-clock speed, which the later estimates depend on.
- **Verify:** `ep_rew_mean` climbs past 0, then toward 200. `ep_len_mean` drops from a hover-length plateau. Entropy loss moves from about -1.39 toward 0 gradually. `explained_variance` rises.
- **Time check:** I'll tell you the measured steps/sec and the predicted duration **before** launching anything over ~10 minutes.

### Stage 3: Evaluation, `evaluate.py` (~30 min)
- **Build:** the evaluation script described above, using fresh seeds distinct from training.
- **Why:** training reward is a noisy, biased signal. Evaluation on 100 unseen episodes is what we report.
- **Verify:** the output is reproducible for a fixed seed, and the numbers roughly agree with the training curve.

### Stage 4: Demo, `demo.py` (~20 min)
- **Build:** live render plus GIF export.
- **Why:** an intuition check (watch it fly) and README material.
- **Verify:** the GIF shows a landing, and I watch both a good episode and a failure.

### Stage 5: Reward-shaping wrapper, `wrappers.py` (~40 min)
- **Build:** `ShapedRewardWrapper` with a **potential-based** soft-landing term (Φ(s') − γΦ(s), which provably keeps the optimal policy). The potential would penalize speed and tilt more heavily near the ground. The original reward is preserved in `info`.
- **Why:** this is the experimental variable. Using the potential-based form avoids creating hacking incentives.
- **Verify:** unit-style checks: the wrapper's reward differs from the original by exactly the shaping term, `info["original_reward"]` equals the unwrapped reward on identical action sequences, and shaping sums over an episode behave as theory predicts.
- **Negative control** (included in the 10-hour budget): a deliberately bad, non-potential bonus (e.g. per-step "stay alive" reward) to **demonstrate** reward hacking (expected: hovering). It is educational, not part of the headline comparison.

### Stage 6: Multi-seed experiment (~3-4 hours compute, mostly unattended)
- **Build:** a runner that trains baseline and shaped for seeds 0-4 (10 runs), each for the same step budget. I'd run 2 at a time (6 physical cores), keeping `n_envs` modest.
- **Why:** a single RL run is anecdotal. RL results vary a lot between seeds.
- **Verify:** all runs finish with logs and evaluation JSONs. Same hyperparameters, same budgets, only the reward differs.
- **Time check:** estimated duration will be stated before launch.

### Stage 7: Analysis and plots, `plots.py` (~40 min)
- **Build:** learning curves (mean ± std across seeds, x = timesteps, y = **original** reward), final-eval bar/box plot, and a table with success rates.
- **Why:** turn raw logs into the evidence.
- **Verify:** every number in the plots traces back to a file in `runs/`.

### Stage 8: README and polish (~40 min)
- **Build:** README with the results table, curves, GIF, setup instructions, and a "what I learned / limitations" section. Cleanup and a final commit.
- **Verify:** a fresh clone plus the README steps reproduces a short run.

## 5. Experiment design

- **Conditions:** (A) baseline, original reward. (B) shaped reward (potential-based soft-landing term).
- **Seeds:** 5 per condition (0-4), so 10 main runs. The negative control gets 3 seeds.
- **Controls:** identical PPO hyperparameters, network, timesteps (target 1M), and number of envs. Only the reward wrapper differs.
- **Evaluation:** each final model is evaluated on **100 episodes with the original, unwrapped environment** using fixed evaluation seeds not seen in training.
- **Metrics:**
  - Mean ± std episode reward (primary)
  - Success rate: fraction of episodes with reward >= 200
  - Landing rate (episode ended at rest, not crash/timeout)
  - Timesteps to first reach a rolling mean of 200 (sample efficiency)
  - Mean episode length and fuel use (to detect hovering or reckless behavior)
- **Success criterion:** baseline reaches >= 200 mean eval reward. For the comparison, I'll report differences with their spread across seeds. With 5 seeds I'll report mean, std and every individual seed, and I'll be cautious about claiming significance. I'll say "suggestive" or "no clear difference", whatever the data shows.
- **Honest outcome policy:** if shaping doesn't help (or hurts), that goes in the README as a finding.

## 6. Risks and mitigations

| Risk | How we handle it |
|---|---|
| **Box2D/swig fails to install** (most likely on Windows) | Use conda + Python 3.11, where prebuilt binaries exist. Fallback: `conda install swig` first, or install the Visual C++ build tools. Verify in Stage 0 before anything else |
| **Python 3.14 incompatibility** | Avoid it: isolated 3.11 env, never touch the system Python |
| **Reward hacking** (hovering, leg-bouncing) | Use potential-based shaping only. Always evaluate on the original reward. Watch `ep_len_mean` (hover shows as ~1000) and watch the GIFs, not just the numbers |
| **Unstable / stalled training** | Known-good PPO hyperparameters as a starting point. Monitor `approx_kl` and `clip_fraction`. Lower learning rate or raise `ent_coef` if the entropy collapses early. Try 2M steps if 1M is short |
| **Seed variance** (one seed fails, another succeeds) | That is exactly why we run multiple seeds. Report all of them, never cherry-pick |
| **Long runs eat the laptop** | Limit parallel runs, checkpoint regularly so crashes don't lose everything, and announce the time before any run >10 min |
| **Only 7.4 GB RAM** | Keep `n_envs` around 8, no pixel observations, close heavy apps during runs |
| **Windows multiprocessing quirks** | Guard entry points with `if __name__ == "__main__":`. Fall back to `DummyVecEnv` if subprocess envs misbehave (the env is cheap) |
| **Rendering/GIF issues** | Render with `rgb_array` and write via `imageio`. Fall back to MP4 or screenshots |
| **Results don't hit 200** | Debug via the metrics, not by guessing. Worst case, report honestly with the diagnosis |

## 7. Approval

Seeds (5), time budget (~10 h) and the other assumptions in section 2 are now set. Reply **approved** (or request changes) to start Stage 0.

Already decided: **no extra algorithm comparison for now.**

## 8. Deviation log

- **Stage 0 (2026-11-09):** Windows Smart App Control blocked the unsigned Box2D binary, so the project runs in **WSL2 (Ubuntu-22.04, Python 3.10, plain virtualenv)** at `~/lunar-lander-ppo`, not conda/Python 3.11. Windows security settings were not changed. Installed: gymnasium 1.4.0, SB3 2.9.2, torch 2.14.1 (CPU). WSL has ~3.5 GB RAM, so keep `n_envs` around 8.
- **Live viewing (2026-11-09):** WSLg opened windows but never drew them (cause not found; a WSL restart was not tried). Real-time viewing uses `live_view.py`, a browser MJPEG stream at `http://localhost:8000`. It will also play trained models in Stage 4 (`--model`).
