"""Stage 7: analysis and plots for the baseline / shaped / hover-control experiment.

    python plots.py            # reads runs/*, writes results/*.png, summary_per_run.csv, summary_stats.json

Sources (nothing is typed in by hand):
  runs/<cond>_seed<k>/progress.csv  training log; ep_rew_mean is the ORIGINAL reward (Monitor sits inside the wrapper)
  runs/<cond>_seed<k>/eval.json     final 100-episode evaluation on the unwrapped env, seeds 1000-1099
"""
import argparse
import glob
import itertools
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

CONDS = [("baseline", "Baseline"), ("shaped", "Shaped"), ("hover", "Hover control")]
MARKERS = ["o", "s", "D"]            # secondary encoding: shape (CVD / low-contrast relief)
LINESTYLES = ["-", "-", "--"]
THRESHOLD = 200.0

THEMES = {  # validated categorical slots 1-3 (blue, orange, aqua) and chrome from the reference palette
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", series=["#2a78d6", "#eb6834", "#1baf7a"]),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835", series=["#3987e5", "#d95926", "#199e70"]),
}


# ----------------------------------------------------------------------------- data
def load_runs(runs_dir):
    runs = {}
    for cond, _ in CONDS:
        for p in sorted(glob.glob(os.path.join(runs_dir, f"{cond}_seed*"))):
            seed = int(p.rsplit("seed", 1)[1])
            if not os.path.exists(os.path.join(p, "eval.json")):
                print(f"[warn] skipping unfinished run {p}")
                continue
            d = pd.read_csv(os.path.join(p, "progress.csv"))[
                ["time/total_timesteps", "rollout/ep_rew_mean", "rollout/ep_len_mean"]].dropna()
            ev = json.load(open(os.path.join(p, "eval.json")))["summary"]
            runs[(cond, seed)] = dict(steps=d.iloc[:, 0].to_numpy(float), rew=d.iloc[:, 1].to_numpy(float),
                                      length=d.iloc[:, 2].to_numpy(float), eval=ev)
    return runs


def steps_to_threshold(steps, rew, thr=THRESHOLD):
    idx = np.flatnonzero(rew >= thr)
    return float(steps[idx[0]]) if len(idx) else float("nan")


def seeds_of(runs, cond):
    return sorted(s for (c, s) in runs if c == cond)


# ----------------------------------------------------------------------------- stats
def perm_test(a, b):
    """Exact two-sided permutation test on the difference of means (b - a)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled, n = np.concatenate([a, b]), len(a)
    obs = b.mean() - a.mean()
    count = total = 0
    for idx in itertools.combinations(range(len(pooled)), n):
        mask = np.zeros(len(pooled), bool)
        mask[list(idx)] = True
        d = pooled[~mask].mean() - pooled[mask].mean()
        total += 1
        count += abs(d) >= abs(obs) - 1e-12
    return count / total


def compare(a, b, rng):
    """Shaped (b) minus baseline (a): difference, exact permutation p, Welch t, bootstrap 95% CI."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = float(b.mean() - a.mean())
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    boot = [rng.choice(b, len(b)).mean() - rng.choice(a, len(a)).mean() for _ in range(10000)]
    return dict(baseline_mean=float(a.mean()), baseline_sd=float(a.std(ddof=1)),
                shaped_mean=float(b.mean()), shaped_sd=float(b.std(ddof=1)),
                diff_shaped_minus_baseline=diff, welch_t=float(diff / se),
                perm_p_two_sided=float(perm_test(a, b)),
                boot95_low=float(np.percentile(boot, 2.5)), boot95_high=float(np.percentile(boot, 97.5)),
                n_baseline=len(a), n_shaped=len(b))


# ----------------------------------------------------------------------------- plotting helpers
def style_axes(ax, th):
    ax.set_facecolor(th["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(th["axis"])
    ax.tick_params(colors=th["muted"], labelsize=9, length=3)
    ax.grid(axis="y", color=th["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def new_fig(th, **kw):
    fig = plt.figure(facecolor=th["surface"], **kw)
    return fig


def kfmt(x, _):
    return "0" if x == 0 else (f"{x / 1e6:g}M" if x >= 1e6 else f"{x / 1e3:g}k")


def title(fig, th, main, sub):
    fig.text(0.04, 0.975, main, color=th["ink"], fontsize=13, fontweight="bold", va="top")
    fig.text(0.04, 0.935, sub, color=th["ink2"], fontsize=9.5, va="top")


# ----------------------------------------------------------------------------- figures
def fig_curves(runs, th, path):
    fig = new_fig(th, figsize=(9, 7.4))
    gs = fig.add_gridspec(2, 1, left=0.09, right=0.80, top=0.86, bottom=0.08, hspace=0.18)
    axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
    last = min(r["steps"][-1] for r in runs.values())
    grid = next(iter(runs.values()))["steps"]
    grid = grid[grid <= last]
    for ax, key, ylab in ((axes[0], "rew", "Episode reward (rolling mean, original reward)"),
                          (axes[1], "length", "Episode length (steps)")):
        style_axes(ax, th)
        ends = []
        for i, (cond, label) in enumerate(CONDS):
            seeds = seeds_of(runs, cond)
            if not seeds:
                continue
            Y = np.vstack([np.interp(grid, runs[(cond, s)]["steps"], runs[(cond, s)][key]) for s in seeds])
            col = th["series"][i]
            ax.fill_between(grid, Y.min(0), Y.max(0), color=col, alpha=0.16, linewidth=0)
            ax.plot(grid, Y.mean(0), color=col, lw=2, ls=LINESTYLES[i], label=f"{label} (n={len(seeds)})")
            ends.append((Y.mean(0)[-1], label))
        # direct labels at the line ends, nudged apart when the values are close
        ends.sort()
        offs, prev = [], None
        for v, lab in ends:
            dy = 0
            if prev is not None and (v - prev) < (0.07 * (ax.get_ylim()[1] - ax.get_ylim()[0])):
                dy = 10
            offs.append((v, lab, dy))
            prev = v
        for v, lab, dy in offs:
            ax.annotate(lab, (grid[-1], v), xytext=(6, dy - 3), textcoords="offset points",
                        color=th["ink2"], fontsize=9, annotation_clip=False)
        ax.set_ylabel(ylab, color=th["ink2"], fontsize=9)
        ax.xaxis.set_major_formatter(FuncFormatter(kfmt))
        ax.set_xlim(0, grid[-1])
    axes[0].axhline(THRESHOLD, color=th["muted"], lw=1, ls=(0, (4, 3)))
    axes[0].text(grid[-1] * 0.01, THRESHOLD + 6, "solved = 200", color=th["muted"], fontsize=8.5)
    axes[1].axhline(1000, color=th["muted"], lw=1, ls=(0, (4, 3)))
    axes[1].text(grid[-1] * 0.01, 1000 - 55, "time limit = 1000", color=th["muted"], fontsize=8.5)
    axes[0].tick_params(labelbottom=False)
    axes[1].set_xlabel("Training steps", color=th["ink2"], fontsize=9)
    leg = axes[0].legend(loc="lower right", frameon=False, fontsize=9, labelcolor=th["ink2"])
    title(fig, th, "Learning curves", "Line = mean over seeds, band = min to max across seeds. "
                                      "Rewards are always the original (unshaped) reward.")
    fig.savefig(path, dpi=150, facecolor=th["surface"])
    plt.close(fig)


def strip(ax, runs, th, get, ylabel, fmt="{:.0f}"):
    style_axes(ax, th)
    for i, (cond, label) in enumerate(CONDS):
        seeds = seeds_of(runs, cond)
        vals = np.array([get(runs[(cond, s)]) for s in seeds], float)
        xs = i + (np.arange(len(seeds)) - (len(seeds) - 1) / 2) * 0.09
        ok = ~np.isnan(vals)
        ax.scatter(xs[ok], vals[ok], s=62, marker=MARKERS[i], color=th["series"][i],
                   edgecolor=th["surface"], linewidth=1.4, zorder=3)
        if ok.any():
            m = vals[ok].mean()
            ax.plot([i - 0.3, i + 0.3], [m, m], color=th["ink"], lw=2, zorder=2)
            ax.annotate(fmt(m) if callable(fmt) else fmt.format(m), (i + 0.3, m), xytext=(5, -3),
                        textcoords="offset points",
                        color=th["ink"], fontsize=9)
    ax.set_xticks(range(len(CONDS)))
    ax.set_xticklabels([f"{l}\n(n={len(seeds_of(runs, c))})" for c, l in CONDS], color=th["ink2"], fontsize=9)
    ax.set_xlim(-0.55, len(CONDS) - 0.2)
    ax.set_ylabel(ylabel, color=th["ink2"], fontsize=9)


def fig_final(runs, th, path):
    fig = new_fig(th, figsize=(9.5, 4.6))
    gs = fig.add_gridspec(1, 2, left=0.08, right=0.97, top=0.80, bottom=0.14, wspace=0.28)
    a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    strip(a1, runs, th, lambda r: r["eval"]["mean_reward"], "Mean reward, 100 eval episodes", "{:.1f}")
    a1.axhline(THRESHOLD, color=th["muted"], lw=1, ls=(0, (4, 3)))
    strip(a2, runs, th, lambda r: r["eval"]["mean_steps"], "Mean episode length (steps)", "{:.0f}")
    title(fig, th, "Final performance per seed", "One dot per training seed; bar = mean over seeds. "
                                                  "Evaluated on the original environment, seeds 1000-1099.")
    fig.savefig(path, dpi=150, facecolor=th["surface"])
    plt.close(fig)


def fig_efficiency(runs, th, path, s2t):
    fig = new_fig(th, figsize=(6.6, 4.6))
    gs = fig.add_gridspec(1, 1, left=0.13, right=0.96, top=0.80, bottom=0.17)
    ax = fig.add_subplot(gs[0])
    strip(ax, runs, th, lambda r: s2t[id(r)], "Steps until rolling mean first reaches 200",
          lambda m: f"{m / 1e3:.0f}k")
    ax.yaxis.set_major_formatter(FuncFormatter(kfmt))
    labels = []
    for c, l in CONDS:
        v = [s2t[id(runs[(c, s)])] for s in seeds_of(runs, c)]
        labels.append(f"{l}\n{int(np.sum(~np.isnan(v)))}/{len(v)} reached 200")
    ax.set_xticklabels(labels, color=th["ink2"], fontsize=9)
    title(fig, th, "Sample efficiency", "Lower is faster. Training-time rolling mean over the last 100 episodes.")
    fig.savefig(path, dpi=150, facecolor=th["surface"])
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--themes", nargs="+", default=["light", "dark"])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    runs = load_runs(args.runs_dir)
    s2t = {id(r): steps_to_threshold(r["steps"], r["rew"]) for r in runs.values()}

    rows = []
    for (cond, seed), r in sorted(runs.items()):
        e = r["eval"]
        rows.append(dict(condition=cond, seed=seed, eval_mean_reward=e["mean_reward"], eval_sem=e["sem_reward"],
                         success_rate=e["success_rate"], rest_rate=e["rest_rate"], crash_rate=e["crash_rate"],
                         timeout_rate=e["timeout_rate"], on_pad_rate=e["on_pad_rate"],
                         mean_steps=e["mean_steps"], main_steps=e["mean_main_steps"],
                         side_steps=e["mean_side_steps"], steps_to_200=s2t[id(r)],
                         final_train_ep_rew_mean=float(r["rew"][-1])))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "summary_per_run.csv"), index=False, float_format="%.4f")
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))

    # statistics: shaped vs baseline
    rng = np.random.default_rng(0)
    base, shp = df[df.condition == "baseline"], df[df.condition == "shaped"]
    stats = {}
    for name, col in (("eval_mean_reward", "eval_mean_reward"), ("mean_steps", "mean_steps"),
                      ("steps_to_200", "steps_to_200")):
        a, b = base[col].to_numpy(), shp[col].to_numpy()
        if np.isnan(a).any() or np.isnan(b).any():
            print(f"[warn] {name}: some runs never reached the threshold; excluded from the test")
            a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        stats[name] = compare(a, b, rng)
    stats["note"] = ("Exact two-sided permutation tests, 3 metrics, no multiple-comparison correction; "
                     "n=5 seeds per condition. Treat as exploratory.")
    json.dump(stats, open(os.path.join(args.out_dir, "summary_stats.json"), "w"), indent=1)
    print("\nShaped minus baseline:")
    for k, v in stats.items():
        if isinstance(v, dict):
            print(f"  {k:18s} base {v['baseline_mean']:9.1f} (sd {v['baseline_sd']:.1f}) | shaped {v['shaped_mean']:9.1f} "
                  f"(sd {v['shaped_sd']:.1f}) | diff {v['diff_shaped_minus_baseline']:+9.1f} "
                  f"[95% boot {v['boot95_low']:+.1f}, {v['boot95_high']:+.1f}] | perm p = {v['perm_p_two_sided']:.3f}")

    for t in args.themes:
        th, suf = THEMES[t], ("" if t == "light" else "_dark")
        fig_curves(runs, th, os.path.join(args.out_dir, f"learning_curves{suf}.png"))
        fig_final(runs, th, os.path.join(args.out_dir, f"final_performance{suf}.png"))
        fig_efficiency(runs, th, os.path.join(args.out_dir, f"sample_efficiency{suf}.png"), s2t)
    print("\nsaved figures + tables to", args.out_dir)


if __name__ == "__main__":
    main()
