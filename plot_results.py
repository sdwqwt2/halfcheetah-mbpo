"""Plot learning curves from runs/*/progress.csv.

    python plot_results.py --runs runs --out results

Produces
  learning_curves.png  eval return vs env steps (mean over seeds, min-max band)
  model_loss.png       dynamics-model holdout MSE vs env steps (MBPO runs)
  results_table.csv    final numbers per run (from summary.json)
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"mbpo": "#2a78d6", "sac": "#eb6834"}   # fixed colour per algorithm
LABELS = {"mbpo": "MBPO (ours)", "sac": "SAC (model-free baseline)"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load_runs(root):
    runs = []
    for csv_path in sorted(glob.glob(os.path.join(root, "*", "progress.csv"))):
        run_dir = os.path.dirname(csv_path)
        with open(os.path.join(run_dir, "config.json")) as f:
            cfg = json.load(f)
        df = pd.read_csv(csv_path)
        if len(df):
            runs.append({"dir": run_dir, "algo": cfg["algo"], "seed": cfg["seed"], "df": df})
    return runs


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=13, fontweight="bold")
    ax.set_xlabel("environment steps", color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(True, color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x, _: f"{x / 1000:.0f}k" if x else "0"))


def smooth(y, w):
    return pd.Series(y).rolling(w, min_periods=1).mean().to_numpy()


def curve(ax, runs, algo, col, w):
    group = [r for r in runs if r["algo"] == algo and col in r["df"]]
    if not group:
        return
    n = min(len(r["df"]) for r in group)  # align seeds on the shortest run
    x = group[0]["df"]["env_steps"].to_numpy()[:n]
    ys = np.stack([smooth(r["df"][col].to_numpy()[:n], w) for r in group])
    c = COLORS.get(algo, "#6250d6")
    label = f"{LABELS.get(algo, algo)} ({len(group)} seed{'s' if len(group) > 1 else ''})"
    if len(group) > 1:
        ax.fill_between(x, ys.min(0), ys.max(0), color=c, alpha=0.18, linewidth=0)
    ax.plot(x, ys.mean(0), color=c, linewidth=2, label=label)
    ax.annotate(f"{ys.mean(0)[-1]:.0f}", (x[-1], ys.mean(0)[-1]), xytext=(6, 0),
                textcoords="offset points", color=INK, fontsize=10, va="center")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="results")
    p.add_argument("--smooth", type=int, default=5, help="moving-average window (epochs)")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    runs = load_runs(args.runs)
    if not runs:
        print("no runs found")
        return
    algos = [a for a in ("mbpo", "sac") if any(r["algo"] == a for r in runs)]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    for algo in algos:
        curve(ax, runs, algo, "eval_return", args.smooth)
    style(ax, "HalfCheetah-v5: evaluation return", "return (deterministic policy)")
    ax.legend(frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "learning_curves.png"), dpi=150)
    plt.close(fig)

    if "mbpo" in algos:
        fig, ax = plt.subplots(figsize=(9, 4), facecolor=SURFACE)
        curve(ax, runs, "mbpo", "model_holdout_mse", 1)
        style(ax, "Dynamics model: holdout MSE (elites)", "MSE on held-out real data")
        ax.set_yscale("log")
        ax.legend(frameon=False, labelcolor=INK)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "model_loss.png"), dpi=150)
        plt.close(fig)

    rows = []
    for r in runs:
        s_path = os.path.join(r["dir"], "summary.json")
        row = {"run": os.path.basename(r["dir"]), "algo": r["algo"], "seed": r["seed"],
               "env_steps": int(r["df"]["env_steps"].iloc[-1]),
               "last_eval_return": float(r["df"]["eval_return"].iloc[-1]),
               "max_eval_return": float(r["df"]["eval_return"].max())}
        if os.path.exists(s_path):
            with open(s_path) as f:
                s = json.load(f)
            row.update({k: s[k] for k in ("final_policy_return", "best_policy_return", "hours") if k in s})
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.out, "results_table.csv"), index=False)
    print(table.to_string(index=False))

    # sample efficiency: env steps needed to first reach a return threshold
    for thr in (3000, 5000, 8000):
        hits = {}
        for r in runs:
            ok = r["df"][r["df"]["eval_return"] >= thr]
            hits.setdefault(r["algo"], []).append(int(ok["env_steps"].iloc[0]) if len(ok) else None)
        print(f"steps to reach return {thr}: {hits}")


if __name__ == "__main__":
    main()
