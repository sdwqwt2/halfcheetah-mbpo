"""Plot learning curves from runs/*/progress.csv.

    python plot_results.py --runs runs --out results

Produces
  learning_curves.png  eval return vs env steps (mean over seeds, min-max band)
  wall_clock.png       eval return vs training hours (the compute cost of MBPO)
  model_loss.png       dynamics-model holdout MSE vs env steps (MBPO runs)
  model_uncertainty.png  epistemic (ensemble disagreement) and aleatoric std in rollouts
  results_table.csv    final numbers per run (from summary.json)

Runs are grouped by directory name without the "_seedN" suffix, so
runs/mbpo_seed0 + runs/mbpo_seed1 form the group "mbpo", and an ablation
such as runs/mbpo_k5_seed0 forms its own group "mbpo_k5".
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"mbpo": "#2a78d6", "sac": "#eb6834"}   # fixed colour for the two main groups
EXTRA_COLORS = ["#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
LABELS = {"mbpo": "MBPO (ours)", "sac": "SAC (model-free baseline)"}


def color(group, groups):
    if group in COLORS:
        return COLORS[group]
    others = [g for g in groups if g not in COLORS]
    return EXTRA_COLORS[others.index(group) % len(EXTRA_COLORS)]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load_runs(root):
    runs = []
    for csv_path in sorted(glob.glob(os.path.join(root, "*", "progress.csv"))):
        run_dir = os.path.dirname(csv_path)
        with open(os.path.join(run_dir, "config.json")) as f:
            cfg = json.load(f)
        df = pd.read_csv(csv_path)
        if len(df):
            group = re.sub(r"_seed\d+$", "", os.path.basename(run_dir))
            runs.append({"dir": run_dir, "algo": cfg["algo"], "group": group,
                         "seed": cfg["seed"], "df": df})
    return runs


def style(ax, title, ylabel, xlabel="environment steps"):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(True, color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)
    if xlabel == "environment steps":
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda x, _: f"{x / 1000:g}k" if x else "0"))


def smooth(y, w):
    return pd.Series(y).rolling(w, min_periods=1).mean().to_numpy()


def curve(ax, runs, name, col, w, groups, x_col="env_steps", x_scale=1.0, label_suffix=""):
    group = [r for r in runs if r["group"] == name and col in r["df"]]
    if not group:
        return
    n = min(len(r["df"]) for r in group)  # align seeds on the shortest run
    x = group[0]["df"][x_col].to_numpy()[:n] * x_scale
    ys = np.stack([smooth(r["df"][col].to_numpy()[:n], w) for r in group])
    c = color(name, groups)
    label = f"{LABELS.get(name, name)}{label_suffix} ({len(group)} seed{'s' if len(group) > 1 else ''})"
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
    groups = sorted({r["group"] for r in runs}, key=lambda g: (g not in COLORS, g))
    model_groups = [g for g in groups if any(r["group"] == g and r["algo"] == "mbpo" for r in runs)]

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, name), dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    for g in groups:
        curve(ax, runs, g, "eval_return", args.smooth, groups)
    style(ax, "HalfCheetah-v5: evaluation return", "return (deterministic policy)")
    ax.legend(frameon=False, labelcolor=INK)
    save(fig, "learning_curves.png")

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    for g in groups:
        curve(ax, runs, g, "eval_return", args.smooth, groups, x_col="time_min", x_scale=1 / 60)
    style(ax, "Evaluation return vs wall-clock time (Kaggle T4)", "return", xlabel="training hours")
    ax.legend(frameon=False, labelcolor=INK)
    save(fig, "wall_clock.png")

    if model_groups:
        fig, ax = plt.subplots(figsize=(9, 4), facecolor=SURFACE)
        for g in model_groups:
            curve(ax, runs, g, "model_holdout_mse", 1, groups)
        style(ax, "Dynamics model: holdout MSE (elites)", "MSE on held-out real data")
        ax.set_yscale("log")
        ax.legend(frameon=False, labelcolor=INK)
        save(fig, "model_loss.png")

        if any("rollout_epistemic_std" in r["df"] for r in runs):
            fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor=SURFACE)
            for g in model_groups:
                curve(axes[0], runs, g, "rollout_epistemic_std", 1, groups)
                curve(axes[1], runs, g, "rollout_aleatoric_std", 1, groups)
            style(axes[0], "Epistemic: ensemble disagreement", "std across elite means")
            style(axes[1], "Aleatoric: predicted noise", "mean predicted std")
            axes[0].legend(frameon=False, labelcolor=INK)
            save(fig, "model_uncertainty.png")

    rows = []
    for r in runs:
        s_path = os.path.join(r["dir"], "summary.json")
        row = {"run": os.path.basename(r["dir"]), "group": r["group"], "algo": r["algo"], "seed": r["seed"],
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
            hits.setdefault(r["group"], []).append(int(ok["env_steps"].iloc[0]) if len(ok) else None)
        print(f"steps to reach return {thr}: {hits}")


if __name__ == "__main__":
    main()
