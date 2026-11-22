"""Generate notebooks/MBPO_HalfCheetah_Kaggle.ipynb from the source files.

The notebook is fully self-contained: every module of the `mbpo` package is
written to disk by a readable `%%writefile` cell, so on Kaggle you only upload
the .ipynb and press "Run All". Re-run this script after editing any .py file:

    python tools/build_kaggle_notebook.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "notebooks", "MBPO_HalfCheetah_Kaggle.ipynb")

SOURCES = [
    ("mbpo/__init__.py", "Package entry point."),
    ("mbpo/config.py", "All hyper-parameters (paper defaults for HalfCheetah)."),
    ("mbpo/utils.py", "Seeding, device selection, CSV + TensorBoard logging."),
    ("mbpo/networks.py", "Batched ensemble layers, twin Q-network, squashed-Gaussian actor."),
    ("mbpo/replay_buffer.py", "GPU-resident replay buffer (used for both D_env and D_model)."),
    ("mbpo/sac.py", "Soft Actor-Critic: the policy optimiser inside MBPO."),
    ("mbpo/dynamics.py", "Probabilistic ensemble dynamics model p(s', r | s, a)."),
    ("mbpo/trainer.py", "The MBPO loop: fit model -> act -> branched rollouts -> SAC updates."),
    ("train.py", "Command-line entry point used to launch each training run."),
    ("record_video.py", "Renders a policy to mp4."),
    ("plot_results.py", "Learning curves and the results table."),
]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


INTRO = r"""
# MBPO on HalfCheetah-v5 — training notebook (Group 18)

**Model-Based Policy Optimization** (Janner et al., 2019) on Gymnasium's MuJoCo `HalfCheetah-v5`.

Team: Azamat Kharisov, Alexander Kurilenko, Egor Novokreshchenov, Ivan Prikhodko, Maxim Izyumov.

### How to run on Kaggle
1. *Settings → Accelerator →* **GPU T4 x2**, and *Settings → Internet →* **On** (needed for `pip install`).
2. Click **Save Version → Save & Run All (Commit)**. The commit runs in the background for up to 12 h,
   even with the browser closed, and keeps everything written to `/kaggle/working`.
3. When it finishes, open the version's **Output** tab and download `mbpo_results.zip`
   (learning curves, logs, TensorBoard files, checkpoints, videos).

### What this notebook does
| step | what | time |
|---|---|---|
| 1 | install `gymnasium[mujoco]`, check GPUs | ~2 min |
| 2 | write the `mbpo` package to disk (`%%writefile` cells below) | instant |
| 3 | check MuJoCo off-screen rendering, record a *random-policy* video | ~1 min |
| 4 | 1-minute smoke test of the full pipeline (fails fast if anything is broken) | ~2 min |
| 5 | **train**: MBPO seed 0 on GPU 0, MBPO seed 1 on GPU 1, SAC baseline on GPU 0 | `TRAIN_HOURS` |
| 6 | plots, final evaluation table, video of the trained cheetah, zip of all results | ~5 min |

The two GPUs are used to run **two MBPO seeds in parallel** (MBPO is a sequential algorithm, so one run
cannot use two GPUs efficiently); the cheap model-free SAC baseline shares GPU 0 and shows the
sample-efficiency gain.
"""

SETTINGS = r"""
# ================= settings =================
import os
TRAIN_HOURS = 11.0          # wall-clock budget per run; Kaggle kills a session at 12 h
TOTAL_ENV_STEPS = 400_000   # paper budget for HalfCheetah; the time budget usually stops us first
UPDATES_PER_STEP = 20       # G: SAC gradient steps per env step (paper: 40; 20 to fit in 12 h)

# The run plan. Each entry is one training process. Optional keys:
#   steps=...  env-step budget for this run (default TOTAL_ENV_STEPS)
#   hours=...  time budget for this run (default TRAIN_HOURS)
#   set={...}  overrides of any field in mbpo/config.py  <- tuning / ablations go here
# Name runs "<group>_seed<N>": plots average all seeds of a group.
RUNS = [
    dict(name="mbpo_seed0", algo="mbpo", seed=0, gpu=0),
    dict(name="mbpo_seed1", algo="mbpo", seed=1, gpu=1),
    dict(name="sac_seed0",  algo="sac",  seed=0, gpu=0),   # cheap model-free baseline
]
# Ablation examples (for a separate session, see docs/EXPERIMENTS.md):
#   dict(name="mbpo_k5_seed0", algo="mbpo", seed=0, gpu=0, steps=60_000, set={"rollout_schedule": [0, 1, 5, 5]})
#   dict(name="mbpo_ens1_seed0", algo="mbpo", seed=0, gpu=1, steps=60_000, set={"ensemble_size": 1, "num_elites": 1})

SMOKE_TEST = os.environ.get("MBPO_SMOKE_TEST") == "1"  # set True for a ~3-minute dry run only
WORK_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
os.chdir(WORK_DIR)
print("working directory:", WORK_DIR, "| smoke test only:", SMOKE_TEST)
"""

INSTALL = r"""
%pip install -q "gymnasium[mujoco]>=1.0" imageio imageio-ffmpeg
"""

GPU_CHECK = r"""
import subprocess, sys, torch, gymnasium, mujoco
print("python", sys.version.split()[0], "| torch", torch.__version__,
      "| gymnasium", gymnasium.__version__, "| mujoco", mujoco.__version__)
N_GPU = torch.cuda.device_count()
for i in range(N_GPU):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
if N_GPU == 0:
    print("WARNING: no GPU found - enable 'GPU T4 x2' in the notebook settings (training on CPU is far too slow).")
"""

PKG_MD = r"""
## The implementation
Each cell below writes one source file. The same files live in our repository; the notebook just
bundles them so that Kaggle needs nothing but this `.ipynb`.
"""

RENDER = r"""
# MuJoCo needs an off-screen OpenGL backend on a headless server: try EGL (GPU), then OSMesa (CPU).
import subprocess, sys, os

def try_backend(backend):
    test = ("import gymnasium as gym; e = gym.make('HalfCheetah-v5', render_mode='rgb_array'); "
            "e.reset(seed=0); print(e.render().shape)")
    r = subprocess.run([sys.executable, "-c", test], capture_output=True, text=True,
                       env={**os.environ, "MUJOCO_GL": backend}, timeout=300)
    return r.returncode == 0, (r.stdout + r.stderr).strip().splitlines()[-1:]

RENDER_BACKEND = None
for backend in ["egl", "osmesa", "glfw"]:  # glfw only works with a display (local runs)
    ok, msg = try_backend(backend)
    print(f"MUJOCO_GL={backend}: {'OK' if ok else 'failed'} {msg}")
    if ok:
        RENDER_BACKEND = backend
        break
if RENDER_BACKEND is None:  # install OSMesa and retry
    subprocess.run("apt-get -qq update && apt-get -qq install -y libosmesa6 libosmesa6-dev > /dev/null",
                   shell=True)
    ok, msg = try_backend("osmesa")
    print("after installing libosmesa6:", "OK" if ok else "failed", msg)
    RENDER_BACKEND = "osmesa" if ok else None
print("render backend:", RENDER_BACKEND, "(training does not need rendering; only the videos do)")

def record(args):
    if RENDER_BACKEND is None:
        print("skipping video: no rendering backend")
        return
    r = subprocess.run([sys.executable, "record_video.py", *args], capture_output=True, text=True,
                       env={**os.environ, "MUJOCO_GL": RENDER_BACKEND})
    if r.returncode == 0:
        print(r.stdout.strip().splitlines()[-1])
    else:
        print(f"video failed (exit {r.returncode}):", r.stderr[-1500:])

record(["--random", "--out", "videos/random_policy.mp4"])   # "before training" clip
"""

SMOKE = r"""
# Fast end-to-end check of the whole pipeline before spending hours on the real run.
import subprocess, sys
r = subprocess.run([sys.executable, "train.py", "--algo", "mbpo", "--smoke", "--out-dir", "smoke/mbpo"],
                   capture_output=True, text=True)
print("\n".join(l for l in r.stdout.splitlines() if l.startswith("[")))
if r.returncode != 0:
    print(r.stderr[-3000:])
    raise RuntimeError("smoke test failed - fix this before launching the long run")
print("smoke test passed")
"""

TRAIN_MD = r"""
## Training
Every run is a separate process (`train.py`), pinned to one GPU with `CUDA_VISIBLE_DEVICES`.
Progress is logged once per epoch (= 1000 env steps) to `runs/<run>/progress.csv`, TensorBoard
(`runs/<run>/tb`) and `runs/<run>/stdout.log`; this cell prints a status line every 15 minutes.
Each run stops by itself when `TRAIN_HOURS` is used up, saves `best.pt` / `final.pt`, and evaluates
both policies on 10 episodes (`summary.json`). For later analysis it also keeps
`snapshots/step_XXXXXX.pt` every 25k env steps (policy, critics and dynamics model) and all real
transitions in `env_buffer.npz`.
"""

TRAIN = r"""
import subprocess, sys, time, os, json

def launch(run):
    out = f"runs/{run['name']}"
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, "train.py", "--algo", run["algo"], "--seed", str(run["seed"]),
           "--out-dir", out, "--time-budget-hours", str(run.get("hours", TRAIN_HOURS))]
    if SMOKE_TEST:
        cmd += ["--smoke"]
    else:
        cmd += ["--total-env-steps", str(run.get("steps", TOTAL_ENV_STEPS))]
        if run["algo"] == "mbpo":
            cmd += ["--updates-per-step", str(UPDATES_PER_STEP)]
    for key, value in run.get("set", {}).items():
        cmd += ["--set", f"{key}={json.dumps(value)}"]
    gpu = run.get("gpu", 0) % max(N_GPU, 1)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu) if N_GPU else ""}
    log = open(f"{out}/stdout.log", "w")
    print(f"launch {out} on GPU {gpu if N_GPU else 'none (CPU)'}: {' '.join(cmd[1:])}")
    return out, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)

procs = [launch(run) for run in RUNS]

def last_epoch_line(out):
    try:
        lines = [l for l in open(f"{out}/stdout.log") if "epoch" in l]
        return lines[-1].strip() if lines else "(starting...)"
    except FileNotFoundError:
        return "(no log yet)"

t0, last_print = time.time(), -1e9
while any(p.poll() is None for _, p in procs):
    if time.time() - last_print > (60 if SMOKE_TEST else 900):
        last_print = time.time()
        print(f"--- {(time.time() - t0) / 3600:.2f} h elapsed ---")
        for out, p in procs:
            print(("running " if p.poll() is None else "finished") + " | " + last_epoch_line(out))
    time.sleep(10)

print(f"all runs finished after {(time.time() - t0) / 3600:.2f} h")
for out, p in procs:
    status = "OK" if p.returncode == 0 else f"FAILED (exit {p.returncode})"
    print(f"{out}: {status}")
    if p.returncode != 0:
        print("".join(open(f"{out}/stdout.log").readlines()[-30:]))
    elif os.path.exists(f"{out}/summary.json"):
        print("   ", json.load(open(f"{out}/summary.json")))
"""

RESULTS_MD = r"""
## Results
Learning curves (mean evaluation return of the deterministic policy over 5 episodes after every 1000 env
steps, smoothed over 5 epochs; shaded band = min-max over seeds), return vs wall-clock hours, the
dynamics-model error and uncertainty, and the final table.
"""

PLOTS = r"""
import subprocess, sys
from IPython.display import Image, display
print(subprocess.run([sys.executable, "plot_results.py", "--runs", "runs", "--out", "results"],
                     capture_output=True, text=True).stdout)
for name in ["learning_curves.png", "wall_clock.png", "model_loss.png", "model_uncertainty.png"]:
    if os.path.exists(f"results/{name}"):
        display(Image(f"results/{name}"))
"""

VIDEO = r"""
# Video of the best MBPO policy (chosen by its 10-episode evaluation in summary.json).
import glob, json, base64
from IPython.display import HTML, display

best = None
for s_path in glob.glob("runs/mbpo_seed*/summary.json"):  # main MBPO runs only
    s = json.load(open(s_path))
    for which in ("best", "final"):
        score = s[f"{which}_policy_return"]
        if best is None or score > best[0]:
            best = (score, os.path.join(os.path.dirname(s_path), f"{which}.pt"))
if best:
    print(f"recording {best[1]} (10-episode mean return {best[0]:.0f})")
    record(["--checkpoint", best[1], "--out", "videos/mbpo_trained.mp4"])
sac_ckpt = "runs/sac_seed0/best.pt"
if os.path.exists(sac_ckpt):
    record(["--checkpoint", sac_ckpt, "--out", "videos/sac_baseline.mp4"])
# "learning progress" clips from the snapshots of the best MBPO run (25k, 100k, ... env steps)
if best:
    snaps = sorted(glob.glob(os.path.join(os.path.dirname(best[1]), "snapshots", "step_*.pt")))
    for snap in [s for s in snaps if int(s[-9:-3]) in (25_000, 50_000, 100_000, 200_000, 300_000)][:5]:
        record(["--checkpoint", snap, "--out", f"videos/progress_{os.path.basename(snap)[:-3]}.mp4"])

for v in ["videos/random_policy.mp4", "videos/mbpo_trained.mp4"]:
    if os.path.exists(v):
        b64 = base64.b64encode(open(v, "rb").read()).decode()
        display(HTML(f"<p><b>{v}</b></p><video width=640 controls src='data:video/mp4;base64,{b64}'></video>"))
"""

ZIP = r"""
# Bundle everything for download (Output tab -> mbpo_results.zip).
import shutil, tempfile
staging = tempfile.mkdtemp()
for d in ["runs", "results", "videos"]:
    if os.path.isdir(d):
        shutil.copytree(d, os.path.join(staging, d))
shutil.make_archive("mbpo_results", "zip", staging)
print(f"mbpo_results.zip: {os.path.getsize('mbpo_results.zip') / 1e6:.1f} MB")
"""


def build():
    cells = [md(INTRO), code(SETTINGS), code(INSTALL), code(GPU_CHECK), md(PKG_MD),
             code("import os\nos.makedirs('mbpo', exist_ok=True)")]
    for path, desc in SOURCES:
        with open(os.path.join(ROOT, path)) as f:
            src = f.read()
        cells.append(md(f"**`{path}`** — {desc}"))
        cells.append(code(f"%%writefile {path}\n{src}"))
    cells += [md("## Rendering check and a random-policy video (the \"before\" clip)"), code(RENDER),
              md("## Smoke test"), code(SMOKE),
              md(TRAIN_MD), code(TRAIN),
              md(RESULTS_MD), code(PLOTS),
              md("## Videos"), code(VIDEO),
              md("## Download"), code(ZIP)]
    nb = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"cell-{i:02d}"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", OUT, f"({len(cells)} cells)")


if __name__ == "__main__":
    build()
