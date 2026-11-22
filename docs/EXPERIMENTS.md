# Experiments and ablations: guidelines

These are recommendations for the experiments track, not strict requirements. Adapt them to the GPU
time you actually have. The **main run** (MBPO ×2 seeds + SAC baseline, ~11.5 h on Kaggle) is done
separately and must not be repeated. Everything below either uses short runs or reuses its outputs.

**Deadline:** the zip goes to Moodle by **Wednesday 23:00**, so results are needed by about Wednesday noon
to make it into the slides.

## 1. What we want to show

The proposal promised to **tune the rollout horizon and the ensemble size**. The presentation also
needs evidence for its main claims:

| claim (slide) | evidence |
|---|---|
| short rollouts keep model error small (slide 6) | horizon ablation + model error vs. horizon |
| the ensemble captures uncertainty (slide 5) | ensemble-size ablation + epistemic std plot |
| MBPO is sample-efficient (slide 4, 7) | main run: MBPO vs SAC (already produced) |
| MBPO costs extra compute (slide 7) | main run: `wall_clock.png` (already produced) |

## 2. Recommended ablations (short runs)

Change **one factor at a time** relative to the main configuration (k = 1, 7 networks / 5 elites, G = 20).
Use the **first 50–60k steps of the main run** as the baseline, so there is no need to rerun it.
Differences in MBPO show up early. In the paper, HalfCheetah reaches ~5–6k return by 60k steps.

| priority | run name | change (`set=`) | why |
|---|---|---|---|
| must | `mbpo_k5_seed0` | `{"rollout_schedule": [0, 1, 5, 5]}` | longer horizon |
| must | `mbpo_k15_seed0` | `{"rollout_schedule": [0, 1, 15, 15]}` | long horizon: should degrade (compounding error) |
| must | `mbpo_ens1_seed0` | `{"ensemble_size": 1, "num_elites": 1}` | no ensemble = no epistemic uncertainty |
| nice | `mbpo_ens3_seed0` | `{"ensemble_size": 3, "num_elites": 2}` | smaller ensemble |
| nice | `mbpo_real50_seed0` | `{"real_ratio": 0.5}` | how much synthetic data matters |
| nice | `mbpo_g5_seed0` | run with `UPDATES_PER_STEP = 5` | the update-to-data ratio is MBPO's main lever |
| optional | second seed of the "must" runs | same, `seed=1` | error bars |

Suggested budget is **60k env steps per run** (`steps=60_000`). Run **2 per GPU**, so 4 at once on
T4 x2. That gives roughly 3–4 h for the four "must"/"nice" runs, but we haven't measured T4 speed, so
check `epoch_time_s` in the first epochs and adjust. Each run also stops at its `hours=` budget, so set
it to fit the session.

## 3. How to run

Use the same notebook, `notebooks/MBPO_HalfCheetah_Kaggle.ipynb`, in a **new Kaggle notebook**. Only
the settings cell changes:

```python
TRAIN_HOURS = 4.0
RUNS = [
    dict(name="mbpo_k5_seed0",   algo="mbpo", seed=0, gpu=0, steps=60_000, set={"rollout_schedule": [0, 1, 5, 5]}),
    dict(name="mbpo_k15_seed0",  algo="mbpo", seed=0, gpu=0, steps=60_000, set={"rollout_schedule": [0, 1, 15, 15]}),
    dict(name="mbpo_ens1_seed0", algo="mbpo", seed=0, gpu=1, steps=60_000, set={"ensemble_size": 1, "num_elites": 1}),
    dict(name="mbpo_ens3_seed0", algo="mbpo", seed=0, gpu=1, steps=60_000, set={"ensemble_size": 3, "num_elites": 2}),
]
```

Then use **Save Version → Save & Run All**. Any field of `mbpo/config.py` can go in `set=`. Locally,
the equivalent is `python train.py --algo mbpo --set rollout_schedule=[0,1,5,5] --total-env-steps 60000`.

Notes:
- `rollout_schedule = [min_epoch, max_epoch, min_k, max_k]`. A fixed k is `[0, 1, k, k]`.
- The model buffer grows with k (100k × k × 4 transitions). k = 15 gives about 6M rows, ~1 GB of
  GPU memory. That's fine on a T4, but avoid k > 25.
- For a fair comparison, keep everything else identical to the main run, including G = 20 and the seed.
- Kaggle's GPU quota is about 30 h/week, and the main run uses ~12 h of it. Check the remaining quota
  first.

## 4. Analyses that need no training

These all use files the main run saves (in `mbpo_results.zip`):

| file | contents |
|---|---|
| `runs/<run>/progress.csv` | one row per 1000 env steps: eval return (mean/std over 5 episodes), SAC losses, α, entropy, Q mean, model holdout MSE, model error on recent data, rollout epistemic/aleatoric std, timings |
| `runs/<run>/snapshots/step_XXXXXX.pt` | every 25k steps: actor, critics, dynamics model (+ input scaler, elites) |
| `runs/<run>/final.pt`, `best.pt` | same format, end of training / best evaluation |
| `runs/<run>/env_buffer.npz` | every real transition: `obs, act, rew, next_obs, done` |
| `runs/<run>/tb/` | TensorBoard logs (`tensorboard --logdir runs`) |

Loading a checkpoint:

```python
import torch
from mbpo.sac import SAC
from mbpo.dynamics import ProbabilisticEnsemble

ckpt = torch.load("runs/mbpo_seed0/final.pt", map_location="cpu")
cfg = ckpt["config"]
agent = SAC(ckpt["obs_dim"], ckpt["act_dim"], torch.device("cpu"), hidden_dim=cfg["hidden_dim"])
agent.load_state_dict(ckpt["agent"])
model = ProbabilisticEnsemble(ckpt["obs_dim"], ckpt["act_dim"], cfg["ensemble_size"], cfg["num_elites"],
                              cfg["model_hidden_dim"], cfg["model_num_layers"], cfg["model_weight_decay"])
model.load_state_dict(ckpt["model"])
model.scaler.mu, model.scaler.std = ckpt["model_scaler"]
model.elites = ckpt["model_elites"]
# model.step(obs, act) -> (next_obs, reward); agent.act(obs, deterministic=True) -> action
```

Recommended analyses, in order of value for the slides:

1. **Model error vs. rollout horizon.** This is the main evidence for "short rollouts". Take real
   trajectories from `env_buffer.npz`, or roll out the final policy in the real env. From states along
   them, run the model open-loop for h = 1…30 steps with the *same actions*, and plot the state/reward
   error against h. Use several snapshots (25k, 100k, final) to show that the model gets better but
   errors still compound.
2. **Epistemic uncertainty vs. horizon / off-distribution.** Plot ensemble disagreement along the same
   rollouts. It should grow with h, which is why ensembles help.
3. **Q-value calibration.** For the final policy, compare the critic's Q(s, a) at the start states with
   the actual discounted return from the real env (γ = 0.99). This shows whether the critic over- or
   under-estimates.
4. **Predicted vs. real reward.** A scatter plot for 1-step predictions on held-out real data. It's an
   easy, intuitive picture of model quality.
5. **Ablation figure.** Use `python plot_results.py --runs <folder with ablation runs + main runs> --out results_ablation`.
   It groups runs by name. Also make a bar chart of return at 60k steps per configuration. Include the
   main MBPO runs truncated to 60k as the baseline bar.

Treat 1, 3 and 5 as the core; 2 and 4 are nice to have.

## 5. What to hand back

- The plots (PNG), plus a small table: configuration → return at 30k / 60k steps, mean ± std where
  there are 2 seeds.
- 2–3 sentences per ablation: what changed, what happened, why. Tie each one to the paper's argument.
  For example: longer k should help only while model error stays small; HalfCheetah's paper setting
  is k = 1.
- The analysis notebook (`.ipynb`) with outputs saved, so it can go in the submission zip.

## 6. Pitfalls

- **Compare at equal environment steps**, not equal wall-clock time or equal epochs, unless the plot is
  explicitly about compute.
- One seed is noisy. With only one seed, say so on the slide and don't over-interpret differences
  under ~10–15%.
- `best.pt` is selected by evaluation return, which is mildly optimistic. Report `final_policy_return`
  from `summary.json` as the headline number, and mention best.
- Don't edit files in `runs/` from the main run. Copy them first.
- After changing any `.py` file, rebuild the Kaggle notebook with `python tools/build_kaggle_notebook.py`.
  Otherwise the notebook still contains the old code.
