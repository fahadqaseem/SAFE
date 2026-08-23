# SAFE Failure Detection — Two-Pager

---

## The Simple Version

Imagine a robot arm doing chores — picking things up, placing objects. Sometimes it fails. The question is: **can software watching the robot's brain predict that it's about to fail, early enough to do something about it?**

The robot's brain is a large AI model (OpenVLA). Every time it decides what to do next, its internal state — a list of 4,096 numbers — gets recorded. Those numbers are the "brain activity" we use as a signal.

**What SAFE does**: It trains a small detector (like a second, simpler brain) that watches those numbers step-by-step and outputs a single "failure score" from 0 to 1. When the score crosses a threshold, the system raises an alarm. The threshold is set using a statistical technique called **conformal prediction**, which promises: *"I will give a false alarm on at most X% of successful runs."*

**What we tested**: Does that promise actually hold when the robot is doing a chore it has never seen before?

**The short answer**: No — it breaks, badly. On new tasks the scores are inflated even when nothing goes wrong, so the threshold sits in the wrong place. We then investigated *why*, and tested a fix.

**What we found, in plain terms**:

1. 🚨 **The alarm system breaks on new tasks.** One variant raised a false alarm on almost *every* successful run of a new task — the opposite of useful.

2. 🌡️ **The cause is like a thermometer that reads hot in every new room.** The failure score runs roughly 3× higher on unfamiliar tasks even during successful runs. The threshold, set on known tasks, becomes meaningless.

3. 🔧 **An adaptive fix works.** Instead of a fixed threshold, the detector watches its own false alarm rate and adjusts on the fly. The guarantee is restored.

4. ⏰ **No method warns early enough.** Halfway through every single attempt, every detector tested is near random chance. They only become accurate right at the end — when there is no time left to intervene. This is the most important honest limitation.

5. 🏋️ **Training the detector smarter helps — but less than expected.** A new "contrastive" training trick that forces the failure score to spike sharply at the moment of failure (instead of drifting gradually) improved detection accuracy significantly (+0.146 AUC). But SAFE's own best configuration, using a different architectural trick (cumulative score aggregation), performs comparably. The architecture matters at least as much as the training objective.

---

## The Technical Version

### Problem Statement

SAFE trains a neural probe on frozen VLA hidden states to produce a per-timestep failure probability. A functional conformal band built from held-out successful episodes provides a calibrated alarm threshold at target miscoverage rate α. We tested whether this guarantee survives task distribution shift — the deployment setting the paper targets — and investigated two failure modes: (1) threshold miscalibration under shift and (2) label smearing in the probe's training objective.

---

### Data

**Source**: `openvla_widowx/` — 532 real-robot episodes across **8 distinct tasks** (WidowX arm, OpenVLA policy), 244 success / 288 failure, each exactly 50 steps.

**Features**: OpenVLA hidden states, shape `(50, 7, 4096)` bfloat16 per episode (7 action tokens × 4096-dimensional transformer representations), reduced to a single 4096-vector per step by selecting the last action token — matching SAFE's default `token_idx_rel = 1.0`.

**Important data trap**: The 19 folder names collapse to only 8 real `task_id`s (e.g., `task_lift_red_bottle_1..4` are all `task_id=2`). Splitting on folder names leaks the same task into both train and test. All experiments here group by `task_id`.

**Not used** (scope): `rollouts_all/` — 46 GB Franka/DROID rollouts across 6 collection dates; no failure-onset annotations exist for WidowX, so onset localisation is only measurable on a synthetic testbed.

---

### Code Written

All new code lives in `experiments/` (~2,000 lines). SAFE's own source (`failure_prob/`) was called unmodified except for a single Python 3.13 compatibility fix.

| File | Role |
|------|------|
| `cache_widowx.py` | Converts 532 pickles → one tensor; validates episode counts and task IDs |
| `online_cp.py` | Implements Adaptive Conformal Inference (ACI), DtACI, SAFE's conformal scorer, and a task-relative score transform |
| `run_real_widowx.py` + `make_figures_real.py` | Conformal coverage experiment on real rollouts → Figs 4–6 |
| `smearing_experiment.py` + `make_figures.py` | Synthetic loss × aggregation factorial → Figs 1–3 |
| `run_experiments2.py` + `make_figures2.py` | Task-relative scoring, early detection, per-task breakdown → Figs 7–9 |
| `compare_gemini_loss.py` | Four-arm replication of the contrastive loss, 12 runs |
| `failure_prob/model/hns_loss.py` + `hns.py` | Hide-and-Seek inter/intra losses as a drop-in for SAFE's pipeline |
| `train_baseline.py`, `train_modified.py`, `generate_figures.py` | Initial standalone experiment scripts (project root) |

---

### Four Studies, in Order

**Study 1 — Synthetic loss × aggregation factorial** (`smearing_experiment.py`)

Generated synthetic rollouts with *known* failure onsets (WidowX has no annotations). Crossed loss type (SAFE hinge vs. Hide-and-Seek contrastive) against temporal aggregation (`cumsum` on vs. off). Result: removing `cumsum` alone moved post-onset AUC from **0.593 → 0.801**; swapping the loss at matched aggregation added nothing statistically significant. **Aggregation drives the gain, not the loss.**

**Study 2 — Conformal coverage under task shift** (`run_real_widowx.py`)

Ran SAFE's conformal pipeline with a true unseen-task holdout (3 tasks withheld, calibrated on 5). Realised coverage on unseen tasks: 0.710 / 0.567 / **0.001** vs nominal 0.80. The exchangeable control (held-out episodes of *seen* tasks) sits on the diagonal — confirming the effect is task-shift, not a code bug. Root cause: peak failure score on successful runs scales from 5.43 (seen tasks) to 15.29 (unseen) — a 3× shift that pushes all test points above the calibrated band. Applied Adaptive Conformal Inference (ACI) with a growing calibration pool: coverage recovers to **0.817 / 0.797 / 0.774**, standard deviation collapses from 0.148 → 0.029.

**Study 3 — Task-relative scoring and early detection** (`run_experiments2.py`)

Per-task AUROC spread: **0.69 (Lift Red Bottle) → 0.98 (Put Red Block into Pot)** — lifting failures are subtle (gripper closes on air), placing failures are gross. AUROC at mid-episode (first 50% of steps): all four probe variants cluster at **0.595–0.602**, statistically indistinguishable. No method provides actionable early warning. All seven handcrafted signals from episode CSVs score 0.39–0.60 AUROC vs ~0.80 for learned probes — supporting SAFE's core premise.

**Study 4 — Four-arm replication** (`compare_gemini_loss.py`)

12 runs (6 three-task holdouts × 2 seeds):

| Arm | Configuration | val\_unseen AUROC |
|-----|--------------|-------------------|
| A | SAFE-LSTM, BCE, no time-weighting | 0.646 ± 0.048 |
| B | SAFE-LSTM, BCE, time-weighted (original) | 0.630 ± 0.037 |
| **C** | **Intra-trajectory contrastive (this experiment)** | **0.792 ± 0.065** |
| D | Hide-and-Seek proper (inter + intra) | 0.765 ± 0.072 |
| E | SAFE-MLP, hinge + `cumsum` (SAFE's published default) | **0.832 ± 0.059** |

C beats A by **+0.146 (6.3 standard errors)** — real and replicates. C vs E: −0.040 (1.6 se) — not statistically separable, nominally behind. **The contrastive loss lifts SAFE's LSTM/BCE variant to approximately SAFE's best level. It does not exceed it.**

---

### The Intra-Trajectory Contrastive Loss (Core Contribution)

Standard SAFE training applies a per-timestep loss using the trajectory-level label, pushing every timestep's score toward 1 on failure trajectories (label smearing). The fix enforces a sharp pre→post step change without needing onset annotations:

```python
# For each failure trajectory:
diff      = scores[1:] - scores[:-1]          # step-to-step differences
k         = int(diff.detach().argmax()) + 1   # proxy onset — .detach() stops gradient
pre_mean  = scores[:k].mean()
post_mean = scores[k:].mean()
loss      = relu(0.5 - (post_mean - pre_mean))  # enforce post > pre by margin 0.5

# For each success trajectory:
loss = relu(scores).mean()                    # suppress scores toward 0
```

The `.detach()` on `argmax` is essential: without it, gradients act on the *position* of the sharpest transition rather than the *values* before and after it, destabilising training.

---

### Key Figures

| Figure | What It Shows | File |
|--------|--------------|------|
| **Fig 5** — calibration diagram | Guarantee holds on seen tasks (blue, on diagonal), fails on unseen (red, near zero), ACI restores it (teal) | `experiments/out/fig5_calibration.png` |
| **Fig 8** — early detection | AUROC vs. fraction of episode observed; all methods near chance at the halfway mark | `experiments/out/fig8_early_detection.png` |
| **Fig 9** — per-task + handcrafted | Task-level AUROC spread and why learned probes beat handcrafted signals | `experiments/out/fig9_pertask_handcrafted.png` |
| **Fig 2** — synthetic factorial | `cumsum` aggregation drives AUC gain more than loss type | `experiments/out/fig2_factorial.png` |
| **Fig 4** — rolling coverage | Coverage recovering live along deployment stream; dips at task boundaries | `experiments/out/fig4_rolling_coverage.png` |

---

### Summary of Findings

> SAFE's conformal false-alarm guarantee holds on calibration tasks and fails on unseen ones — the exact deployment setting the paper targets. The cause is a 3× task-dependent score offset. Adaptive conformal inference closes it. Along the way: temporal aggregation (`cumsum`) matters more than the loss function; no method gives actionable early warning at mid-episode; and the embedding space is organised by task identity (31% of variance), not outcome (2.1%), which explains why zero-shot calibration is hard.

### Honest Limitations

- Only 8 distinct tasks; 3-task holdouts are coarse
- No failure-onset annotations for WidowX — onset sharpness is only measurable on the synthetic testbed
- No hyperparameter search for any arm
- `rollouts_all/` (Franka, 6 collection dates) not yet analysed — temporal drift remains open
- ACI assumes episode outcome is observed after each episode (same assumption SAFE already makes for training)
