# SAFE Probe Comparative Experiment: Baseline vs. Intra-Trajectory Contrastive Loss

## Overview

This experiment compares two failure detection probes on the pre-extracted OpenVLA/WidowX rollout features:

1. **Baseline SAFE Probe** (`failure_prob/model/lstm.py`): Uses the existing time-weighted BCE/hinge loss.
2. **Modified Probe** (`train_modified.py`): Replaces the per-timestep loss with an **intra-trajectory contrastive loss** (Hide-and-Seek inspired) to fix label smearing on failure trajectories.

Both models will be trained and evaluated without `wandb` cloud sync, saving checkpoints and validation scores locally. Two publication-quality figures will be generated.

---

## Key Design Decisions

> [!IMPORTANT]
> **MPS device** — All tensor ops use `mps` (Apple M4 Pro) or fall back to `cpu`. No `cuda` calls anywhere. We must patch `train.py`'s `model.to("cuda")` and the dataset's `load_to_cuda` flag.

> [!IMPORTANT]
> **Self-contained scripts** — Rather than modifying `train.py` in-place (which uses Hydra + wandb), we write two standalone Python scripts (`train_baseline.py`, `train_modified.py`) that directly call the existing model/data utilities without Hydra, and run wandb in offline mode. This avoids breaking existing infrastructure.

> [!NOTE]
> **Data** — The `openvla_widowx` directory already has pre-extracted CSV + PKL feature files in the exact format the `openvla.py` loader expects. The CSV filename pattern is `task_<name>--ep<N>--succ<0|1>.csv` with a matching `.pkl`.

---

## Proposed Changes

### Step 1: Baseline Training Script

#### [NEW] `train_baseline.py` (workspace root)
- Standalone script (no Hydra) that loads `openvla_widowx` rollouts via `failure_prob.data.openvla`
- Builds `LstmModel` with config matching `LstmModelConfig` defaults
- Trains using the **existing time-weighted loss** in `lstm.py`
- Device: `mps` (falls back to `cpu`)
- Saves:
  - `baseline_model.ckpt` — model weights
  - `baseline_val_scores.pkl` — dict of `{split: [np.ndarray per rollout]}` of failure scores
  - `baseline_features.pkl` — per-rollout last-hidden-state embeddings for PCA/t-SNE

---

### Step 2: Intra-Trajectory Contrastive Loss

#### [NEW] `train_modified.py` (workspace root)
- Copy of the baseline training loop, but with a **modified loss function** plugged in at the `forward_compute_loss` step
- The new `intra_contrastive_loss(scores, valid_masks, success_labels, margin=0.5)` function:
  1. Iterates over failure trajectories (where `success_labels == 0`)
  2. Computes step-to-step score differences: `diff[t] = scores[t+1] - scores[t]`
  3. Finds proxy onset step: `k = argmax(diff).detach()`  ← gradient-stopped index selection
  4. Computes `pre_mean = scores[:k].mean()`, `post_mean = scores[k:].mean()`
  5. Applies ReLU margin loss: `relu(margin - (post_mean - pre_mean))`
  6. Averages over all failure trajectories
- Also retains the baseline success-side BCE loss (to keep gradients flowing on success trajectories)
- Saves:
  - `modified_model.ckpt`
  - `modified_val_scores.pkl`
  - `modified_features.pkl`

---

### Step 3: Visualization Script

#### [NEW] `generate_figures.py` (workspace root)
Produces two `.png` figures in the workspace root:

**Figure 1 — `figure_trajectory_comparison.png`**
- Line graph: failure score vs. time step `t`
- Two lines on same axis: baseline (blue, dashed) and modified (red, solid)
- Both from the same failed validation rollout (selected for visual clarity)
- Expected pattern: baseline smears gradually, modified stays flat → sharp spike at onset

**Figure 2 — `figure_latent_space_pca.png`**  
- 2D PCA scatter of mean hidden-state embeddings
- Green = success rollouts, red = failure rollouts
- From the validation set to show separability

---

## Verification Plan

### Automated Checks
- Scripts complete without error and produce the four `.pkl` artifacts + two `.ckpt` files
- Both figures render correctly as `.png`

### Manual Verification
- Figure 1 visual pattern matches the expected "spike vs. smear" narrative
- Figure 2 shows some cluster separation between success/failure

---

## Open Questions

> [!NOTE]
> The openvla_widowx dataset has a non-standard filename pattern: `task_<desc>--ep<N>--succ<0|1>.csv` (not `task<ID>--ep<ID>--succ<0|1>.csv`). The existing `extract_info_from_path` regex won't match. We will handle this by parsing the task name from the directory name for task_id and episode/success from the filename.

> [!NOTE]
> The LSTM with `-1` history steps (full sequence mode) may be slow on MPS for long rollouts; we'll use `n_epochs=100` and `batch_size=16` for a fast but meaningful training run on this dataset size (~400 rollouts).
