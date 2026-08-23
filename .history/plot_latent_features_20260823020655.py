"""
plot_latent_features.py
=======================
Comprehensive latent feature diagram — 4 panels:

  Panel A  PCA coloured by TASK        (shows what PC1/PC2 actually captures)
  Panel B  PCA coloured by OUTCOME     (shows why task-shift breaks calibration)
  Panel C  Score timeline              (mean failure score vs. timestep, by outcome)
  Panel D  Feature variance breakdown  (what drives the feature space)

Run: python3 plot_latent_features.py
Output: figure_latent_features.png  (project root)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import torch

# ── load cached features ─────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
blob = torch.load(os.path.join(ROOT, "experiments", "out", "widowx_mean.pt"),
                  weights_only=False)

feats   = blob["features"].float().numpy()    # (532, 50, 4096)
success = blob["success"].numpy()             # (532,)  1=success 0=failure
task_id = blob["task_id"].numpy()             # (532,)
descs   = blob["descs"]                       # {task_id: str}

# ── aesthetics ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.22,
    "grid.linewidth":    0.6,
    "legend.frameon":    False,
    "legend.fontsize":   8,
    "figure.dpi":        150,
})

TASK_CMAP   = plt.get_cmap("tab10")
SUCC_COLOR  = "#2E8B57"
FAIL_COLOR  = "#C0392B"

# Short task labels for legend
SHORT = {
    0: "Lift Battery",
    1: "Lift Eggplant",
    2: "Lift Red Bottle",
    3: "Lift Blue Cup",
    4: "Put Blue Cup on Plate",
    5: "Put Red Bottle in Pot",
    6: "Put Carrot on Plate",
    7: "Put Red Block in Pot",
}

# ════════════════════════════════════════════════════════════════════════════
# PCA on mean-pooled embeddings
# ════════════════════════════════════════════════════════════════════════════
X  = feats.mean(axis=1)                       # (532, 4096) float32
# L2-normalise rows, then centre — use torch (handles large dims without overflow)
X_t   = torch.from_numpy(X)
norms = X_t.norm(dim=1, keepdim=True).clamp(min=1e-8)
X_t   = X_t / norms
Xc_t  = X_t - X_t.mean(dim=0, keepdim=True)

_, S_t, Vt_t = torch.linalg.svd(Xc_t, full_matrices=False)
Z_t   = Xc_t @ Vt_t[:2].T                    # (532, 2)
ev_t  = (S_t ** 2) / (S_t ** 2).sum()

Z  = Z_t.numpy().astype(np.float64)
S  = S_t.numpy()
ev = ev_t.numpy()
Vt = Vt_t.numpy()
X  = X_t.numpy()                              # normalised, for variance decomp


# Variance decomposition
def between_group_var(X, g):
    mu  = X.mean(0)
    tot = ((X - mu) ** 2).sum()
    b   = sum(int((g == k).sum()) * ((X[g == k].mean(0) - mu) ** 2).sum()
              for k in np.unique(g))
    return b / tot

v_task    = between_group_var(X, task_id)
v_outcome = between_group_var(X, success)
v_pc12    = float(ev[:2].sum())

# ════════════════════════════════════════════════════════════════════════════
# Score timeline: mean LSTM baseline score per timestep, grouped by outcome
# (loaded from the saved pkl)
# ════════════════════════════════════════════════════════════════════════════
import pickle

with open(os.path.join(ROOT, "baseline_val_scores.pkl"), "rb") as f:
    baseline_scores = pickle.load(f)
with open(os.path.join(ROOT, "modified_val_scores.pkl"), "rb") as f:
    modified_scores = pickle.load(f)

def mean_timeline(scores_dict, label_key=0, T=50):
    """Average score over time across all trajectories in a given outcome class."""
    timelines = []
    for split_data in scores_dict.values():
        scores_list, labels_arr = split_data
        for s, lbl in zip(scores_list, labels_arr):
            if int(lbl) == label_key and len(s) > 0:
                padded = np.full(T, np.nan)
                padded[:len(s)] = s
                timelines.append(padded)
    arr = np.array(timelines)
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)

t = np.arange(50)

base_fail_mean, base_fail_std = mean_timeline(baseline_scores, label_key=0)
base_succ_mean, base_succ_std = mean_timeline(baseline_scores, label_key=1)
mod_fail_mean,  mod_fail_std  = mean_timeline(modified_scores, label_key=0)
mod_succ_mean,  mod_succ_std  = mean_timeline(modified_scores, label_key=1)

# ════════════════════════════════════════════════════════════════════════════
# Build the figure: 2-row layout
#   Top row:    Panel A (PCA by task) | Panel B (PCA by outcome)
#   Bottom row: Panel C (score timeline) | Panel D (variance bar chart)
# ════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(13, 9))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.30)

ax_task    = fig.add_subplot(gs[0, 0])
ax_outcome = fig.add_subplot(gs[0, 1])
ax_time    = fig.add_subplot(gs[1, 0])
ax_var     = fig.add_subplot(gs[1, 1])

# ── Panel A: PCA by TASK ──────────────────────────────────────────────────
for k in sorted(np.unique(task_id)):
    m = task_id == k
    ax_task.scatter(Z[m, 0], Z[m, 1],
                    s=18, color=TASK_CMAP(k % 10), alpha=0.82,
                    label=SHORT[k], linewidths=0)

ax_task.set_title(
    f"A  —  PCA coloured by task\n"
    f"Between-task variance = {v_task:.0%} of total",
    loc="left", pad=6)
ax_task.set_xlabel(f"PC 1  ({ev[0]:.1%} of variance)")
ax_task.set_ylabel(f"PC 2  ({ev[1]:.1%} of variance)")
ax_task.legend(fontsize=7, loc="best", ncol=1, markerscale=1.4,
               title="Task", title_fontsize=7)

# ── Panel B: PCA by OUTCOME ───────────────────────────────────────────────
for lab, col, mk, nm in [(1, SUCC_COLOR, "o", "Success"),
                          (0, FAIL_COLOR, "^", "Failure")]:
    m = success == lab
    ax_outcome.scatter(Z[m, 0], Z[m, 1],
                       s=18, color=col, marker=mk, alpha=0.65,
                       label=f"{nm}  (n={int(m.sum())})", linewidths=0)

ax_outcome.set_title(
    f"B  —  Same PCA coloured by outcome\n"
    f"Between-outcome variance = {v_outcome:.1%} of total  "
    f"(task is {v_task/v_outcome:.0f}× stronger signal)",
    loc="left", pad=6)
ax_outcome.set_xlabel(f"PC 1  ({ev[0]:.1%} of variance)")
ax_outcome.set_ylabel(f"PC 2  ({ev[1]:.1%} of variance)")
ax_outcome.legend(fontsize=8, loc="best", markerscale=1.4)

# Add note explaining the mixing
ax_outcome.text(0.03, 0.04,
    "Success & failure are mixed inside every task cluster.\n"
    "Outcome information exists but lives in lower-variance directions.",
    transform=ax_outcome.transAxes, fontsize=7.5,
    color="dimgray", va="bottom",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgray", alpha=0.85))

# ── Panel C: Score timeline ───────────────────────────────────────────────
# Baseline
ax_time.plot(t, base_fail_mean, color=FAIL_COLOR,   lw=2.0, ls="--",
             label="Baseline — failure")
ax_time.fill_between(t,
    np.clip(base_fail_mean - base_fail_std, 0, 1),
    np.clip(base_fail_mean + base_fail_std, 0, 1),
    color=FAIL_COLOR, alpha=0.12)

ax_time.plot(t, base_succ_mean, color=SUCC_COLOR, lw=2.0, ls="--",
             label="Baseline — success")
ax_time.fill_between(t,
    np.clip(base_succ_mean - base_succ_std, 0, 1),
    np.clip(base_succ_mean + base_succ_std, 0, 1),
    color=SUCC_COLOR, alpha=0.12)

# Modified
ax_time.plot(t, mod_fail_mean, color=FAIL_COLOR,   lw=2.2, ls="-",
             label="Contrastive — failure")
ax_time.fill_between(t,
    np.clip(mod_fail_mean - mod_fail_std, 0, 1),
    np.clip(mod_fail_mean + mod_fail_std, 0, 1),
    color=FAIL_COLOR, alpha=0.10)

ax_time.plot(t, mod_succ_mean, color=SUCC_COLOR, lw=2.2, ls="-",
             label="Contrastive — success")
ax_time.fill_between(t,
    np.clip(mod_succ_mean - mod_succ_std, 0, 1),
    np.clip(mod_succ_mean + mod_succ_std, 0, 1),
    color=SUCC_COLOR, alpha=0.10)

ax_time.axvline(25, color="gray", lw=1.0, ls=":", alpha=0.7)
ax_time.text(25.5, 0.97, "mid-episode\n(detectors near chance here)",
             fontsize=7, color="gray", va="top")

ax_time.set_xlim(0, 49)
ax_time.set_ylim(-0.02, 1.05)
ax_time.set_xlabel("Timestep  t")
ax_time.set_ylabel("Mean failure score  (± 1 std)")
ax_time.set_title(
    "C  —  Average failure score over time\n"
    "Dashed = baseline (BCE),  Solid = contrastive loss",
    loc="left", pad=6)
ax_time.legend(fontsize=7.5, ncol=2, loc="upper left")

# ── Panel D: Variance breakdown bar chart ─────────────────────────────────
categories = [
    ("Task identity\n(31%)", v_task,    "#4C72B0"),
    ("Outcome\n(2%)",        v_outcome, "#DD8452"),
    ("PC1 + PC2\n(top-2)",  v_pc12,    "#8172B2"),
]
labels, vals, cols = zip(*categories)
bars = ax_var.barh(range(len(labels)), [v * 100 for v in vals],
                   color=cols, height=0.5, alpha=0.88, edgecolor="white")

for bar, val in zip(bars, vals):
    ax_var.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontsize=9, color="dimgray")

ax_var.set_yticks(range(len(labels)))
ax_var.set_yticklabels(labels, fontsize=9)
ax_var.set_xlabel("% of total feature variance explained")
ax_var.set_xlim(0, max(v * 100 for v in vals) * 1.25)
ax_var.set_title(
    "D  —  What drives the 4096-D embedding space?\n"
    "Task identity dominates; outcome is a weak, low-variance direction",
    loc="left", pad=6)

ax_var.text(0.97, 0.08,
    "Outcome info IS present\n(linear probe: 0.876 acc.)\nbut invisible to PCA",
    transform=ax_var.transAxes, fontsize=8, color="dimgray",
    ha="right", va="bottom",
    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="lightgray", alpha=0.9))

# ── Super-title ───────────────────────────────────────────────────────────
fig.suptitle(
    "Latent Feature Space of OpenVLA Hidden States  —  532 real WidowX robot episodes, 8 tasks",
    fontsize=11, y=1.01, fontweight="medium")

# ── Save ──────────────────────────────────────────────────────────────────
out_path = os.path.join(ROOT, "figure_latent_features.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")
print(f"  PC1={ev[0]:.1%}, PC2={ev[1]:.1%}, PC1+PC2={v_pc12:.1%}")
print(f"  Between-task variance:    {v_task:.1%}")
print(f"  Between-outcome variance: {v_outcome:.1%}")
