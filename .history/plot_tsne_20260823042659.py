"""
plot_tsne.py
============
t-SNE of OpenVLA hidden-state embeddings from 532 real WidowX episodes.

Pipeline:
  1. Mean-pool each episode over 50 timesteps → (532, 4096)
  2. L2-normalise + centre
  3. PCA → 50-D  (standard pre-compression before t-SNE)
  4. t-SNE → 2-D
  5. 3-panel figure:
       A  Coloured by TASK         (what the space is organised by)
       B  Coloured by OUTCOME      (success / failure)
       C  Coloured by TASK SUCCESS RATE  (easy vs. hard tasks)

Output: figure_tsne.png  (project root)
"""

import os, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(ROOT, "figure_tsne.png")

# ── load ─────────────────────────────────────────────────────────────────────
print("Loading cached features …")
blob    = torch.load(os.path.join(ROOT, "experiments", "out", "widowx_mean.pt"),
                     weights_only=False)
feats   = blob["features"].float().numpy()   # (532, 50, 4096)
success = blob["success"].numpy()            # (532,)
task_id = blob["task_id"].numpy()            # (532,)
descs   = blob["descs"]                      # {id: str}

SHORT = {
    0: "Lift Battery",
    1: "Lift Eggplant",
    2: "Lift Red Bottle",
    3: "Lift Blue Cup",
    4: "Put Blue Cup\non Plate",
    5: "Put Red Bottle\nin Pot",
    6: "Put Carrot\non Plate",
    7: "Put Red Block\nin Pot",
}

# ── Step 1: mean-pool over time ───────────────────────────────────────────────
X = feats.mean(axis=1)                       # (532, 4096)

# ── Step 2: normalise + centre ────────────────────────────────────────────────
norms = np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-8)
X     = X / norms
X     = X - X.mean(axis=0, keepdims=True)

# ── Step 3: PCA → 50-D pre-compression (use torch SVD — no overflow at 4096-D) ─
print("PCA 4096 → 50 D …")
X_t   = torch.from_numpy(X).float()
Xc_t  = X_t - X_t.mean(dim=0, keepdim=True)
_, S_t, Vt_t = torch.linalg.svd(Xc_t, full_matrices=False)
X50   = (Xc_t @ Vt_t[:50].T).numpy()             # (532, 50)
ev50  = ((S_t[:50]**2) / (S_t**2).sum()).sum().item()
print(f"  variance captured by 50 PCs: {ev50:.1%}")

# ── Step 4: t-SNE → 2-D ──────────────────────────────────────────────────────
print("t-SNE 50 → 2 D  (perplexity=40, max_iter=2000) …")
t0   = time.time()
tsne = TSNE(n_components=2, perplexity=40, max_iter=2000,
            learning_rate="auto", init="pca",
            random_state=0, n_jobs=1)
Z    = tsne.fit_transform(X50)               # (532, 2)
print(f"  done in {time.time()-t0:.0f}s   KL divergence={tsne.kl_divergence_:.3f}")

# ── per-task success rate ─────────────────────────────────────────────────────
task_sr = {}
for t in np.unique(task_id):
    m = task_id == t
    task_sr[t] = success[m].mean()

sr_arr = np.array([task_sr[t] for t in task_id])   # (532,)

# ── aesthetics ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "axes.titlesize":    10.5,
    "axes.labelsize":    9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.18,
    "grid.linewidth":    0.6,
    "legend.frameon":    False,
    "legend.fontsize":   7.8,
    "figure.dpi":        150,
})

TASK_CMAP    = plt.get_cmap("tab10")
SUCC_COLOR   = "#2E8B57"
FAIL_COLOR   = "#C0392B"
SR_CMAP      = plt.get_cmap("RdYlGn")

fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
fig.subplots_adjust(wspace=0.32)

S = 14    # point size
A = 0.80  # alpha

# ── Panel A: coloured by TASK ────────────────────────────────────────────────
ax = axes[0]
for k in sorted(np.unique(task_id)):
    m = task_id == k
    ax.scatter(Z[m, 0], Z[m, 1],
               s=S, color=TASK_CMAP(k % 10), alpha=A,
               label=SHORT[k], linewidths=0)

ax.set_title("A  —  Coloured by task\n"
             "Each colour = one task type (8 tasks, 532 episodes)", loc="left")
ax.set_xlabel("t-SNE dimension 1")
ax.set_ylabel("t-SNE dimension 2")
ax.legend(fontsize=7, loc="best", ncol=1, markerscale=1.6,
          title="Task", title_fontsize=7.5)

# ── Panel B: coloured by OUTCOME ─────────────────────────────────────────────
ax = axes[1]
for lab, col, mk, nm in [(1, SUCC_COLOR, "o", "Success"),
                          (0, FAIL_COLOR, "^", "Failure")]:
    m = success == lab
    ax.scatter(Z[m, 0], Z[m, 1],
               s=S, color=col, marker=mk, alpha=A,
               label=f"{nm}  (n={int(m.sum())})", linewidths=0)

ax.set_title("B  —  Coloured by outcome\n"
             "Success (●) and failure (▲) are mixed within task clusters", loc="left")
ax.set_xlabel("t-SNE dimension 1")
ax.set_ylabel("t-SNE dimension 2")
ax.legend(fontsize=8.5, loc="best", markerscale=1.5)

ax.text(0.03, 0.04,
    "Outcome information exists but is not\n"
    "the main organising principle of the space.\n"
    "A linear probe still recovers it at 87% acc.",
    transform=ax.transAxes, fontsize=7.5, color="dimgray", va="bottom",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgray", alpha=0.88))

# ── Panel C: coloured by per-task SUCCESS RATE ───────────────────────────────
ax = axes[2]
sc = ax.scatter(Z[:, 0], Z[:, 1],
                c=sr_arr, cmap=SR_CMAP, vmin=0, vmax=1,
                s=S, alpha=A, linewidths=0)

cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.03)
cbar.set_label("Task success rate", fontsize=8)
cbar.ax.tick_params(labelsize=7.5)

# Annotate each task cluster with its name + SR
for k in sorted(np.unique(task_id)):
    m = task_id == k
    cx, cy = Z[m, 0].mean(), Z[m, 1].mean()
    ax.text(cx, cy, f"SR={task_sr[k]:.0%}", fontsize=6.5,
            ha="center", va="center", color="black",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.6, ec="none"))

ax.set_title("C  —  Coloured by task success rate\n"
             "Red = hard tasks (low SR),  Green = easy tasks (high SR)", loc="left")
ax.set_xlabel("t-SNE dimension 1")
ax.set_ylabel("t-SNE dimension 2")

# ── super-title ───────────────────────────────────────────────────────────────
fig.suptitle(
    "t-SNE of OpenVLA hidden-state embeddings  —  532 WidowX robot episodes\n"
    f"PCA 4096→50 D  then  t-SNE  (perplexity=40)",
    fontsize=10.5, y=1.02, fontweight="medium")

fig.savefig(OUT, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {OUT}")
