"""
The PCA figure, done honestly: the same embedding space coloured two ways.

The earlier attempt's PCA plot was captioned "Separation confirms the 4096-D
representations carry discriminative outcome information." The clusters in it are
tasks, not outcomes. Measured on these embeddings, outcome accounts for ~2% of
total variance while the top two principal components are dominated by task
identity -- so an outcome split simply cannot be visible in PC1/PC2, whether or
not the information is present (it is: a linear probe recovers it).

Colouring the identical projection by task and by outcome side by side makes the
real structure legible, and turns a misleading figure into direct evidence for
why zero-shot conformal calibration fails.

Run:  python3 experiments/make_figure_pca.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "experiments", "out")

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130,
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "legend.frameon": False,
})

blob = torch.load(os.path.join(OUT, "widowx_mean.pt"), weights_only=False)
feats, succ, tid = blob["features"], blob["success"], blob["task_id"]
descs = blob["descs"]

# mean-pool each episode over time, as the earlier figure did
X = feats.mean(dim=1).numpy().astype(np.float64)          # (532, 4096)
y = succ.numpy()
t = tid.numpy()

Xc = X - X.mean(0, keepdims=True)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
Z = Xc @ Vt[:2].T                                          # (532, 2)
ev = (S ** 2) / (S ** 2).sum()


def between_group_var(X, g):
    mu = X.mean(0); tot = ((X - mu) ** 2).sum()
    b = sum(int((g == k).sum()) * ((X[g == k].mean(0) - mu) ** 2).sum()
            for k in np.unique(g))
    return b / tot


v_task = between_group_var(X, t)
v_out = between_group_var(X, y)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

# --- coloured by task -------------------------------------------------------
ax = axes[0]
cmap = plt.get_cmap("tab10")
for k in sorted(np.unique(t)):
    m = t == k
    ax.scatter(Z[m, 0], Z[m, 1], s=22, color=cmap(k % 10), alpha=0.85,
               label=f"{k}: {descs[k]}", linewidths=0)
ax.set_title(f"Coloured by TASK — clean separation\n"
             f"between-task variance = {v_task:.1%} of total")
ax.set_xlabel(f"PC 1 ({ev[0]:.1%} var.)")
ax.set_ylabel(f"PC 2 ({ev[1]:.1%} var.)")
ax.legend(fontsize=7, loc="best", ncol=1)

# --- coloured by outcome, same projection -----------------------------------
ax = axes[1]
for lab, col, mk, nm in [(1, "#2E8B57", "o", "success"), (0, "#B4413C", "^", "failure")]:
    m = y == lab
    ax.scatter(Z[m, 0], Z[m, 1], s=22, color=col, marker=mk, alpha=0.7,
               label=f"{nm} (n={int(m.sum())})", linewidths=0)
ax.set_title(f"Coloured by OUTCOME — thoroughly mixed\n"
             f"between-outcome variance = {v_out:.1%} of total")
ax.set_xlabel(f"PC 1 ({ev[0]:.1%} var.)")
ax.set_ylabel(f"PC 2 ({ev[1]:.1%} var.)")
ax.legend(fontsize=8, loc="best")

fig.suptitle("The same PCA of mean-pooled OpenVLA hidden states, coloured two ways "
             "(532 real WidowX episodes, 8 tasks)", fontsize=10, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig10_pca_task_vs_outcome.png"), bbox_inches="tight")
plt.close(fig)

print("wrote fig10_pca_task_vs_outcome.png")
print(f"  PC1 {ev[0]:.1%}, PC2 {ev[1]:.1%}  (top-2 total {ev[:2].sum():.1%})")
print(f"  between-TASK variance    {v_task:.1%} of total")
print(f"  between-OUTCOME variance {v_out:.1%} of total")
print(f"  ratio task/outcome       {v_task / max(v_out, 1e-9):.1f}x")
