"""Figures for the label-smearing ablation. Run smearing_experiment.py first."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "experiments", "out")

ORDER = [
    "SAFE (cumsum, published)",
    "SAFE + time weighting",
    "SAFE loss, no cumsum",
    "H&S inter only",
    "H&S intra only",
    "H&S inter+intra",
    "H&S inter+intra, cumsum",
]
COLORS = {
    "SAFE (cumsum, published)": "#B4413C",
    "SAFE + time weighting":    "#D98C7A",
    "SAFE loss, no cumsum":     "#E0A030",
    "H&S inter only":           "#7FA8C9",
    "H&S intra only":           "#9A9A9A",
    "H&S inter+intra":          "#2E6E8E",
    "H&S inter+intra, cumsum":  "#5B4E8C",
}

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130,
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "legend.frameon": False,
})

d = np.load(os.path.join(OUT, "curves_seed0.npz"))
masks, succ, onset = d["masks"], d["succ"], d["onset"]
val_unseen = d["val_unseen"]
lens = masks.sum(1).astype(int)
score_keys = {k[len("scores::"):]: d[k] for k in d.files if k.startswith("scores::")}

rows = json.load(open(os.path.join(OUT, "results.json")))
df = pd.DataFrame(rows)
uns = df[df.split == "val_unseen"]


def norm_scores(s, idx):
    """Min-max to [0,1] on the 1st/99th percentile of this variant's own eval
    scores, so variants living on different scales (cumsum grows to ~T, per-step
    sigmoid is in [0,1]) can share an axis."""
    pool = np.concatenate([s[i, :lens[i]] for i in idx])
    lo, hi = np.percentile(pool, 1), np.percentile(pool, 99)
    return (s - lo) / max(hi - lo, 1e-9)


# =====================================================================
# FIG 1 - onset-aligned mean score profile
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
GRID = np.linspace(-0.5, 0.4, 60)   # relative time from the true onset

for panel, variants, title in [
    (0, ["SAFE (cumsum, published)", "SAFE loss, no cumsum"],
        "Effect of AGGREGATION\n(SAFE's loss held fixed)"),
    (1, ["SAFE loss, no cumsum", "H&S inter+intra"],
        "Effect of LOSS\n(aggregation held fixed at per-step)"),
]:
    ax = axes[panel]
    for v in variants:
        s = norm_scores(score_keys[v], val_unseen)
        prof = []
        for g in GRID:
            vals = []
            for i in val_unseen:
                if succ[i] == 1:
                    continue
                L, on = lens[i], onset[i]
                t = int(round(on + g * L))
                if 0 <= t < L:
                    vals.append(s[i, t])
            prof.append(np.mean(vals) if vals else np.nan)
        ax.plot(GRID, prof, lw=2.0, color=COLORS[v], label=v)

    ax.axvline(0, color="k", ls="--", lw=1.2)
    ax.annotate("true failure onset", xy=(0, 0.03), xytext=(0.02, 0.03),
                fontsize=8, color="k")
    ax.set_xlabel("time relative to true failure onset  (fraction of rollout)")
    ax.set_ylabel("normalised failure score")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(0, 1.0)

fig.suptitle("Mean failure score of held-out-task FAILURE rollouts, aligned on the true onset "
             "(seed 0, n=%d unseen-task rollouts)" % len(val_unseen),
             fontsize=9.5, y=1.005)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_onset_aligned.png"), bbox_inches="tight")
plt.close(fig)
print("wrote fig1_onset_aligned.png")


# =====================================================================
# FIG 2 - the 2x2: which factor actually moves the metric
# =====================================================================
cells = [("SAFE (cumsum, published)", "SAFE loss", "cumsum"),
         ("SAFE loss, no cumsum",     "SAFE loss", "per-step"),
         ("H&S inter+intra, cumsum",  "H&S loss",  "cumsum"),
         ("H&S inter+intra",          "H&S loss",  "per-step")]

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
for ax, metric, label, ceiling in [
    (axes[0], "post_auc", "post-onset AUC  (higher = better)", 0.835),
    (axes[1], "bal_acc",  "trajectory balanced accuracy", None),
]:
    x = np.arange(2)
    w = 0.36
    for k, loss in enumerate(["SAFE loss", "H&S loss"]):
        means, stds = [], []
        for agg in ["cumsum", "per-step"]:
            v = [c[0] for c in cells if c[1] == loss and c[2] == agg][0]
            r = uns[uns.variant == v][metric]
            means.append(r.mean()); stds.append(r.std())
        ax.bar(x + (k - 0.5) * w, means, w, yerr=stds, capsize=3,
               color=["#B4413C", "#2E6E8E"][k], alpha=0.9, label=loss)
        for xi, m in zip(x + (k - 0.5) * w, means):
            ax.text(xi, m + 0.015, f"{m:.3f}", ha="center", fontsize=7.5)

    if ceiling:
        ax.axhline(ceiling, color="green", ls=":", lw=1.3)
        ax.text(1.35, ceiling + 0.008, "step-supervised ceiling", fontsize=7.5,
                color="green", ha="right")
    ax.axhline(0.5, color="grey", ls="--", lw=1.0)
    ax.text(-0.42, 0.51, "chance", fontsize=7.5, color="grey")
    ax.set_xticks(x); ax.set_xticklabels(["cumsum\n(SAFE default)", "per-step\n(no cumsum)"])
    ax.set_ylabel(label); ax.set_ylim(0.4, 0.95)
    ax.legend(fontsize=8, loc="upper left")

fig.suptitle("Loss x aggregation, crossed on one shared architecture — zero-shot on held-out tasks "
             "(mean ± sd, 3 seeds)", fontsize=9.5, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_factorial.png"), bbox_inches="tight")
plt.close(fig)
print("wrote fig2_factorial.png")


# =====================================================================
# FIG 3 - single failure rollout, all variants
# =====================================================================
fail_unseen = [i for i in val_unseen if succ[i] == 0]
# pick a rollout with a mid-trajectory onset for legibility
pick = sorted(fail_unseen, key=lambda i: abs(onset[i] / lens[i] - 0.55))[0]
L, on = lens[pick], onset[pick]

fig, ax = plt.subplots(figsize=(8.2, 3.8))
for v in ORDER:
    if v in ("H&S intra only", "SAFE + time weighting", "H&S inter only"):
        continue
    s = norm_scores(score_keys[v], val_unseen)[pick, :L]
    ax.plot(np.arange(L) / L, s, lw=2.0, color=COLORS[v], label=v)

ax.axvline(on / L, color="k", ls="--", lw=1.3)
ax.text(on / L + 0.01, 0.95, "true onset", fontsize=8)
ax.axvspan(0, on / L, color="green", alpha=0.05)
ax.axvspan(on / L, 1, color="red", alpha=0.05)
ax.text(0.02, 0.02, "nominal execution", fontsize=8, color="darkgreen")
ax.text(0.99, 0.02, "failure in progress", fontsize=8, color="darkred", ha="right")
ax.set_xlabel("relative time in rollout")
ax.set_ylabel("normalised failure score")
ax.set_title("One held-out-task failure rollout (task %d, ep %d)" % (0, pick))
ax.legend(fontsize=8, loc="upper left")
ax.set_ylim(0, 1.05)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_single_rollout.png"), bbox_inches="tight")
plt.close(fig)
print("wrote fig3_single_rollout.png")


# =====================================================================
# Console table
# =====================================================================
print("\n" + "=" * 92)
print("val_unseen, mean +/- sd over 3 seeds")
print("=" * 92)
print(f"{'variant':<28s}{'smearAUC':>14s}{'postAUC':>14s}{'balAcc':>14s}{'onsetErr':>14s}")
print("-" * 92)
for v in ORDER:
    r = uns[uns.variant == v]
    print(f"{v:<28s}"
          f"{r.smear_auc.mean():>8.3f}+-{r.smear_auc.std():<5.3f}"
          f"{r.post_auc.mean():>8.3f}+-{r.post_auc.std():<5.3f}"
          f"{r.bal_acc.mean():>8.3f}+-{r.bal_acc.std():<5.3f}"
          f"{r.onset_err.mean():>+8.3f}+-{r.onset_err.std():<5.3f}")
print("=" * 92)
