"""Figures for round 2. Run run_experiments2.py first."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

cp = pd.read_csv(os.path.join(OUT, "r2_cp.csv"))
early = pd.read_csv(os.path.join(OUT, "r2_early.csv"))
task = pd.read_csv(os.path.join(OUT, "r2_task.csv"))
ks = pd.read_csv(os.path.join(OUT, "r2_ksweep.csv"))
descs = torch.load(os.path.join(OUT, "widowx_mean.pt"), weights_only=False)["descs"]

DETS = ["SAFE-MLP", "H&S", "SAFE-Embed"]
NOM = 0.80
blocked = cp[(cp.order == "blocked") & (cp.alpha == 0.20)]
full = task[task.task == -1]

SPLIT, ACI = "split CP (SAFE)", "ACI g=0.05 (grow cal)"


# ======================================================================
# FIG 7 - the proposed fix: what it repairs and what it costs
# ======================================================================
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.1))

# (a) coverage under the four combinations
ax = axes[0]
combos = [
    ("split CP (SAFE's setup)",      SPLIT, "",            "#B4413C"),
    ("+ task-relative (no feedback)", SPLIT, " [task-rel]", "#E0A030"),
    ("+ ACI (needs feedback)",        ACI,   "",            "#2E6E8E"),
    ("+ both",                        ACI,   " [task-rel]", "#5B4E8C"),
]
x = np.arange(len(DETS)); w = 0.8 / len(combos)
for j, (lab, meth, suf, col) in enumerate(combos):
    mu, sd = [], []
    for d in DETS:
        s = blocked[(blocked.detector == d + suf) & (blocked.method == meth)].coverage
        mu.append(s.mean()); sd.append(s.std())
    ax.bar(x + (j - 1.5) * w, mu, w, yerr=sd, capsize=2, color=col, alpha=0.92, label=lab)
ax.axhline(NOM, color="green", ls="--", lw=1.5)
ax.text(-0.45, NOM + 0.02, "nominal 0.80", color="green", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(DETS, fontsize=8.5)
ax.set_ylabel("realised coverage on unseen tasks")
ax.set_ylim(0, 1.05); ax.set_title("What fixes the calibration")
ax.legend(fontsize=7.2, loc="lower left")

# (b) the cost: AUROC before/after the transform
ax = axes[1]
for j, (suf, col, lab) in enumerate([("", "#B4413C", "absolute score"),
                                     (" [task-rel]", "#E0A030", "task-relative score")]):
    mu = [full[full.detector == d + suf].auroc.mean() for d in DETS]
    sd = [full[full.detector == d + suf].auroc.std() for d in DETS]
    ax.bar(x + (j - 0.5) * 0.35, mu, 0.35, yerr=sd, capsize=3, color=col, alpha=0.92, label=lab)
    for xi, m in zip(x + (j - 0.5) * 0.35, mu):
        ax.text(xi, m + 0.012, f"{m:.3f}", ha="center", fontsize=7.5)
ax.axhline(0.5, color="k", ls="--", lw=1.1)
ax.text(-0.45, 0.512, "chance", fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels(DETS, fontsize=8.5)
ax.set_ylabel("trajectory AUROC, unseen tasks")
ax.set_ylim(0.4, 0.9); ax.set_title("What it costs")
ax.legend(fontsize=8)

# (c) k sweep - the reference window length
ax = axes[2]
ax2 = ax.twinx(); ax2.grid(False)
for d, col in zip(DETS, ["#B4413C", "#2E6E8E", "#E0A030"]):
    g = ks[ks.detector == d].groupby("k")
    ax.plot(g.auroc.mean().index, g.auroc.mean().values, "-o", color=col, lw=1.8, ms=4,
            label=f"{d} AUROC")
    ax2.plot(g.coverage.mean().index, g.coverage.mean().values, "--s", color=col,
             lw=1.3, ms=3.5, alpha=0.65)
ax.axhline(0.5, color="k", ls=":", lw=1.0)
ax2.axhline(NOM, color="green", ls="--", lw=1.3)
ax.set_xlabel("k  (leading steps treated as nominal, of 50)")
ax.set_ylabel("AUROC (solid)"); ax2.set_ylabel("coverage (dashed), nominal 0.80")
ax.set_title("Reference-window length")
ax.legend(fontsize=7.2, loc="lower right")

fig.suptitle("Task-relative scoring: subtract each episode's own early trend. "
             "Unseen tasks, alpha=0.20, mean ± sd over 12 runs.", fontsize=9.5, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig7_task_relative.png"), bbox_inches="tight")
plt.close(fig)
print("wrote fig7_task_relative.png")


# ======================================================================
# FIG 8 - early detection: is there time to stop the arm?
# ======================================================================
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1))

ax = axes[0]
show = [("SAFE-MLP", "#B4413C"), ("H&S", "#2E6E8E"), ("SAFE-Embed", "#E0A030"),
        ("hand:cum_mean_token_entropy", "#8A8A8A")]
for d, col in show:
    g = early[early.detector == d].groupby("frac").auroc
    mu, sd = g.mean(), g.std()
    ax.plot(mu.index, mu.values, "-o", color=col, lw=2.0, ms=4,
            label=d.replace("hand:", "handcrafted: "))
    ax.fill_between(mu.index, mu - sd, mu + sd, color=col, alpha=0.13)
ax.axhline(0.5, color="k", ls="--", lw=1.1)
ax.text(0.11, 0.512, "chance", fontsize=7.5)
ax.set_xlabel("fraction of the episode observed")
ax.set_ylabel("trajectory AUROC, unseen tasks")
ax.set_title("Detection quality vs how much you have seen")
ax.legend(fontsize=7.5, loc="upper left")

# how much of the final performance is available early?
ax = axes[1]
for d, col in show:
    g = early[early.detector == d].groupby("frac").auroc.mean()
    lift = (g - 0.5) / max(g.loc[1.0] - 0.5, 1e-9)
    ax.plot(g.index, lift.values, "-o", color=col, lw=2.0, ms=4,
            label=d.replace("hand:", "handcrafted: "))
ax.axhline(1.0, color="k", ls="--", lw=1.1)
ax.axhline(0.5, color="grey", ls=":", lw=1.0)
ax.text(0.11, 0.52, "half of the final signal", fontsize=7.5, color="grey")
ax.set_xlabel("fraction of the episode observed")
ax.set_ylabel("fraction of final above-chance signal")
ax.set_title("Only about half the signal exists by mid-episode")
ax.set_ylim(-0.2, 1.25)

fig.suptitle("Early detection on unseen tasks. Score = max over the observed prefix. "
             "Mean ± sd over 12 runs.", fontsize=9.5, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig8_early_detection.png"), bbox_inches="tight")
plt.close(fig)
print("wrote fig8_early_detection.png")


# ======================================================================
# FIG 9 - per-task difficulty, and the handcrafted baselines
# ======================================================================
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4),
                         gridspec_kw={"width_ratios": [1.15, 1]})

# (a) per-task AUROC for the learned detectors
ax = axes[0]
pt = task[task.task >= 0]
tasks = sorted(pt.task.unique())
order = sorted(tasks, key=lambda t: pt[(pt.detector == "SAFE-MLP") & (pt.task == t)].auroc.mean())
xx = np.arange(len(order))
for j, (d, col) in enumerate(zip(DETS, ["#B4413C", "#2E6E8E", "#E0A030"])):
    mu, sd = [], []
    for t in order:
        s = pt[(pt.detector == d) & (pt.task == t)].auroc
        mu.append(s.mean()); sd.append(s.std())
    ax.bar(xx + (j - 1) * 0.27, mu, 0.27, yerr=sd, capsize=2, color=col, alpha=0.92, label=d)
ax.axhline(0.5, color="k", ls="--", lw=1.1)
ax.set_xticks(xx)
ax.set_xticklabels([f"{descs[t]}\n(task {t}, n={int(pt[(pt.detector=='SAFE-MLP')&(pt.task==t)].n.mean())})"
                    for t in order], rotation=35, ha="right", fontsize=7)
ax.set_ylabel("AUROC on that held-out task")
ax.set_ylim(0.35, 1.03)
ax.set_title("A single average hides a 0.69 → 0.98 spread across chores")
ax.legend(fontsize=7.5, loc="upper left")

# (b) handcrafted vs learned: discrimination and calibration together
ax = axes[1]
names = [d for d in full.detector.unique() if "[task-rel]" not in d]
for d in names:
    a = full[full.detector == d].auroc.mean()
    g = blocked[(blocked.detector == d) & (blocked.method == SPLIT)].cov_gap.abs().mean()
    is_hand = d.startswith("hand:")
    ax.scatter(a, g, s=64 if not is_hand else 38,
               color="#8A8A8A" if is_hand else "#B4413C",
               marker="o" if not is_hand else "^",
               edgecolor="k" if not is_hand else "none", zorder=3, linewidths=0.6)
    lab = d.replace("hand:", "").replace("cum_", "cum ")
    ax.annotate(lab, (a, g), textcoords="offset points", xytext=(5, 3), fontsize=6.5)
ax.axvline(0.5, color="k", ls="--", lw=1.0)
ax.set_xlabel("trajectory AUROC (higher = better)")
ax.set_ylabel("|coverage gap| under split CP (lower = better)")
ax.set_title("Learned probes (red) vs 7 free handcrafted signals (grey)")
ax.annotate("better", xy=(0.80, 0.05), xytext=(0.66, 0.35), fontsize=8,
            arrowprops=dict(arrowstyle="->", lw=1.1))

fig.suptitle("Per-task breakdown and the training-free baselines from the episode CSVs "
             "(unseen tasks, 12 runs)", fontsize=9.5, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig9_pertask_handcrafted.png"), bbox_inches="tight")
plt.close(fig)
print("wrote fig9_pertask_handcrafted.png")


# ======================================================================
# Console summary
# ======================================================================
print("\n" + "=" * 88)
print("COVERAGE at nominal 0.80 on unseen tasks, and the AUROC that buys it")
print("=" * 88)
print(f"{'detector':<13s}{'splitCP':>9s}{'+taskrel':>10s}{'+ACI':>8s}{'+both':>8s}"
      f"{'AUROC abs':>11s}{'AUROC trel':>12s}")
print("-" * 88)
for d in DETS:
    f = lambda m, s: blocked[(blocked.detector == d + s) & (blocked.method == m)].coverage.mean()
    print(f"{d:<13s}{f(SPLIT, ''):>9.3f}{f(SPLIT, ' [task-rel]'):>10.3f}"
          f"{f(ACI, ''):>8.3f}{f(ACI, ' [task-rel]'):>8.3f}"
          f"{full[full.detector == d].auroc.mean():>11.3f}"
          f"{full[full.detector == d + ' [task-rel]'].auroc.mean():>12.3f}")

print("\nSAFE-Embed with a longer reference window (k=20):")
k20 = ks[(ks.detector == "SAFE-Embed") & (ks.k == 20)]
base_e = blocked[(blocked.detector == "SAFE-Embed") & (blocked.method == SPLIT)]
print(f"  AUROC    {full[full.detector=='SAFE-Embed'].auroc.mean():.3f} (absolute) "
      f"-> {k20.auroc.mean():.3f} (task-relative)")
print(f"  coverage {base_e.coverage.mean():.3f} (absolute) "
      f"-> {k20.coverage.mean():.3f} (task-relative), nominal 0.80")
print("=" * 88)
