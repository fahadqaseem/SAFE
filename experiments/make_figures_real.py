"""Figures for the real-WidowX online-CP experiment. Run run_real_widowx.py first."""

import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "experiments", "out")

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130,
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "legend.frameon": False,
})

C = {
    "split CP (SAFE)": "#B4413C",
    "split CP (n+1 corr.)": "#D98C7A",
    "ACI g=0.01": "#E0A030",
    "ACI g=0.05": "#2E6E8E",
    "ACI g=0.05 (grow cal)": "#1F4E5F",
    "DtACI": "#5B4E8C",
    "DtACI (grow cal)": "#8A7FB5",
}
DETS = ["SAFE-MLP", "SAFE-Embed", "H&S inter+intra"]
ALPHA_MAIN = 0.20

df = pd.read_csv(os.path.join(OUT, "real_cp_results.csv"))
curves = pickle.load(open(os.path.join(OUT, "real_cp_curves.pkl"), "rb"))


def rolling(err, window=30):
    idx = np.where(~np.isnan(err))[0]
    e = err[idx]
    y = [1 - e[max(0, j - window + 1):j + 1].mean() for j in range(len(e))]
    return idx, np.asarray(y)


# ======================================================================
# FIG 1 - the headline: coverage under task shift, and who fixes it
# ======================================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4.0), sharey=True)
shown = ["split CP (SAFE)", "ACI g=0.05", "ACI g=0.05 (grow cal)", "DtACI (grow cal)"]

for ax, det in zip(axes, DETS):
    meta = curves.get("__meta__|blocked")
    for m in shown:
        k = f"{det}|blocked|{m}"
        if k not in curves:
            continue
        x, y = rolling(curves[k]["err"])
        ax.plot(x, y, lw=1.9, color=C.get(m, "grey"), label=m)

    if meta is not None:
        for b in meta["bounds"]:
            ax.axvline(b, color="k", ls=":", lw=1.0, alpha=0.6)
    ax.axhline(1 - ALPHA_MAIN, color="green", ls="--", lw=1.4)
    ax.set_title(det)
    ax.set_xlabel("episode index in deployment stream")
    ax.set_ylim(-0.03, 1.03)

axes[0].set_ylabel(f"rolling coverage (window 30)")
axes[0].text(2, 1 - ALPHA_MAIN + 0.03, f"nominal {1 - ALPHA_MAIN:.2f}",
             color="green", fontsize=8)
axes[2].legend(fontsize=7.5, loc="lower right")
fig.suptitle("Rolling coverage along the task-blocked deployment stream on UNSEEN tasks "
             f"(alpha={ALPHA_MAIN}, fold 0). Dotted lines = task boundaries.",
             fontsize=9.5, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_rolling_coverage.png"), bbox_inches="tight")
plt.close(fig)
print("wrote fig4_rolling_coverage.png")


# ======================================================================
# FIG 2 - calibration diagram: nominal vs realised
# ======================================================================
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
alphas = sorted(df.alpha.unique())
nom = [1 - a for a in alphas]

for ax, det in zip(axes, DETS):
    ax.plot([0.7, 0.95], [0.7, 0.95], color="k", ls="--", lw=1.2, label="ideal")

    series = [
        ("exch",    "split CP (SAFE)",        "#7FA8C9", "o", "split CP, exchangeable (control)"),
        ("blocked", "split CP (SAFE)",        "#B4413C", "s", "split CP, unseen tasks"),
        ("blocked", "ACI g=0.05",             "#E0A030", "^", "ACI, unseen tasks"),
        ("blocked", "ACI g=0.05 (grow cal)",  "#2E6E8E", "D", "ACI + growing cal, unseen"),
        ("blocked", "DtACI (grow cal)",       "#5B4E8C", "v", "DtACI + growing cal, unseen"),
    ]
    for order, m, col, mk, lab in series:
        sub = df[(df.detector == det) & (df.order == order) & (df.method == m)]
        if sub.empty:
            continue
        mu = [sub[sub.alpha == a].coverage.mean() for a in alphas]
        sd = [sub[sub.alpha == a].coverage.std() for a in alphas]
        ax.errorbar(nom, mu, yerr=sd, marker=mk, color=col, lw=1.6, ms=5,
                    capsize=2.5, label=lab)

    ax.set_title(det)
    ax.set_xlabel("nominal coverage  1 - alpha")
    ax.set_xlim(0.71, 0.94)

axes[0].set_ylabel("realised coverage on the stream")
axes[0].set_ylim(-0.05, 1.05)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, fontsize=8, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.09))
fig.suptitle("Nominal vs realised coverage, mean ± sd over 12 runs. On the dashed line = valid.",
             fontsize=9.5, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig5_calibration.png"), bbox_inches="tight")
plt.close(fig)
print("wrote fig5_calibration.png")


# ======================================================================
# FIG 3 - validity vs usefulness, and the alpha_t trace
# ======================================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))

# (a) alpha_t traces
ax = axes[0]
for m in ["ACI g=0.01", "ACI g=0.05", "ACI g=0.05 (grow cal)", "DtACI (grow cal)"]:
    k = f"SAFE-MLP|blocked|{m}"
    if k in curves:
        ax.plot(curves[k]["alpha_t"], lw=1.8, color=C.get(m, "grey"), label=m)
ax.axhline(ALPHA_MAIN, color="green", ls="--", lw=1.3)
meta = curves.get("__meta__|blocked")
if meta is not None:
    for b in meta["bounds"]:
        ax.axvline(b, color="k", ls=":", lw=1.0, alpha=0.6)
ax.set_xlabel("episode index"); ax.set_ylabel(r"adaptive level $\alpha_t$")
ax.set_title("SAFE-MLP: how the level adapts")
ax.legend(fontsize=7.5)

# (b) coverage gap vs (c) detection power, unseen-task blocked stream
main = df[(df.order == "blocked") & (df.alpha == ALPHA_MAIN)]
methods = ["split CP (SAFE)", "ACI g=0.05", "ACI g=0.05 (grow cal)", "DtACI (grow cal)"]

for ax, metric, label, ref in [
    (axes[1], "cov_gap", "coverage gap  (realised - nominal)", 0.0),
    (axes[2], "bal_acc", "balanced accuracy on the stream", 0.5),
]:
    w = 0.8 / len(methods)
    x = np.arange(len(DETS))
    for j, m in enumerate(methods):
        mu, sd = [], []
        for d in DETS:
            s = main[(main.detector == d) & (main.method == m)][metric]
            mu.append(s.mean()); sd.append(s.std())
        ax.bar(x + (j - (len(methods) - 1) / 2) * w, mu, w, yerr=sd, capsize=2,
               color=C.get(m, "grey"), alpha=0.92,
               label=m if ax is axes[1] else None)
    ax.axhline(ref, color="k", ls="--", lw=1.1)
    ax.set_xticks(x); ax.set_xticklabels(DETS, fontsize=8)
    ax.set_ylabel(label)

axes[1].set_title(f"Validity (alpha={ALPHA_MAIN}, unseen tasks)")
axes[1].legend(fontsize=7)
axes[2].set_title("Usefulness: is validity bought by never firing?")

fig.suptitle("Adaptation, validity, and whether the detector is still useful "
             "(mean ± sd over 12 runs)", fontsize=9.5, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig6_adaptation_power.png"), bbox_inches="tight")
plt.close(fig)
print("wrote fig6_adaptation_power.png")


# ======================================================================
# Tables
# ======================================================================
def table(order, alpha):
    sub = df[(df.order == order) & (df.alpha == alpha)]
    rows = []
    for d in DETS:
        for m in ["split CP (SAFE)", "split CP (n+1 corr.)", "ACI g=0.005",
                  "ACI g=0.01", "ACI g=0.05", "ACI g=0.05 (grow cal)",
                  "DtACI", "DtACI (grow cal)"]:
            s = sub[(sub.detector == d) & (sub.method == m)]
            if s.empty:
                continue
            rows.append(dict(detector=d, method=m,
                             coverage=s.coverage.mean(), coverage_sd=s.coverage.std(),
                             gap=s.cov_gap.mean(), tpr=s.tpr.mean(),
                             bal_acc=s.bal_acc.mean(), det_time=s.det_time.mean()))
    return pd.DataFrame(rows)

for order in ["exch", "blocked"]:
    t = table(order, ALPHA_MAIN)
    t.to_csv(os.path.join(OUT, f"real_cp_table_{order}.csv"), index=False)
    print(f"\n{'=' * 104}\n{order.upper()} stream, alpha={ALPHA_MAIN} "
          f"(nominal coverage {1 - ALPHA_MAIN:.2f}), mean over 12 runs\n{'=' * 104}")
    print(f"{'detector':<17s}{'method':<24s}{'coverage':>16s}{'gap':>8s}"
          f"{'TPR':>7s}{'balAcc':>8s}{'detTime':>9s}")
    print("-" * 104)
    for _, r in t.iterrows():
        print(f"{r.detector:<17s}{r.method:<24s}"
              f"{r.coverage:>9.3f}+-{r.coverage_sd:<5.3f}{r.gap:>+8.3f}"
              f"{r.tpr:>7.3f}{r.bal_acc:>8.3f}{r.det_time:>9.3f}")

print("\nAUROC on unseen tasks (order-independent), mean +- sd over 12 runs:")
for d in DETS:
    s = df[df.detector == d].groupby(["fold", "seed"]).auroc.first()
    print(f"  {d:<18s} {s.mean():.3f} +- {s.std():.3f}")
