"""
generate_figures.py
===================
Generates two publication-quality figures for the SAFE comparative experiment.

Prerequisites (produced by the training scripts):
    baseline_val_scores.pkl
    modified_val_scores.pkl
    baseline_features.pkl      (or modified_features.pkl for embeddings)

Outputs (workspace root):
    figure_trajectory_comparison.png   – failure score over time for one failed rollout
    figure_latent_space_pca.png        – 2-D PCA of mean embeddings (success vs. failure)
"""

import pickle, sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA

# ── Aesthetic settings ───────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize":10,
    "figure.dpi":     150,
})

ROOT = os.path.dirname(os.path.abspath(__file__))


# ════════════════════════════════════════════════════════════════════════════
# Helper: pick the best failure trajectory to display
#   – longest trajectory (most visual detail)
#   – from the val_seen or val_unseen split
# ════════════════════════════════════════════════════════════════════════════

def pick_failure_trajectory(scores_dict: dict, labels_dict: dict) -> int:
    """
    Returns the index of the failure trajectory with the highest score range
    (max - min) across any available validation split.
    """
    best_idx   = None
    best_range = -1.0

    for split in ("val_seen", "val_unseen", "train"):
        if split not in scores_dict:
            continue
        scores_list, labels_list = scores_dict[split]
        for i, (s, lbl) in enumerate(zip(scores_list, labels_list)):
            if int(lbl) == 0:                 # failure
                rng = float(np.max(s) - np.min(s))
                if rng > best_range:
                    best_range = rng
                    best_idx   = (split, i)

    return best_idx


# ════════════════════════════════════════════════════════════════════════════
# Figure 1: Trajectory line comparison
# ════════════════════════════════════════════════════════════════════════════

def plot_trajectory_comparison(
    baseline_scores: dict,
    modified_scores: dict,
    save_path: str,
):
    # Find a good failure trajectory present in *both* pkl files
    # We pick from the same split & same index for a fair comparison.
    candidate = pick_failure_trajectory(baseline_scores, {})
    if candidate is None:
        raise RuntimeError("No failure trajectory found in baseline val scores.")

    split, idx = candidate
    if split not in modified_scores:
        # Fall back: pick any split present in both
        for s in ("val_seen", "val_unseen", "train"):
            if s in baseline_scores and s in modified_scores:
                split = s
                # find a failure in this split
                s_list, l_list = baseline_scores[split]
                for j, lbl in enumerate(l_list):
                    if int(lbl) == 0:
                        idx = j
                        break
                break

    base_s = baseline_scores[split][0][idx]
    mod_s  = modified_scores[split][0][idx]

    # Align lengths (pad/truncate to common length)
    L = min(len(base_s), len(mod_s))
    base_s = base_s[:L]
    mod_s  = mod_s[:L]
    t = np.arange(L)

    fig, ax = plt.subplots(figsize=(7, 3.5))

    ax.plot(t, base_s, color="#2166ac", linewidth=2, linestyle="--",
            label="Baseline (time-weighted BCE)", alpha=0.9)
    ax.plot(t, mod_s,  color="#d6604d", linewidth=2, linestyle="-",
            label="Modified (intra-traj. contrastive)", alpha=0.9)

    ax.set_xlabel("Timestep $t$")
    ax.set_ylabel("Failure score")
    ax.set_title("Failure Score Over Time — Single Failed Trajectory")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper left", framealpha=0.85)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    # Annotate the onset spike for the modified model
    onset_idx = int(np.argmax(np.diff(mod_s))) + 1
    ax.axvline(onset_idx, color="#d6604d", linewidth=1, linestyle=":",
               alpha=0.6, label="_nolegend_")
    ax.annotate(
        "onset↑",
        xy=(onset_idx, mod_s[onset_idx]),
        xytext=(onset_idx + max(L // 10, 2), mod_s[onset_idx] + 0.08),
        arrowprops=dict(arrowstyle="->", color="#d6604d", lw=1.2),
        color="#d6604d",
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 2: PCA scatter of latent embeddings
# ════════════════════════════════════════════════════════════════════════════

def plot_latent_pca(
    features_pkl: str,
    save_path: str,
):
    with open(features_pkl, "rb") as f:
        feat_dict = pickle.load(f)

    # Collect embeddings from all validation splits (exclude train for cleaner viz)
    embeddings, labels = [], []
    for split in ("val_seen", "val_unseen", "train"):
        if split not in feat_dict:
            continue
        feats_list, lbls = feat_dict[split]
        for emb, lbl in zip(feats_list, lbls):
            embeddings.append(emb)
            labels.append(int(lbl))

    embeddings = np.stack(embeddings, axis=0)    # (N, D)
    labels     = np.array(labels)                # (N,)

    # L2-normalize embeddings to unit sphere before PCA
    # (prevents overflow in sklearn's randomized SVD with 4096-D vectors)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-8)

    # Also standardize to zero mean (per-feature)
    embeddings = embeddings - embeddings.mean(axis=0)

    # PCA → 2-D  (use full SVD with float64 to avoid overflow in randomized SVD
    #              when embedding dim is very large, e.g. 4096)
    pca = PCA(n_components=2, random_state=0, svd_solver="full")
    coords = pca.fit_transform(embeddings.astype(np.float64))    # (N, 2)

    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100

    success_idx = labels == 1
    fail_idx    = labels == 0

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    ax.scatter(
        coords[success_idx, 0], coords[success_idx, 1],
        c="#4dac26", alpha=0.55, s=20, label=f"Success (n={success_idx.sum()})",
        edgecolors="none",
    )
    ax.scatter(
        coords[fail_idx, 0], coords[fail_idx, 1],
        c="#d01c8b", alpha=0.55, s=20, label=f"Failure (n={fail_idx.sum()})",
        edgecolors="none",
        marker="^",
    )

    ax.set_xlabel(f"PC 1 ({var1:.1f}% var.)")
    ax.set_ylabel(f"PC 2 ({var2:.1f}% var.)")
    ax.set_title("Latent Embedding Space — Success vs. Failure\n(PCA of mean hidden-state embeddings)")
    ax.legend(loc="best", framealpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(linestyle=":", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    baseline_scores_path  = os.path.join(ROOT, "baseline_val_scores.pkl")
    modified_scores_path  = os.path.join(ROOT, "modified_val_scores.pkl")
    baseline_features_path= os.path.join(ROOT, "baseline_features.pkl")

    for path in [baseline_scores_path, modified_scores_path, baseline_features_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                "Please run train_baseline.py and train_modified.py first."
            )

    with open(baseline_scores_path,  "rb") as f:
        baseline_scores = pickle.load(f)
    with open(modified_scores_path,  "rb") as f:
        modified_scores = pickle.load(f)

    print("=== Figure 1: Trajectory comparison ===")
    plot_trajectory_comparison(
        baseline_scores,
        modified_scores,
        save_path=os.path.join(ROOT, "figure_trajectory_comparison.png"),
    )

    print("\n=== Figure 2: Latent space PCA ===")
    plot_latent_pca(
        features_pkl=baseline_features_path,
        save_path=os.path.join(ROOT, "figure_latent_space_pca.png"),
    )

    print("\nAll figures saved.")


if __name__ == "__main__":
    main()
