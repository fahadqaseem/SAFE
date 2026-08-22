"""
Label-smearing ablation: does the Hide-and-Seek loss fix temporal smearing, or
does simply dropping SAFE's cumulative aggregation account for the difference?

SAFE's MLP probe and Hide-and-Seek differ in (at least) two places at once:

    LOSS         SAFE per-timestep hinge on the trajectory label
                 vs.  L_inter + lambda * L_intra
    AGGREGATION  cumsum over per-step sigmoid (IndepModelConfig.cumsum = True)
                 vs.  per-step score

A cumulative sum of strictly positive increments is monotone non-decreasing, so
the cumsum head cannot produce a flat-then-spike score under ANY loss. Comparing
"SAFE with cumsum" against "Hide-and-Seek without cumsum" therefore cannot tell
you which change mattered. This script crosses the two factors on one shared
architecture, on data where the failure onset is known exactly.

Testbed. Multitask synthetic rollouts in which the pre-onset segment of a
failure trajectory is drawn from EXACTLY the same distribution as a success
trajectory of the same task. That is the label-smearing setting in its pure
form: the trajectory-level label says "failure", but the early timesteps carry
no failure evidence, so any model that scores them high is smearing the label
backwards in time rather than detecting anything.

Metrics (both threshold-free and scale-free, so variants whose scores live on
different scales stay comparable):

    smear AUC  AUC separating PRE-onset failure timesteps from success
               timesteps, computed within relative-time bins and averaged, so
               that the monotone growth of a cumsum score is not itself scored
               as smearing. 0.50 = no smearing. Higher = worse.
    post AUC   Same, for POST-onset failure timesteps. Higher = better.
    onset err  (first threshold crossing - true onset) / length, signed.
               Negative = fires before anything has gone wrong.
    bal acc    Trajectory-level balanced accuracy, max-over-time score.

Run:  python3 experiments/smearing_experiment.py
"""

import importlib.util
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "experiments", "out")

# Fraction of the failure direction that is shared across tasks (vs task-specific).
SHARED_FAIL_FRAC = 0.75
os.makedirs(OUT, exist_ok=True)


def _load(name, relpath):
    """Load a module by file path, bypassing failure_prob's package __init__
    (which pulls in wandb/hydra). The source under test is unchanged."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


safe_utils = _load("safe_model_utils", "failure_prob/model/utils.py")
hns = _load("safe_hns_loss", "failure_prob/model/hns_loss.py")

get_time_weight = safe_utils.get_time_weight
aggregate_monitor_loss = safe_utils.aggregate_monitor_loss


# --------------------------------------------------------------------------
# Synthetic multitask rollouts with known failure onset
# --------------------------------------------------------------------------

def make_dataset(seed, n_tasks=10, n_per_task=48, dim=32, t_lo=60, t_hi=120):
    """Returns features (B,T,D), valid_masks (B,T), success (B,), onset (B,), task_id (B,).

    Success and failure trajectories of the same task are statistically identical
    up to the onset. `onset` is len(traj) for successes (never fires).
    """
    rng = np.random.default_rng(seed)

    # Per-task nominal mean.
    task_mean = rng.normal(0, 1.0, size=(n_tasks, dim))

    # Failure direction = a shared cross-task component plus a task-specific one.
    # The shared component is what makes zero-shot transfer to held-out tasks
    # possible at all; it is SAFE's own premise that failure signatures share
    # structure across tasks. With a purely task-specific direction, no probe
    # can generalise to an unseen task and every method scores ~0.5 by
    # construction, which measures nothing.
    shared_dir = rng.normal(0, 1.0, size=(1, dim))
    shared_dir /= np.linalg.norm(shared_dir)
    task_dir = rng.normal(0, 1.0, size=(n_tasks, dim))
    task_dir /= np.linalg.norm(task_dir, axis=1, keepdims=True)
    fail_dir = SHARED_FAIL_FRAC * shared_dir + (1 - SHARED_FAIL_FRAC) * task_dir
    fail_dir /= np.linalg.norm(fail_dir, axis=1, keepdims=True)

    # A shared low-rank "phase" basis so features drift smoothly with progress,
    # which is what makes time itself weakly predictive - as in real rollouts.
    phase_basis = rng.normal(0, 1.0, size=(3, dim))

    feats, masks, succ, onsets, tids = [], [], [], [], []
    T_max = t_hi

    for task in range(n_tasks):
        for i in range(n_per_task):
            L = int(rng.integers(t_lo, t_hi + 1))
            is_fail = bool(rng.random() < 0.5)

            prog = np.linspace(0, 1, L)[:, None]                       # (L,1)
            phase = (np.sin(2 * np.pi * prog) * phase_basis[0]
                     + prog * phase_basis[1]
                     + np.cos(np.pi * prog) * phase_basis[2]) * 0.6

            # Smooth per-trajectory drift (random walk, low-pass filtered).
            walk = rng.normal(0, 0.12, size=(L, dim)).cumsum(0)
            walk -= walk.mean(0, keepdims=True)

            x = task_mean[task][None, :] + phase + walk
            x += rng.normal(0, 0.35, size=(L, dim))                    # sensor noise

            if is_fail:
                t_on = int(rng.integers(int(0.40 * L), int(0.80 * L)))
                # Failure grows after onset: a ramp along the task's failure
                # direction, saturating. Pre-onset rows are untouched.
                ramp = np.clip((np.arange(L) - t_on) / (0.15 * L), 0, 1)[:, None]
                x = x + 2.2 * ramp * fail_dir[task][None, :]
            else:
                t_on = L

            padded = np.zeros((T_max, dim), dtype=np.float32)
            padded[:L] = x
            mask = np.zeros(T_max, dtype=np.float32)
            mask[:L] = 1.0

            feats.append(padded)
            masks.append(mask)
            succ.append(0 if is_fail else 1)
            onsets.append(t_on)
            tids.append(task)

    return (
        torch.from_numpy(np.stack(feats)),
        torch.from_numpy(np.stack(masks)),
        torch.tensor(succ, dtype=torch.long),
        torch.tensor(onsets, dtype=torch.long),
        torch.tensor(tids, dtype=torch.long),
    )


def split_by_task(task_id, n_tasks, unseen_ratio=0.3, train_ratio=0.6, seed=0):
    """SAFE's protocol: hold out whole tasks to measure zero-shot transfer."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_tasks)
    n_unseen = max(1, int(round(unseen_ratio * n_tasks)))
    unseen_tasks = set(perm[:n_unseen].tolist())

    is_unseen = np.isin(task_id.numpy(), list(unseen_tasks))
    seen_idx = np.where(~is_unseen)[0]
    rng.shuffle(seen_idx)
    n_tr = int(round(train_ratio * len(seen_idx)))

    return {
        "train": torch.tensor(seen_idx[:n_tr]),
        "val_seen": torch.tensor(seen_idx[n_tr:]),
        "val_unseen": torch.tensor(np.where(is_unseen)[0]),
    }


# --------------------------------------------------------------------------
# The probe: SAFE's IndepModel architecture, loss and aggregation as switches
# --------------------------------------------------------------------------

class Probe(nn.Module):
    def __init__(self, dim, hidden=256, n_layers=2, agg="cumsum"):
        super().__init__()
        layers = [nn.Linear(dim, hidden), nn.ReLU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)
        self.agg = agg

    def forward(self, x):
        s = self.net(x)                                                # (B,T,1)
        if self.agg == "cumsum":
            s = torch.cumsum(s, dim=-2)
        elif self.agg == "rmean":
            s = torch.cumsum(s, dim=-2)
            s = s / torch.arange(1, s.shape[1] + 1, device=s.device).view(1, -1, 1)
        elif self.agg != "none":
            raise ValueError(self.agg)
        return s.squeeze(-1)                                           # (B,T)


def safe_loss(scores, valid_masks, labels, weights, use_time_weighting):
    """SAFE's hinge, using SAFE's own get_time_weight / aggregate_monitor_loss."""
    time_weights = get_time_weight(use_time_weighting, valid_masks).to(scores)
    seq_loss_success = torch.relu(scores - 0)
    seq_loss_fail = time_weights * (-scores)          # use_threshold = False default
    losses = (labels == 1).float()[:, None] * seq_loss_success + \
             (labels == 0).float()[:, None] * seq_loss_fail
    monitor_loss, _, _ = aggregate_monitor_loss(losses, valid_masks, labels, weights)
    return monitor_loss


VARIANTS = [
    # label,                     agg,       loss,    time_weight
    ("SAFE (cumsum, published)", "cumsum", "safe",  False),
    ("SAFE + time weighting",    "cumsum", "safe",  True),
    ("SAFE loss, no cumsum",     "none",   "safe",  False),
    ("H&S inter only",           "none",   "inter", False),
    ("H&S intra only",           "none",   "intra", False),
    ("H&S inter+intra",          "none",   "hns",   False),
    ("H&S inter+intra, cumsum",  "cumsum", "hns",   False),
]


def standardize(feats, masks, train_idx):
    """Zero-mean unit-variance per feature dimension, mirroring SAFE's
    normalize_rollouts_hidden_states (cfg.dataset.normalize_hidden_states).
    Statistics come from the TRAIN split only, so there is no leakage into the
    held-out tasks."""
    m = masks[train_idx].bool()
    pool = feats[train_idx][m]                                     # (n_steps, D)
    mean, std = pool.mean(0), pool.std(0).clamp(min=1e-6)
    return (feats - mean) / std


def train_variant(agg, loss_type, time_weight, data, splits, seed,
                  n_epochs=300, lr=1e-3, batch_size=64,
                  margin_r=0.5, margin_o=0.3, lambda_intra=1.0):
    feats, masks, succ, onset, tid = data
    torch.manual_seed(seed)

    tr = splits["train"]
    feats = standardize(feats, masks, tr)
    model = Probe(feats.shape[-1], agg=agg)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # SAFE's class weighting, copied from RolloutDataset.__init__:
    #     freq_c = (count_c + 1) / N ;  weight_c = 1 / freq_c
    n_fail = int((succ[tr] == 0).sum())
    n_succ = int((succ[tr] == 1).sum())
    weights = [len(tr) / (n_fail + 1), len(tr) / (n_succ + 1)]

    g = torch.Generator().manual_seed(seed)
    for _ in range(n_epochs):
        perm = tr[torch.randperm(len(tr), generator=g)]
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 4:
                continue
            f, m, y = feats[idx], masks[idx], succ[idx]
            s = model(f)

            if loss_type == "safe":
                loss = safe_loss(s, m, y, weights, time_weight)
            else:
                fail_mask = (y == 0)
                loss, _ = hns.hns_loss(
                    s, m, fail_mask,
                    margin_r=margin_r, margin_o=margin_o, lambda_intra=lambda_intra,
                    use_inter=loss_type in ("inter", "hns"),
                    use_intra=loss_type in ("intra", "hns"),
                )

            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        scores = model(feats)
    return model, scores


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def _auc(pos, neg):
    """Rank-based AUC. pos should score higher than neg."""
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv), dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    r_pos = ranks[:len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def time_binned_auc(scores, masks, succ, onset, idx, segment, n_bins=10):
    """AUC of failure-trajectory timesteps in `segment` vs success timesteps,
    computed within relative-time bins then averaged (count-weighted).

    Binning on relative time is what makes this fair to the cumsum head: a
    monotonically growing score is not credited or penalised for growing.
    """
    s = scores.numpy()
    m = masks.numpy()
    lens = m.sum(1).astype(int)
    y = succ.numpy()
    on = onset.numpy()
    idx = idx.numpy()

    pos_bins = [[] for _ in range(n_bins)]
    neg_bins = [[] for _ in range(n_bins)]

    for i in idx:
        L = lens[i]
        if L < 2:
            continue
        rel = np.arange(L) / L
        b = np.minimum((rel * n_bins).astype(int), n_bins - 1)
        if y[i] == 1:
            for t in range(L):
                neg_bins[b[t]].append(s[i, t])
        else:
            lo, hi = (0, on[i]) if segment == "pre" else (on[i], L)
            for t in range(lo, min(hi, L)):
                pos_bins[b[t]].append(s[i, t])

    aucs, wts = [], []
    for k in range(n_bins):
        p, n = np.asarray(pos_bins[k]), np.asarray(neg_bins[k])
        if len(p) == 0 or len(n) == 0:
            continue
        a = _auc(p, n)
        if not np.isnan(a):
            aucs.append(a)
            wts.append(len(p))
    if not aucs:
        return np.nan
    return float(np.average(aucs, weights=wts))


def traj_metrics(scores, masks, succ, onset, calib_idx, test_idx, target_fpr=0.2):
    """Trajectory-level balanced accuracy + signed onset error.

    The threshold is chosen on the calibration split (seen tasks) at a target
    false-alarm rate on successes - the same spirit as SAFE's conformal band,
    without pulling in the conformal machinery.
    """
    s = scores.numpy()
    m = masks.numpy()
    lens = m.sum(1).astype(int)
    y = succ.numpy()
    on = onset.numpy()

    def peaks(idx):
        return np.array([s[i, :lens[i]].max() for i in idx])

    ci, ti = calib_idx.numpy(), test_idx.numpy()
    calib_succ_peaks = peaks(ci[y[ci] == 1])
    if len(calib_succ_peaks) == 0:
        return dict(bal_acc=np.nan, onset_err=np.nan, early_fire_rate=np.nan)
    thresh = float(np.quantile(calib_succ_peaks, 1 - target_fpr))

    tp = fn = tn = fp = 0
    onset_errs, early_fires = [], []
    for i in ti:
        L = lens[i]
        crossing = np.where(s[i, :L] >= thresh)[0]
        fired = len(crossing) > 0
        if y[i] == 0:
            if fired:
                tp += 1
                t_det = crossing[0]
                onset_errs.append((t_det - on[i]) / L)
                early_fires.append(1.0 if t_det < on[i] else 0.0)
            else:
                fn += 1
                early_fires.append(0.0)
        else:
            fp += 1 if fired else 0
            tn += 1 if not fired else 0

    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    return dict(
        bal_acc=0.5 * (tpr + tnr),
        onset_err=float(np.mean(onset_errs)) if onset_errs else np.nan,
        early_fire_rate=float(np.mean(early_fires)) if early_fires else np.nan,
        threshold=thresh,
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-tasks", type=int, default=10)
    ap.add_argument("--tag", type=str, default="")
    a = ap.parse_args()

    N_TASKS, SEEDS = a.n_tasks, a.seeds
    rows = []
    curves = {}     # for the figures, from seed 0

    for seed in SEEDS:
        data = make_dataset(seed, n_tasks=N_TASKS)
        feats, masks, succ, onset, tid = data
        splits = split_by_task(tid, N_TASKS, seed=seed)

        n_f = int((succ[splits['train']] == 0).sum())
        print(f"\n=== seed {seed} | train {len(splits['train'])} "
              f"(fail {n_f}) | val_seen {len(splits['val_seen'])} "
              f"| val_unseen {len(splits['val_unseen'])} ===", flush=True)

        for label, agg, loss_type, tw in VARIANTS:
            _, scores = train_variant(agg, loss_type, tw, data, splits, seed,
                                      n_epochs=a.epochs)

            for split in ["val_seen", "val_unseen"]:
                idx = splits[split]
                smear = time_binned_auc(scores, masks, succ, onset, idx, "pre")
                post = time_binned_auc(scores, masks, succ, onset, idx, "post")
                tm = traj_metrics(scores, masks, succ, onset, splits["val_seen"], idx)
                rows.append(dict(seed=seed, variant=label, agg=agg, loss=loss_type,
                                 time_weight=tw, split=split,
                                 smear_auc=smear, post_auc=post, **tm))
                if split == "val_unseen":
                    print(f"  {label:<28s} smearAUC {smear:.3f}  postAUC {post:.3f}  "
                          f"balAcc {tm['bal_acc']:.3f}  onsetErr {tm['onset_err']:+.3f}",
                          flush=True)

            if seed == 0:
                curves[label] = scores.numpy()

        if seed == 0:
            np.savez_compressed(
                os.path.join(OUT, f"curves_seed0{a.tag}.npz"),
                masks=masks.numpy(), succ=succ.numpy(), onset=onset.numpy(),
                val_unseen=splits["val_unseen"].numpy(),
                val_seen=splits["val_seen"].numpy(),
                **{f"scores::{k}": v for k, v in curves.items()},
            )

    with open(os.path.join(OUT, f"results{a.tag}.json"), "w") as f:
        json.dump(rows, f, indent=1, default=float)

    # Aggregate over seeds
    import pandas as pd
    df = pd.DataFrame(rows)
    agg_df = (df.groupby(["variant", "agg", "loss", "split"])
                [["smear_auc", "post_auc", "bal_acc", "onset_err", "early_fire_rate"]]
                .agg(["mean", "std"]))
    agg_df.to_csv(os.path.join(OUT, f"summary{a.tag}.csv"))

    order = [v[0] for v in VARIANTS]
    print("\n\n" + "=" * 104)
    print("VAL_UNSEEN (zero-shot on held-out tasks), mean +/- std over 3 seeds")
    print("=" * 104)
    print(f"{'variant':<28s} {'smearAUC':>16s} {'postAUC':>16s} {'balAcc':>16s} {'onsetErr':>16s}")
    print("-" * 104)
    sub = df[df.split == "val_unseen"]
    for v in order:
        r = sub[sub.variant == v]
        def f(c, sign=False):
            fmt = "+.3f" if sign else ".3f"
            return f"{r[c].mean():{fmt}}+-{r[c].std():.3f}"
        print(f"{v:<28s} {f('smear_auc'):>16s} {f('post_auc'):>16s} "
              f"{f('bal_acc'):>16s} {f('onset_err', True):>16s}")
    print("=" * 104)
    print("smearAUC: 0.50 = no label smearing, higher = worse.  onsetErr < 0 = fires before onset.")
    print(f"\nwrote {OUT}/results.json, summary.csv, curves_seed0.npz")


if __name__ == "__main__":
    main()
