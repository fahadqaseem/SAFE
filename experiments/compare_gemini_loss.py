"""
Apples-to-apples test of the intra-trajectory contrastive loss from
`.history/train_modified_*.py` against a faithful SAFE-LSTM baseline.

Why this script exists
----------------------
The original comparison (Gemini's walkthrough.md) reported val_unseen AUROC
0.686 (baseline) vs 0.824 (modified) from a SINGLE seed and a SINGLE split, and
scored balanced accuracy at a fixed threshold of 0.5. Two problems:

  * one split cannot separate a real effect from split noise -- our own faithful
    SAFE reproduction varies by +-0.05 AUROC across 12 runs;
  * a fixed 0.5 threshold is not how SAFE decides anything. SAFE thresholds via
    conformal calibration. A BCE-trained score sits mostly above 0.5 for BOTH
    classes, so balanced accuracy at 0.5 collapses toward 0.5 by construction --
    that is a property of the threshold, not of the loss.

What the original got RIGHT, and we reproduce faithfully: SAFE's `lstm.py` really
does use time-weighted BCE with failure as the positive class when
`cumsum == False`, and `LstmModelConfig.cumsum` really is `False` by default
(`failure_prob/model/lstm.py:99-107`, `conf/__init__.py:226`). So the baseline
loss family was correct. The one deviation is `use_time_weighting=True`, where
SAFE's default is `False` and no released script enables it -- and since that
weighting puts up to 6x weight on the EARLIEST timesteps of failure rollouts, it
plausibly handicaps the baseline. We therefore run it both ways.

Arms (identical LSTM, identical splits, identical features):
  A  SAFE-LSTM faithful        time-weighted BCE, use_time_weighting=False
  B  SAFE-LSTM as-run          same, use_time_weighting=True
  C  intra-contrastive         Gemini's loss: intra margin + success suppression
  D  Hide-and-Seek proper      inter + intra (arXiv:2605.30834)

Non-circular check. The headline "post/pre ratio 33x" is measured at the split
point the modified loss is explicitly trained to maximise, on a dataset with NO
failure-onset annotations -- so it restates the training objective rather than
testing it. We report that ratio for every arm (so it is comparable) AND
early-detection AUROC at 50% of the episode, which no arm is trained on. A score
that genuinely stays flat then spikes late should look WORSE early and better at
the end; that is a falsifiable prediction.

Run:  python3 experiments/compare_gemini_loss.py
"""

import argparse
import importlib.util
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "experiments", "out")
sys.path.insert(0, os.path.join(REPO, "experiments"))

from run_real_widowx import make_fold  # noqa: E402


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


safe_utils = _load("safe_model_utils2", "failure_prob/model/utils.py")
hnsmod = _load("safe_hns_loss2", "failure_prob/model/hns_loss.py")
get_time_weight = safe_utils.get_time_weight
aggregate_monitor_loss = safe_utils.aggregate_monitor_loss

DEV = "mps" if torch.backends.mps.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Model: SAFE's LstmModel with n_history_steps=-1, cumsum=False
# ---------------------------------------------------------------------------

class LstmProbe(nn.Module):
    def __init__(self, input_dim, hidden=256, layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers,
                            batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return torch.sigmoid(self.fc(self.drop(out))).squeeze(-1)   # (B, T)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def safe_lstm_bce(scores, masks, labels, weights, use_tw):
    """Verbatim structure of lstm.py's cumsum==False branch."""
    tw = get_time_weight(use_tw, masks).to(scores)
    target = (1 - labels).unsqueeze(-1).expand_as(scores).to(scores)
    losses = nn.BCELoss(reduction="none")(scores.clamp(1e-6, 1 - 1e-6), target)
    fail = (labels == 0)
    if fail.any():
        losses = losses.clone()
        losses[fail] = losses[fail] * tw[fail]
    loss, _, _ = aggregate_monitor_loss(losses, masks, labels, weights)
    return loss


def intra_contrastive(scores, masks, labels, margin=0.5):
    """Gemini's loss: intra-trajectory margin on failures + suppression on successes.

    Note this is NOT the degenerate pure-L_intra objective: the
    relu(scores).mean() term on successes anchors the success class, which is the
    role the inter-trajectory term plays in the published Hide-and-Seek loss.
    """
    terms = []
    B, T = scores.shape
    for i in range(B):
        L = int(masks[i].sum())
        if L < 3:
            continue
        s = scores[i, :L]
        if labels[i] == 0:
            k = int((s[1:] - s[:-1]).detach().argmax().item()) + 1
            k = max(1, min(k, L - 1))
            terms.append(torch.relu(margin - (s[k:].mean() - s[:k].mean())))
        else:
            terms.append(torch.relu(s).mean())
    if not terms:
        return scores.sum() * 0.0
    return torch.stack(terms).mean()


ARMS = {
    "A SAFE-LSTM faithful": dict(kind="bce", use_tw=False),
    "B SAFE-LSTM as-run":   dict(kind="bce", use_tw=True),
    "C intra-contrastive":  dict(kind="intra"),
    "D H&S inter+intra":    dict(kind="hns"),
}


def train_arm(feats, succ, tr_idx, cfg, seed, n_epochs, lr=3e-4, bs=16,
              grad_clip=1.0):
    torch.manual_seed(seed)
    model = LstmProbe(feats.shape[-1]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    n_f = int((succ[tr_idx] == 0).sum()); n_s = int((succ[tr_idx] == 1).sum())
    weights = [len(tr_idx) / (n_f + 1), len(tr_idx) / (n_s + 1)]

    g = torch.Generator().manual_seed(seed)
    for _ in range(n_epochs):
        perm = tr_idx[torch.randperm(len(tr_idx), generator=g)]
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            if len(idx) < 4:
                continue
            x = feats[idx].to(DEV)
            y = succ[idx].to(DEV)
            m = torch.ones(len(idx), x.shape[1], device=DEV)
            s = model(x)

            if cfg["kind"] == "bce":
                loss = safe_lstm_bce(s, m, y, weights, cfg["use_tw"])
            elif cfg["kind"] == "intra":
                loss = intra_contrastive(s, m, y)
            else:
                loss, _ = hnsmod.hns_loss(s, m, (y == 0), margin_r=0.5,
                                          margin_o=0.3, lambda_intra=1.0)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(feats), 64):
            outs.append(model(feats[i:i + 64].to(DEV)).cpu().numpy())
    return np.concatenate(outs)                                    # (N, T)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    o = s.argsort(); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    _, inv, c = np.unique(s, return_inverse=True, return_counts=True)
    sm = np.zeros(len(c)); np.add.at(sm, inv, r); r = (sm / c)[inv]
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def avg_precision(s, y):
    s = np.asarray(s, float); y = np.asarray(y)
    o = np.argsort(-s); y = y[o]
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / max(y.sum(), 1))


def bal_acc(peaks, y, thr):
    pred = peaks >= thr
    tpr = pred[y == 1].mean() if (y == 1).any() else np.nan
    tnr = (~pred[y == 0]).mean() if (y == 0).any() else np.nan
    return float(0.5 * (tpr + tnr))


def post_pre_ratio(curves):
    """Ratio of post-onset to pre-onset mean, at the model's OWN argmax-of-diff
    split. Circular for arm C by construction; reported for all arms so the
    numbers are at least comparable."""
    out = []
    for s in curves:
        L = len(s)
        k = int(np.argmax(np.diff(s))) + 1
        k = max(1, min(k, L - 1))
        pre, post = s[:k].mean(), s[k:].mean()
        out.append(post / max(pre, 1e-6))
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--token", default="last")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    blob = torch.load(os.path.join(OUT, f"widowx_{a.token}.pt"), weights_only=False)
    feats_raw, succ, task_id = blob["features"], blob["success"], blob["task_id"]
    tasks = sorted(task_id.unique().tolist())
    print(f"device={DEV}  features={tuple(feats_raw.shape)}  token={a.token}")

    combos = list(itertools.combinations(tasks, 3))
    pick = np.random.default_rng(0).choice(len(combos), size=a.folds, replace=False)
    holdouts = [set(combos[i]) for i in sorted(pick)]

    rows = []
    t0 = time.time()
    for fi, held in enumerate(holdouts):
        for seed in range(a.seeds):
            sp = make_fold(task_id, succ, held, seed=1000 * fi + seed)
            tr, vs, vu = sp["train"], sp["val_seen"], sp["val_unseen"]

            pool = feats_raw[tr].reshape(-1, feats_raw.shape[-1])
            mu, sd = pool.mean(0), pool.std(0).clamp(min=1e-6)
            feats = (feats_raw - mu) / sd

            for arm, cfg in ARMS.items():
                cur = train_arm(feats, succ, tr, cfg, seed, a.epochs)
                for split, idx in [("val_seen", vs), ("val_unseen", vu)]:
                    ix = idx.numpy()
                    y = (succ[idx].numpy() == 0).astype(int)   # 1 = failure
                    peaks = cur[ix].max(1)

                    # calibrated threshold: 80th pct of val_seen SUCCESS peaks
                    cal = cur[vs.numpy()][succ[vs].numpy() == 1].max(1)
                    thr_cal = float(np.quantile(cal, 0.8)) if len(cal) else 0.5

                    half = cur.shape[1] // 2
                    rows.append(dict(
                        fold=fi, seed=seed, arm=arm, split=split,
                        auroc=auroc(peaks, y),
                        ap=avg_precision(peaks, y),
                        bal_acc_05=bal_acc(peaks, y, 0.5),
                        bal_acc_cal=bal_acc(peaks, y, thr_cal),
                        auroc_half=auroc(cur[ix][:, :half].max(1), y),
                        post_pre=post_pre_ratio(cur[ix][y == 1]),
                        mean_peak_fail=float(peaks[y == 1].mean()),
                        mean_peak_succ=float(peaks[y == 0].mean()),
                    ))
            print(f"  fold {fi} seed {seed} done ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"gemini_compare{a.tag}.csv"), index=False)

    n = a.folds * a.seeds
    for split in ["val_unseen", "val_seen"]:
        d = df[df.split == split]
        print("\n" + "=" * 108)
        print(f"{split.upper()}  —  mean ± sd over {n} runs (6 three-task holdouts x 2 seeds)")
        print("=" * 108)
        print(f"{'arm':<24s}{'AUROC':>14s}{'AP':>14s}{'balAcc@0.5':>14s}"
              f"{'balAcc@cal':>14s}{'AUROC@50%':>13s}{'post/pre':>11s}")
        print("-" * 108)
        for arm in ARMS:
            g = d[d.arm == arm]
            f = lambda c: f"{g[c].mean():.3f}±{g[c].std():.3f}"
            print(f"{arm:<24s}{f('auroc'):>14s}{f('ap'):>14s}{f('bal_acc_05'):>14s}"
                  f"{f('bal_acc_cal'):>14s}{f('auroc_half'):>13s}"
                  f"{g['post_pre'].mean():>11.1f}")
    print("=" * 108)

    u = df[df.split == "val_unseen"]
    A = u[u.arm == "A SAFE-LSTM faithful"].auroc
    C = u[u.arm == "C intra-contrastive"].auroc
    diff = C.mean() - A.mean()
    pooled = np.sqrt(A.var(ddof=1) / len(A) + C.var(ddof=1) / len(C))
    print(f"\nVERDICT  intra-contrastive minus faithful SAFE, unseen-task AUROC:")
    print(f"  delta = {diff:+.3f}   se = {pooled:.3f}   ratio = {diff / max(pooled, 1e-9):+.2f} se")
    print(f"  {'DISTINGUISHABLE' if abs(diff) > 2 * pooled else 'NOT distinguishable'} "
          f"from the baseline at ~2 se")
    print(f"\nwrote gemini_compare{a.tag}.csv  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
