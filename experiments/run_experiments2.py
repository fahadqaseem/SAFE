"""
Round 2 on the real WidowX rollouts. Four things, one pass:

1. TASK-RELATIVE SCORING (the proposed improvement). Score each episode against
   its own early trend instead of an absolute scale, cancelling the measured
   task-dependent offset (successful unseen-task episodes peak at 15.29 vs 5.43
   on seen tasks). Applied as a post-hoc transform of the score curves, so every
   base/task-relative pair shares identical weights -- the comparison isolates
   the transform and nothing else.

2. EARLY DETECTION. AUROC after seeing only the first f of each episode, for
   f = 0.1 ... 1.0. Answers whether there is time to stop the arm.

3. HANDCRAFTED BASELINES. The per-episode CSVs carry 47 training-free signals
   (token entropy/probability, action deltas) -- SAFE's own "handcrafted metrics"
   rows. Zero training. Sign of each metric is chosen on the TRAIN split only.

4. PER-TASK ATTRIBUTION. Metrics recorded separately for each held-out task, so
   the failure can be attributed to specific chores rather than an average.

Protocol matches run_real_widowx.py: 6 three-task holdouts x 2 seeds, val_seen
split into fit/calibration/exchangeable-control, task-blocked deployment stream.

Run:  python3 experiments/run_experiments2.py
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

from online_cp import (  # noqa: E402
    FunctionalScorer, split_cp_stream, aci_stream, detrend_by_early_window,
    detection_metrics,
)
from run_real_widowx import (  # noqa: E402
    Probe, safe_loss, train_probe, embed_euclid_scores, make_fold, build_streams,
)

ALPHAS = [0.10, 0.15, 0.20, 0.25]
FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
DETREND_K = 10          # first 20% of a 50-step episode treated as nominal
K_SWEEP = [5, 8, 10, 15, 20]

HANDCRAFTED = [
    "action/cum_mean_token_entropy",
    "action/mean_token_entropy",
    "action/cum_max_token_entropy",
    "action/cum_mean_token_prob",
    "action/cum_dpos",
    "action/cum_drot",
    "action/dgripper",
]


def auroc(scores, labels):
    """labels: 1 = failure (should score high)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    n1, n0 = int(labels.sum()), int((1 - labels).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    order = scores.argsort()
    r = np.empty(len(scores), dtype=np.float64)
    r[order] = np.arange(1, len(scores) + 1)
    # tie-average
    uniq, inv, cnt = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, r)
    r = (sums / cnt)[inv]
    return float((r[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--token", default="mean")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    blob = torch.load(os.path.join(OUT, f"widowx_{a.token}.pt"), weights_only=False)
    feats_raw, succ, task_id = blob["features"], blob["success"], blob["task_id"]
    episode_idx, descs = blob["episode_idx"], blob["descs"]
    hand, hand_cols = blob["hand"], blob["hand_cols"]
    tasks = sorted(task_id.unique().tolist())
    hand_idx = {c: hand_cols.index(c) for c in HANDCRAFTED}
    print(f"loaded {tuple(feats_raw.shape)} features, {tuple(hand.shape)} handcrafted")

    combos = list(itertools.combinations(tasks, 3))
    pick = np.random.default_rng(0).choice(len(combos), size=a.folds, replace=False)
    holdouts = [set(combos[i]) for i in sorted(pick)]
    print("holdout folds:", [sorted(h) for h in holdouts])

    rows_cp, rows_early, rows_task, rows_k = [], [], [], []
    t0 = time.time()

    for fi, held in enumerate(holdouts):
        for seed in range(a.seeds):
            sp = make_fold(task_id, succ, held, seed=1000 * fi + seed)
            tr, vs, vu = sp["train"], sp["val_seen"], sp["val_unseen"]
            assert not (set(task_id[tr].tolist()) & held)

            pool = feats_raw[tr].reshape(-1, feats_raw.shape[-1])
            mu, sd = pool.mean(0), pool.std(0).clamp(min=1e-6)
            feats = (feats_raw - mu) / sd
            streams, bounds = build_streams(vu, task_id, episode_idx, held, seed)

            # ---- base score curves, one per detector -----------------------
            base = {}
            base["SAFE-MLP"] = train_probe(feats, succ, tr, "safe", "cumsum", seed, a.epochs)
            base["H&S"] = train_probe(feats, succ, tr, "hns", "none", seed, a.epochs)
            base["SAFE-Embed"] = embed_euclid_scores(feats, succ, tr)

            # handcrafted: sign fixed on the TRAIN split only
            lab_tr = (succ[tr].numpy() == 0).astype(int)
            for c in HANDCRAFTED:
                raw = hand[:, :, hand_idx[c]].numpy().astype(np.float64)
                raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
                sgn = 1.0 if auroc(raw[tr.numpy()].max(1), lab_tr) >= 0.5 else -1.0
                base[f"hand:{c.replace('action/', '')}"] = sgn * raw

            # ---- add the task-relative variant of every detector ------------
            curves_all = {}
            for name, cur in base.items():
                curves_all[name] = np.asarray(cur, dtype=np.float64)
                curves_all[f"{name} [task-rel]"] = detrend_by_early_window(cur, DETREND_K)

            vu_np = vu.numpy()
            lab_vu = (succ[vu].numpy() == 0).astype(int)

            # val_seen split: identical for every detector, so fix it once here.
            vs_succ = vs[succ[vs] == 1].numpy()
            vs_fail = vs[succ[vs] == 0].numpy()
            _rng = np.random.default_rng(7 * seed + fi)
            perm = _rng.permutation(vs_succ)
            n_fit = max(2, int(round(len(perm) * 0.25)))
            n_cal = max(2, int(round(len(perm) * 0.45)))
            exch = np.concatenate([perm[n_fit + n_cal:], vs_fail])
            _rng.shuffle(exch)

            for name, cur in curves_all.items():
                # ---- AUROC, full episode and per held-out task -------------
                rows_task.append(dict(fold=fi, seed=seed, detector=name, task=-1,
                                      auroc=auroc(cur[vu_np].max(1), lab_vu),
                                      n=len(vu_np)))
                for t in sorted(held):
                    m = vu_np[task_id[vu].numpy() == t]
                    rows_task.append(dict(
                        fold=fi, seed=seed, detector=name, task=t,
                        auroc=auroc(cur[m].max(1), (succ[m].numpy() == 0).astype(int)),
                        n=len(m)))

                # ---- early detection --------------------------------------
                for f in FRACTIONS:
                    kk = max(1, int(round(f * cur.shape[1])))
                    rows_early.append(dict(fold=fi, seed=seed, detector=name,
                                           frac=f,
                                           auroc=auroc(cur[vu_np][:, :kk].max(1), lab_vu)))

                # ---- conformal coverage -----------------------------------
                fit_c, cal_c = cur[perm[:n_fit]], cur[perm[n_fit:n_fit + n_cal]]
                st_map = {"exch": exch, "blocked": streams["blocked"]}

                for alpha in ALPHAS:
                    scorer = FunctionalScorer(fit_c, alpha)
                    cal_s = scorer(cal_c)
                    for order, st in st_map.items():
                        s_st = scorer(cur[st])
                        ok = (succ[st].numpy() == 1)
                        for m_name, res in [
                            ("split CP (SAFE)", split_cp_stream(cal_s, s_st, ok, alpha)),
                            ("ACI g=0.05 (grow cal)",
                             aci_stream(cal_s, s_st, ok, alpha, 0.05, grow_calibration=True)),
                        ]:
                            dm = detection_metrics(res, cur[st], scorer, ok)
                            rows_cp.append(dict(
                                fold=fi, seed=seed, detector=name, alpha=alpha,
                                method=m_name, order=order,
                                coverage=res["coverage"], nominal=1 - alpha,
                                cov_gap=res["coverage"] - (1 - alpha), **dm))

            # ---- sensitivity of the transform to k -------------------------
            for name in ["SAFE-MLP", "H&S", "SAFE-Embed"]:
                for k in K_SWEEP:
                    cur = detrend_by_early_window(base[name], k)
                    scorer = FunctionalScorer(cur[perm[:n_fit]], 0.20)
                    cal_s = scorer(cur[perm[n_fit:n_fit + n_cal]])
                    st = streams["blocked"]
                    res = split_cp_stream(cal_s, scorer(cur[st]),
                                          (succ[st].numpy() == 1), 0.20)
                    rows_k.append(dict(fold=fi, seed=seed, detector=name, k=k,
                                       auroc=auroc(cur[vu_np].max(1), lab_vu),
                                       coverage=res["coverage"],
                                       cov_gap=res["coverage"] - 0.80))

            print(f"  fold {fi} seed {seed} done ({time.time() - t0:.0f}s)")

    for nm, rows in [("cp", rows_cp), ("early", rows_early),
                     ("task", rows_task), ("ksweep", rows_k)]:
        pd.DataFrame(rows).to_csv(os.path.join(OUT, f"r2_{nm}{a.tag}.csv"), index=False)

    # ---------------- headline console summary ------------------------------
    cp = pd.DataFrame(rows_cp)
    tk = pd.DataFrame(rows_task)
    main_cp = cp[(cp.order == "blocked") & (cp.alpha == 0.20)
                 & (cp.method == "split CP (SAFE)")]
    full = tk[tk.task == -1]

    print("\n" + "=" * 96)
    print("TASK-RELATIVE SCORING, unseen tasks (alpha=0.20, split CP, mean over 12 runs)")
    print("=" * 96)
    print(f"{'detector':<30s}{'AUROC base':>12s}{'AUROC t-rel':>13s}"
          f"{'cov base':>11s}{'cov t-rel':>11s}{'|gap| base':>12s}{'|gap| t-rel':>13s}")
    print("-" * 96)
    for d in ["SAFE-MLP", "H&S", "SAFE-Embed"] + [f"hand:{c.replace('action/', '')}"
                                                  for c in HANDCRAFTED]:
        b, r = d, f"{d} [task-rel]"
        ab, ar = full[full.detector == b].auroc.mean(), full[full.detector == r].auroc.mean()
        cb = main_cp[main_cp.detector == b]
        cr = main_cp[main_cp.detector == r]
        gb, gr = cb.cov_gap.abs().mean(), cr.cov_gap.abs().mean()
        star = "  <--" if (ar > ab and gr < gb) else ""
        print(f"{d:<30s}{ab:>12.3f}{ar:>13.3f}"
              f"{cb.coverage.mean():>11.3f}{cr.coverage.mean():>11.3f}"
              f"{gb:>12.3f}{gr:>13.3f}{star}")
    print("=" * 96)
    print("<-- marks detectors where the transform improved BOTH AUROC and calibration")
    print(f"\nwrote r2_cp/early/task/ksweep{a.tag}.csv  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
