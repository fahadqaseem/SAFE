"""
Real-data experiment: OpenVLA-on-WidowX rollouts, SAFE's detectors, and online
conformal prediction under task shift.

Setting. SAFE calibrates its conformal band on held-out episodes of SEEN tasks
and deploys it on UNSEEN tasks (`routines.py:104`). We reproduce that exactly,
measure how far the realised false-alarm rate drifts from the nominal alpha, and
test whether ACI / DtACI pull it back.

Per fold: 3 of the 8 tasks are held out as the deployment stream; the other 5 are
split by episode into train (66%) / val_seen (34%), mirroring
`conf/dataset/openvla_widowx.yaml` (unseen_task_ratio 0.25, seen_train_ratio 0.66).

Following SAFE's `eval_functional_conformal`, the val_seen SUCCESS curves build
the prediction/modulation trajectories and the calibration quantile pool
(`metrics.py:546`); we carve off a third disjoint slice for the control below.

Three evaluation streams:
    exch      held-out episodes of the SAME tasks as calibration. Exchangeable,
              so split CP MUST reach nominal coverage here. This is the harness
              self-test; nothing else means anything until it passes.
    blocked   unseen tasks, task by task -- sequential deployment, real shift.
    shuffled  the same unseen episodes, permuted. Note split CP's band is fixed,
              so its marginal coverage is exactly order-invariant and identical
              to `blocked`; the contrast matters only for the adaptive methods,
              which do depend on order.

Run:  python3 experiments/run_real_widowx.py [--folds 6] [--seeds 2] [--epochs 300]
"""

import argparse
import importlib.util
import itertools
import json
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
    FunctionalScorer, quantile, split_cp_stream, aci_stream, dtaci_stream,
    detection_metrics,
)


def _load(name, relpath):
    """Load a SAFE module by file path, bypassing the package __init__ (which
    imports wandb/hydra). The source under test is unmodified."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


safe_utils = _load("safe_model_utils", "failure_prob/model/utils.py")
hns = _load("safe_hns_loss", "failure_prob/model/hns_loss.py")
get_time_weight = safe_utils.get_time_weight
aggregate_monitor_loss = safe_utils.aggregate_monitor_loss

ALPHAS = [0.10, 0.15, 0.20, 0.25]
GAMMAS = [0.005, 0.01, 0.05]


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------

class Probe(nn.Module):
    """SAFE's IndepModel architecture: per-timestep MLP -> sigmoid -> cumsum."""

    def __init__(self, dim, hidden=256, agg="cumsum"):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1), nn.Sigmoid())
        self.agg = agg

    def forward(self, x):
        s = self.net(x)
        if self.agg == "cumsum":
            s = torch.cumsum(s, dim=-2)
        elif self.agg != "none":
            raise ValueError(self.agg)
        return s.squeeze(-1)


def safe_loss(scores, valid_masks, labels, weights, use_time_weighting=False):
    """SAFE's hinge, via SAFE's own get_time_weight / aggregate_monitor_loss."""
    tw = get_time_weight(use_time_weighting, valid_masks).to(scores)
    seq_loss_success = torch.relu(scores - 0)
    seq_loss_fail = tw * (-scores)                      # use_threshold=False default
    losses = (labels == 1).float()[:, None] * seq_loss_success + \
             (labels == 0).float()[:, None] * seq_loss_fail
    loss, _, _ = aggregate_monitor_loss(losses, valid_masks, labels, weights)
    return loss


def train_probe(feats, succ, tr_idx, loss_type, agg, seed, n_epochs, lr=1e-3, bs=64):
    torch.manual_seed(seed)
    model = Probe(feats.shape[-1], agg=agg)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    n_f = int((succ[tr_idx] == 0).sum()); n_s = int((succ[tr_idx] == 1).sum())
    weights = [len(tr_idx) / (n_f + 1), len(tr_idx) / (n_s + 1)]   # RolloutDataset.__init__

    g = torch.Generator().manual_seed(seed)
    masks_full = torch.ones(feats.shape[0], feats.shape[1])

    for _ in range(n_epochs):
        perm = tr_idx[torch.randperm(len(tr_idx), generator=g)]
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            if len(idx) < 4:
                continue
            s = model(feats[idx])
            m, y = masks_full[idx], succ[idx]
            if loss_type == "safe":
                loss = safe_loss(s, m, y, weights)
            else:
                loss, _ = hns.hns_loss(s, m, (y == 0), margin_r=0.5, margin_o=0.3,
                                       lambda_intra=1.0,
                                       use_inter=loss_type in ("inter", "hns"),
                                       use_intra=loss_type in ("intra", "hns"))
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        return model(feats).numpy()


def embed_euclid_scores(feats, succ, tr_idx, topk=10, pca_dim=128, cumsum=True):
    """SAFE's training-free embedding baseline: mean distance to the k nearest
    training-success features, accumulated over time.

    Mirrors `EmbedModel` with distance="euclid", topk=10, cumsum=True -- one of
    the configs swept in scripts/batch_training/submit_openvla_widowx.bash.
    Reduced to PCA-128 first (fit on train) purely to make the 26.6k x 10k
    pairwise distance tractable on CPU; noted as a deviation.
    """
    N, T, D = feats.shape
    flat = feats.reshape(-1, D)
    tr_flat = feats[tr_idx].reshape(-1, D)

    mu = tr_flat.mean(0, keepdim=True)
    Xc = tr_flat - mu
    # economy SVD on the training features gives the PCA basis
    V = torch.linalg.svd(Xc, full_matrices=False)[2][:pca_dim].T     # (D, pca_dim)

    ref = (feats[tr_idx][succ[tr_idx] == 1].reshape(-1, D) - mu) @ V  # (M, p)
    allp = (flat - mu) @ V                                            # (N*T, p)

    d = torch.cdist(allp, ref)                                        # (N*T, M)
    k = min(topk, ref.shape[0])
    s = d.topk(k, largest=False, dim=1).values.mean(1).reshape(N, T)
    if cumsum:
        s = torch.cumsum(s, dim=1)
    return s.numpy()


DETECTORS = {
    "SAFE-MLP":        dict(kind="probe", loss="safe", agg="cumsum"),
    "SAFE-Embed":      dict(kind="embed"),
    "H&S inter+intra": dict(kind="probe", loss="hns",  agg="none"),
}


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def make_fold(task_id, succ, held_out, seed, train_ratio=0.66):
    tid = task_id.numpy()
    is_held = np.isin(tid, list(held_out))
    seen = np.where(~is_held)[0]

    rng = np.random.default_rng(seed)
    rng.shuffle(seen)
    n_tr = int(round(train_ratio * len(seen)))
    return {
        "train": torch.tensor(np.sort(seen[:n_tr])),
        "val_seen": torch.tensor(np.sort(seen[n_tr:])),
        "val_unseen": torch.tensor(np.where(is_held)[0]),
    }


def build_streams(unseen_idx, task_id, episode_idx, held_out, seed):
    """blocked = task by task (episode order within task); shuffled = permuted."""
    tid = task_id.numpy(); eid = episode_idx.numpy()
    idx = unseen_idx.numpy()

    blocked = []
    for t in sorted(held_out):
        sel = idx[tid[idx] == t]
        blocked.extend(sel[np.argsort(eid[sel])].tolist())
    blocked = np.asarray(blocked)

    shuffled = blocked.copy()
    np.random.default_rng(seed + 977).shuffle(shuffled)

    boundaries = np.cumsum([int((tid[idx] == t).sum()) for t in sorted(held_out)])[:-1]
    return {"blocked": blocked, "shuffled": shuffled}, boundaries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    tasks = sorted(task_id.unique().tolist())
    print(f"loaded {tuple(feats_raw.shape)}  {len(tasks)} tasks  "
          f"{int(succ.sum())}/{len(succ)} success")

    # Fixed set of 3-task holdouts, deterministic across runs.
    combos = list(itertools.combinations(tasks, 3))
    pick = np.random.default_rng(0).choice(len(combos), size=a.folds, replace=False)
    holdouts = [set(combos[i]) for i in sorted(pick)]
    print("holdout folds:", [sorted(h) for h in holdouts])

    rows, curves_store = [], {}
    gate_a, gate_b = [], []
    t_start = time.time()

    for fi, held in enumerate(holdouts):
        for seed in range(a.seeds):
            sp = make_fold(task_id, succ, held, seed=1000 * fi + seed)
            tr, vs, vu = sp["train"], sp["val_seen"], sp["val_unseen"]

            assert not (set(task_id[tr].tolist()) & held), "task leaked into train"
            assert not (set(task_id[vs].tolist()) & held), "task leaked into val_seen"

            # standardize on train episodes only
            pool = feats_raw[tr].reshape(-1, feats_raw.shape[-1])
            mu, sd = pool.mean(0), pool.std(0).clamp(min=1e-6)
            feats = (feats_raw - mu) / sd

            streams, bounds = build_streams(vu, task_id, episode_idx, held, seed)

            for det_name, cfg in DETECTORS.items():
                if cfg["kind"] == "probe":
                    curves = train_probe(feats, succ, tr, cfg["loss"], cfg["agg"],
                                         seed, a.epochs)
                else:
                    curves = embed_euclid_scores(feats, succ, tr)

                # trajectory-level AUROC on unseen tasks (order-independent)
                peaks = curves[vu.numpy()].max(1)
                lab = (succ[vu].numpy() == 0).astype(int)     # 1 = failure
                order = peaks.argsort()
                r = np.empty(len(peaks)); r[order] = np.arange(1, len(peaks) + 1)
                n1, n0 = lab.sum(), (1 - lab).sum()
                auroc = ((r[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
                         if n1 and n0 else np.nan)

                # --- val_seen successes: fit / calibrate / exchangeable test ---
                # SAFE splits val_seen successes 30/70 into (pred+modulation,
                # calibration quantile) -- metrics.py:546. We carve off a third,
                # disjoint slice from the SAME tasks to serve as the
                # exchangeability control: it is drawn from the same
                # distribution as the calibration pool, so split CP MUST reach
                # nominal coverage on it. Shuffling the unseen-task stream is
                # not a control -- split CP's band is fixed, so its marginal
                # coverage is exactly order-invariant.
                vs_succ = vs[succ[vs] == 1].numpy()
                vs_fail = vs[succ[vs] == 0].numpy()
                rng = np.random.default_rng(7 * seed + fi)
                perm = rng.permutation(vs_succ)
                n_fit = max(2, int(round(len(perm) * 0.25)))
                n_cal = max(2, int(round(len(perm) * 0.45)))
                fit_curves = curves[perm[:n_fit]]
                cal_curves = curves[perm[n_fit:n_fit + n_cal]]
                exch_idx = np.concatenate([perm[n_fit + n_cal:], vs_fail])
                rng.shuffle(exch_idx)
                streams["exch"] = exch_idx

                for alpha in ALPHAS:
                    scorer = FunctionalScorer(fit_curves, alpha)
                    gate_a.append(scorer.verify_against_safe(cal_curves, alpha))
                    cal_scores = scorer(cal_curves)

                    for order_name, st in streams.items():
                        s_stream = scorer(curves[st])
                        ok = (succ[st].numpy() == 1)
                        cur = curves[st]

                        methods = {
                            "split CP (SAFE)": split_cp_stream(cal_scores, s_stream, ok, alpha),
                            "split CP (n+1 corr.)": split_cp_stream(cal_scores, s_stream, ok,
                                                                    alpha, finite_sample=True),
                            "DtACI": dtaci_stream(cal_scores, s_stream, ok, alpha),
                            "DtACI (grow cal)": dtaci_stream(cal_scores, s_stream, ok, alpha,
                                                             grow_calibration=True),
                        }
                        for g in GAMMAS:
                            methods[f"ACI g={g}"] = aci_stream(cal_scores, s_stream, ok, alpha, g)
                            methods[f"ACI g={g} (grow cal)"] = aci_stream(
                                cal_scores, s_stream, ok, alpha, g, grow_calibration=True)

                        gate_b.append(float(np.max(np.abs(
                            aci_stream(cal_scores, s_stream, ok, alpha, 0.0)["q_t"]
                            - methods["split CP (SAFE)"]["q_t"]))))

                        for m_name, res in methods.items():
                            dm = detection_metrics(res, cur, scorer, ok)
                            rows.append(dict(
                                fold=fi, seed=seed, held=",".join(map(str, sorted(held))),
                                detector=det_name, alpha=alpha, method=m_name,
                                order=order_name, auroc=auroc,
                                coverage=res["coverage"],
                                false_alarm=res["false_alarm_rate"],
                                nominal_cov=1 - alpha,
                                cov_gap=res["coverage"] - (1 - alpha),
                                n_success=res["n_success"],
                                alpha_final=float(res["alpha_t"][-1]),
                                **dm))

                        if fi == 0 and seed == 0 and alpha == 0.20:
                            for m_name, res in methods.items():
                                curves_store[f"{det_name}|{order_name}|{m_name}"] = dict(
                                    err=res["err"], alpha_t=res["alpha_t"],
                                    q_t=res["q_t"], fired=res["fired"],
                                )
                            curves_store[f"__meta__|{order_name}"] = dict(
                                is_success=(succ[st].numpy() == 1),
                                task=task_id[st].numpy(), bounds=bounds,
                            )

            print(f"  fold {fi} seed {seed} done ({time.time() - t_start:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"real_cp_results{a.tag}.csv"), index=False)
    import pickle
    with open(os.path.join(OUT, f"real_cp_curves{a.tag}.pkl"), "wb") as f:
        pickle.dump(curves_store, f)

    # ---- gates ----------------------------------------------------------
    print("\n" + "=" * 78)
    print("GATE A  our band == SAFE's get_one_sided_prediction_band : max REL diff %.2e"
          % max(gate_a))
    print("GATE B  ACI(gamma=0) == split CP                          : max diff %.2e"
          % max(gate_b))

    print("\nGATE C  split CP on the EXCHANGEABLE stream (held-out episodes of the")
    print("        same tasks as calibration). Must land on nominal.")
    for det in DETECTORS:
        ctrl = df[(df.order == "exch") & (df.method == "split CP (SAFE)")
                  & (df.detector == det)]
        cells = []
        for al in ALPHAS:
            v = ctrl[ctrl.alpha == al].coverage
            cells.append(f"{al:.2f}: {v.mean():.3f}+-{v.std():.3f} (nom {1 - al:.2f})")
        print(f"    {det:<18s} " + "  ".join(cells))

    print("\nSAFE-MLP unseen-task trajectory AUROC: %.3f +- %.3f"
          % (df[df.detector == 'SAFE-MLP'].auroc.mean(),
             df[df.detector == 'SAFE-MLP'].groupby(['fold', 'seed']).auroc.first().std()))
    print("=" * 78)
    print(f"wrote real_cp_results{a.tag}.csv  ({len(df)} rows, {time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
