"""
Cache the real OpenVLA-on-WidowX rollouts into a single tensor for fast reuse.

Each episode pickle holds:
    hidden_states     list of 50 tensors, each (7, 4096) bfloat16
                      -> 50 timesteps x 7 action tokens x 4096 hidden dim
    task_id           int, the REAL task identity
    episode_success   bool
    task_description  str

The 7 action tokens are reduced to one 4096-vector per timestep the way SAFE's
own WidowX sweep does (scripts/batch_training/submit_openvla_widowx.bash passes
`dataset.token_idx_rel=mean,0.0,1.0`); `mean` is our primary setting.

Grouping is by `task_id`, NOT by folder name. The 19 folders collapse to 8 real
tasks -- `task_lift_red_bottle_1..4` are all task_id 2, `task_put_the_carrot_on_plate_*`
and `put_the_carrot_on_plate_1` are all task_id 6. Splitting on folders would put
the same task in both train and held-out and silently inflate every zero-shot number.

Run:  python3 experiments/cache_widowx.py [--token mean|first|last]
"""

import argparse
import glob
import os
import pickle
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "experiments", "out")
DATA = os.path.join(REPO, "openvla_widowx")

TOKEN_REDUCE = {
    "mean":  lambda h: h.mean(dim=1),      # (T, 7, D) -> (T, D)
    "first": lambda h: h[:, 0],
    "last":  lambda h: h[:, -1],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", choices=list(TOKEN_REDUCE), default="mean",
                    help="how to reduce the 7 action tokens (SAFE's token_idx_rel)")
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(DATA, "*", "*.pkl")))
    if not files:
        sys.exit(f"no pickles under {DATA}")
    print(f"found {len(files)} episode pickles")

    feats, succ, tids, eps, folders = [], [], [], [], []
    descs = {}
    t0 = time.time()

    for n, f in enumerate(files):
        d = pickle.load(open(f, "rb"))
        h = torch.stack(d["hidden_states"])              # (T, 7, 4096) bfloat16
        assert h.ndim == 3 and h.shape[1] == 7 and h.shape[2] == 4096, h.shape
        x = TOKEN_REDUCE[a.token](h.float())             # (T, 4096)

        feats.append(x)
        succ.append(int(d["episode_success"]))
        tids.append(int(d["task_id"]))
        eps.append(int(d["eposide_idx"]))                # sic: typo is in the data
        folders.append(os.path.basename(os.path.dirname(f)))
        descs[int(d["task_id"])] = d["task_description"]

        if (n + 1) % 100 == 0:
            print(f"  {n + 1}/{len(files)}  ({time.time() - t0:.0f}s)")

    lengths = sorted({x.shape[0] for x in feats})
    print(f"\nepisode lengths present: {lengths}")
    assert lengths == [50], f"expected uniform length 50, got {lengths}"

    features = torch.stack(feats)                        # (N, 50, 4096)
    success = torch.tensor(succ, dtype=torch.long)
    task_id = torch.tensor(tids, dtype=torch.long)

    # ---- verification gates from the plan --------------------------------
    n_tasks = len(task_id.unique())
    print(f"\nepisodes      {len(features)}")
    print(f"successes     {int(success.sum())}  ({success.float().mean():.3f})")
    print(f"distinct task_id  {n_tasks}  -> {sorted(task_id.unique().tolist())}")
    assert len(features) == 532, len(features)
    assert n_tasks == 8, n_tasks
    assert int(success.sum()) == 244, int(success.sum())

    print(f"\n{'tid':>4s} {'n':>5s} {'fail':>5s} {'succ':>5s} {'SR':>6s}  description")
    for t in sorted(task_id.unique().tolist()):
        m = task_id == t
        s = int(success[m].sum()); n = int(m.sum())
        print(f"{t:>4d} {n:>5d} {n - s:>5d} {s:>5d} {s / n:>6.2f}  {descs[t]}")

    # folder -> task_id collapse, the thing that would have leaked
    from collections import defaultdict
    f2t = defaultdict(set)
    for fo, t in zip(folders, tids):
        f2t[fo].add(t)
    multi = {k: v for k, v in f2t.items() if len(v) > 1}
    assert not multi, f"folder spans several task_ids: {multi}"
    n_folders = len(f2t)
    print(f"\n{n_folders} folders collapse to {n_tasks} task_ids "
          f"(splitting on folders would leak {n_folders - n_tasks} duplicate-task groups)")

    path = os.path.join(OUT, f"widowx_{a.token}.pt")
    torch.save({
        "features": features, "success": success, "task_id": task_id,
        "episode_idx": torch.tensor(eps, dtype=torch.long),
        "folders": folders, "descs": descs, "token_reduce": a.token,
    }, path)
    print(f"\nwrote {path}  ({os.path.getsize(path) / 1e6:.0f} MB, {time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
