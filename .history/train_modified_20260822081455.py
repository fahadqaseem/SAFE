"""
train_modified.py  –  Intra-trajectory contrastive loss probe
=============================================================
Fully self-contained (shares same data/model code as train_baseline.py
but replaces the loss function entirely).

Intra-trajectory contrastive loss:
  For each failure trajectory in a batch:
    1. diff[t] = score[t+1] - score[t]
    2. k = argmax(diff).detach()          ← no gradient through index
    3. pre_mean  = mean(scores[:k])
    4. post_mean = mean(scores[k:])
    5. loss      = relu(0.5 - (post_mean - pre_mean))

  For each success trajectory:
    loss = relu(scores).mean()  (suppress scores toward 0)

Outputs:
    modified_model.ckpt
    modified_val_scores.pkl
    modified_features.pkl
"""

import os, sys, glob, pickle, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[train_modified] device = {DEVICE}")

SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

HP = dict(
    token_idx          = -1,
    seen_train_ratio   = 0.66,
    unseen_task_ratio  = 0.25,
    hidden_dim         = 256,
    n_layers           = 1,
    dropout            = 0.0,
    n_epochs           = 100,
    batch_size         = 16,
    lr                 = 3e-4,
    lr_step_size       = 50,
    lr_gamma           = 0.5,
    grad_max_norm      = 1.0,
    intra_margin       = 0.5,
)


# ════════════════════════════════════════════════════════════════════════════
# 1. Data  (identical to train_baseline.py)
# ════════════════════════════════════════════════════════════════════════════

def load_rollouts(data_path, token_idx=-1):
    pkls = sorted(glob.glob(os.path.join(data_path, "**", "*.pkl"), recursive=True))
    if not pkls:
        raise RuntimeError(f"No .pkl files under {data_path}")
    rollouts = []
    for p in pkls:
        with open(p, "rb") as f:
            raw = pickle.load(f)
        if "eposide_idx" in raw:
            raw["episode_idx"] = raw.pop("eposide_idx")
        hs = raw["hidden_states"]
        if isinstance(hs, list):
            hs = torch.stack(hs, dim=0)
        hs   = hs.float()
        feat = hs[:, token_idx, :]
        rollouts.append({
            "features"   : feat,
            "label"      : int(raw.get("episode_success", 0)),
            "task_id"    : int(raw.get("task_id", 0)),
            "episode_idx": int(raw.get("episode_idx", 0)),
        })
    print(f"Loaded {len(rollouts)} rollouts. "
          f"Success={sum(r['label'] for r in rollouts)}  "
          f"Fail={sum(1-r['label'] for r in rollouts)}")
    return rollouts


def split_rollouts(rollouts, seen_train_ratio=0.66, unseen_task_ratio=0.25):
    task_ids  = list(set(r["task_id"] for r in rollouts))
    np.random.shuffle(task_ids)
    n_unseen  = max(1, round(unseen_task_ratio * len(task_ids)))
    unseen_ids= set(task_ids[:n_unseen])
    seen_ids  = set(task_ids[n_unseen:])
    unseen    = [r for r in rollouts if r["task_id"] in unseen_ids]
    seen      = [r for r in rollouts if r["task_id"] in seen_ids]
    train, val_seen = [], []
    for tid in seen_ids:
        task_r = [r for r in seen if r["task_id"] == tid]
        perm   = torch.randperm(len(task_r)).tolist()
        n_tr   = int(seen_train_ratio * len(task_r))
        train   += [task_r[i] for i in perm[:n_tr]]
        val_seen+= [task_r[i] for i in perm[n_tr:]]
    splits = {"train": train, "val_seen": val_seen}
    if unseen:
        splits["val_unseen"] = unseen
    for k, v in splits.items():
        ns = sum(r["label"] for r in v)
        print(f"  {k:12s}: {len(v):4d}  (succ={ns}, fail={len(v)-ns})")
    return splits


def collate_rollouts(rollouts, device):
    max_len = max(r["features"].shape[0] for r in rollouts)
    D, B    = rollouts[0]["features"].shape[-1], len(rollouts)
    features    = torch.zeros(B, max_len, D)
    valid_masks = torch.zeros(B, max_len)
    labels      = torch.zeros(B)
    for i, r in enumerate(rollouts):
        T = r["features"].shape[0]
        features[i, :T]    = r["features"]
        valid_masks[i, :T] = 1.0
        labels[i]          = float(r["label"])
    return {
        "features"      : features.to(device),
        "valid_masks"   : valid_masks.to(device),
        "success_labels": labels.to(device),
    }


class RolloutDataset(Dataset):
    def __init__(self, rollouts, device):
        self.rollouts = rollouts
        self.device   = device
    def __len__(self):
        return len(self.rollouts)
    def __getitem__(self, idx):
        return self.rollouts[idx]
    def collate_fn(self, batch):
        return collate_rollouts(batch, self.device)


# ════════════════════════════════════════════════════════════════════════════
# 2. Model  (identical to train_baseline.py)
# ════════════════════════════════════════════════════════════════════════════

class LstmProbe(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, n_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers,
                            batch_first=True, dropout=dropout)
        self.fc   = nn.Linear(hidden_dim, 1)
        self.drop = nn.Dropout(dropout)
    def forward(self, batch):
        x       = batch["features"]
        out, _  = self.lstm(x)
        out     = self.drop(out)
        return torch.sigmoid(self.fc(out))      # (B, T, 1)


# ════════════════════════════════════════════════════════════════════════════
# 3. Intra-trajectory contrastive loss
# ════════════════════════════════════════════════════════════════════════════

def intra_contrastive_loss(scores, valid_masks, success_labels, margin=0.5):
    """
    scores         : (B, T)  failure probability in [0, 1]
    valid_masks    : (B, T)
    success_labels : (B,)    1=success, 0=failure
    margin         : ReLU margin  (default 0.5)
    """
    B, T = scores.shape
    fail_mask = (success_labels == 0)   # (B,)

    intra_terms   = []
    success_terms = []

    for i in range(B):
        L = int(valid_masks[i].sum().item())
        if L < 2:
            continue
        s = scores[i, :L]                          # (L,) – gradients live here

        if fail_mask[i]:
            # ── 1. step-to-step differences ─────────────────────────────
            diff = s[1:] - s[:-1]                  # (L-1,)

            # ── 2. proxy onset  (detached index – no gradient) ──────────
            k = int(diff.detach().argmax().item()) + 1   # in [1, L-1]
            k = max(1, min(k, L - 1))

            # ── 3. pre / post means ──────────────────────────────────────
            pre_mean  = s[:k].mean()
            post_mean = s[k:].mean()

            # ── 4. ReLU margin loss ──────────────────────────────────────
            intra_terms.append(torch.relu(margin - (post_mean - pre_mean)))

        else:
            # Success: suppress the failure score across the whole trajectory
            success_terms.append(torch.relu(s).mean())

    # Combine – both terms need at least one entry to be non-trivial
    acc = scores.sum() * 0.0              # differentiable zero on correct device
    if intra_terms:
        acc = acc + torch.stack(intra_terms).mean()
    if success_terms:
        acc = acc + torch.stack(success_terms).mean()
    return acc


# ════════════════════════════════════════════════════════════════════════════
# 4. Evaluation helper  (identical to train_baseline.py)
# ════════════════════════════════════════════════════════════════════════════

def get_scores(model, rollouts, device, batch_size=32):
    model.eval()
    scores_all, labels_all, feats_all, masks_all = [], [], [], []
    with torch.no_grad():
        for start in range(0, len(rollouts), batch_size):
            chunk = rollouts[start:start + batch_size]
            batch = collate_rollouts(chunk, device)
            s = model(batch).squeeze(-1)
            scores_all.append(s.cpu())
            masks_all.append(batch["valid_masks"].cpu())
            labels_all.append(batch["success_labels"].cpu())
            feats_all.append(batch["features"].cpu())
    scores_cat = torch.cat(scores_all, dim=0)
    masks_cat  = torch.cat(masks_all,  dim=0)
    labels_cat = torch.cat(labels_all, dim=0)
    feats_cat  = torch.cat(feats_all,  dim=0)
    seq_len    = masks_cat.sum(-1).long()
    scores_list= [scores_cat[i, :seq_len[i]].numpy() for i in range(len(seq_len))]
    labels_list= labels_cat.numpy()
    feats_list = [feats_cat[i, :seq_len[i]].mean(0).numpy() for i in range(len(seq_len))]
    return scores_list, labels_list, feats_list


# ════════════════════════════════════════════════════════════════════════════
# 5. Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    rollouts = load_rollouts("openvla_widowx", token_idx=HP["token_idx"])
    splits   = split_rollouts(rollouts, HP["seen_train_ratio"], HP["unseen_task_ratio"])

    input_dim = rollouts[0]["features"].shape[-1]
    print(f"Input dim: {input_dim}")

    datasets = {k: RolloutDataset(v, DEVICE) for k, v in splits.items()}
    loaders  = {
        k: DataLoader(ds, batch_size=HP["batch_size"],
                      shuffle=(k == "train"), num_workers=0,
                      collate_fn=ds.collate_fn)
        for k, ds in datasets.items()
    }

    model = LstmProbe(input_dim, HP["hidden_dim"], HP["n_layers"], HP["dropout"])
    model.to(DEVICE)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=HP["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=HP["lr_step_size"], gamma=HP["lr_gamma"])

    print("\n=== Modified training (intra-trajectory contrastive loss) ===")
    for epoch in range(HP["n_epochs"]):
        model.train()
        losses = []
        for batch in loaders["train"]:
            scores = model(batch).squeeze(-1)         # (B, T)
            loss   = intra_contrastive_loss(
                        scores,
                        batch["valid_masks"],
                        batch["success_labels"],
                        margin=HP["intra_margin"],
                     )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), HP["grad_max_norm"])
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{HP['n_epochs']}  loss={np.mean(losses):.4f}")

    torch.save(model.state_dict(), "modified_model.ckpt")
    print("Saved modified_model.ckpt")

    print("\n=== Evaluating ===")
    val_scores, val_features = {}, {}
    for split_name, ds in datasets.items():
        s_list, l_list, f_list = get_scores(model, ds.rollouts, DEVICE, HP["batch_size"])
        val_scores[split_name]   = (s_list, l_list)
        val_features[split_name] = (f_list, l_list)
        print(f"  {split_name:12s}: {len(s_list)} rollouts")

    with open("modified_val_scores.pkl",  "wb") as f: pickle.dump(val_scores,   f)
    with open("modified_features.pkl",    "wb") as f: pickle.dump(val_features, f)
    print("\nSaved modified_val_scores.pkl  modified_features.pkl\nDone.")


if __name__ == "__main__":
    main()
