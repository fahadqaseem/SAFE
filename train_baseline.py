"""
train_baseline.py  –  SAFE baseline probe (time-weighted loss)
==============================================================
Fully self-contained: does NOT import from failure_prob.model
(which would chain-import wandb which has a broken install).

Instead this script inlines:
  • the LSTM architecture
  • the time-weighted BCE loss
  • the data loading / splitting logic

Device: mps (Apple M4 Pro) → cpu fallback
Outputs:
    baseline_model.ckpt
    baseline_val_scores.pkl    {split: (scores_list, labels_arr)}
    baseline_features.pkl      {split: (feats_list, labels_arr)}
"""

import os, sys, glob, pickle, random, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── Device ──────────────────────────────────────────────────────────────────
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[train_baseline] device = {DEVICE}")

# ── Reproducibility ─────────────────────────────────────────────────────────
SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ── Hyper-parameters ─────────────────────────────────────────────────────────
HP = dict(
    token_idx          = -1,          # last token of the 7 tokens per step
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
    use_time_weighting = True,
)


# ════════════════════════════════════════════════════════════════════════════
# 1. Data loading
# ════════════════════════════════════════════════════════════════════════════

def load_rollouts(data_path: str, token_idx: int = -1):
    """
    Returns list of dicts:
        features     : Tensor (T, D)
        label        : int  (1=success, 0=failure)
        task_id      : int
        episode_idx  : int
    """
    pkls = sorted(glob.glob(os.path.join(data_path, "**", "*.pkl"), recursive=True))
    if not pkls:
        raise RuntimeError(f"No .pkl files found under {data_path}")

    rollouts = []
    for pkl_path in pkls:
        with open(pkl_path, "rb") as f:
            raw = pickle.load(f)

        if "eposide_idx" in raw:
            raw["episode_idx"] = raw.pop("eposide_idx")

        hs = raw["hidden_states"]
        if isinstance(hs, list):
            hs = torch.stack(hs, dim=0)          # (T, n_tok, D)
        hs = hs.float()                           # (T, n_tok, D)
        feat = hs[:, token_idx, :]               # (T, D)

        rollouts.append({
            "features"    : feat,
            "label"       : int(raw.get("episode_success", 0)),
            "task_id"     : int(raw.get("task_id", 0)),
            "episode_idx" : int(raw.get("episode_idx", 0)),
        })

    print(f"Loaded {len(rollouts)} rollouts. "
          f"Success={sum(r['label'] for r in rollouts)}  "
          f"Fail={sum(1-r['label'] for r in rollouts)}")
    return rollouts


def split_rollouts(rollouts, seen_train_ratio=0.66, unseen_task_ratio=0.25):
    task_ids = list(set(r["task_id"] for r in rollouts))
    np.random.shuffle(task_ids)
    n_unseen      = max(1, round(unseen_task_ratio * len(task_ids)))
    unseen_ids    = set(task_ids[:n_unseen])
    seen_ids      = set(task_ids[n_unseen:])

    unseen = [r for r in rollouts if r["task_id"] in unseen_ids]
    seen   = [r for r in rollouts if r["task_id"] in seen_ids]

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


# ════════════════════════════════════════════════════════════════════════════
# 2. Dataset
# ════════════════════════════════════════════════════════════════════════════

def collate_rollouts(rollouts, device):
    """Pad a list of rollout dicts to a batch dict."""
    max_len = max(r["features"].shape[0] for r in rollouts)
    D       = rollouts[0]["features"].shape[-1]
    B       = len(rollouts)

    features   = torch.zeros(B, max_len, D, dtype=torch.float32)
    valid_masks= torch.zeros(B, max_len, dtype=torch.float32)
    labels     = torch.zeros(B, dtype=torch.float32)

    for i, r in enumerate(rollouts):
        T = r["features"].shape[0]
        features[i, :T]    = r["features"]
        valid_masks[i, :T] = 1.0
        labels[i]          = float(r["label"])

    return {
        "features"     : features.to(device),
        "valid_masks"  : valid_masks.to(device),
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

    def get_class_weights(self):
        n = len(self.rollouts)
        n_fail = sum(1 - r["label"] for r in self.rollouts)
        n_succ = n - n_fail
        # w[0]=fail weight, w[1]=success weight
        return [n / (2 * max(n_fail, 1)), n / (2 * max(n_succ, 1))]


# ════════════════════════════════════════════════════════════════════════════
# 3. Model
# ════════════════════════════════════════════════════════════════════════════

class LstmProbe(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, n_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers,
                            batch_first=True, dropout=dropout)
        self.fc   = nn.Linear(hidden_dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, batch):
        x    = batch["features"]               # (B, T, D)
        out, _= self.lstm(x)                   # (B, T, H)
        out  = self.drop(out)
        p    = torch.sigmoid(self.fc(out))     # (B, T, 1)
        return p


# ════════════════════════════════════════════════════════════════════════════
# 4. Loss (original SAFE time-weighted BCE)
# ════════════════════════════════════════════════════════════════════════════

def time_weighted_bce_loss(scores, valid_masks, success_labels,
                           weights, use_time_weighting=True):
    """
    scores         : (B, T)  – sigmoid failure probability
    valid_masks    : (B, T)
    success_labels : (B,)    – 1=success, 0=failure
    weights        : [w_fail, w_succ]
    """
    B, T = scores.shape
    seq_lengths = valid_masks.sum(dim=1).long()   # (B,)

    # Time weights (exponential decay from start)
    if use_time_weighting:
        t = torch.arange(T, device=scores.device).float()  # (T,)
        time_w = t.unsqueeze(0).expand(B, -1)              # (B, T)
        time_w = time_w / seq_lengths.unsqueeze(1).clamp(min=1)
        time_w = 5 * torch.exp(-3 * time_w) + 1
        time_w = time_w * valid_masks
        norm   = time_w.sum(-1) / seq_lengths.clamp(min=1)
        time_w = time_w / norm.unsqueeze(1).clamp(min=1e-8)
    else:
        time_w = valid_masks

    # BCE: failure is the positive class (score → 1 for failure)
    target    = (1 - success_labels).unsqueeze(1).expand_as(scores)   # (B, T)
    criterion = nn.BCELoss(reduction="none")
    losses    = criterion(scores, target)                               # (B, T)

    # Apply time weights only on failure samples
    fail_mask = (success_labels == 0)                                   # (B,)
    losses[fail_mask] = losses[fail_mask] * time_w[fail_mask]

    # Aggregate
    seq_loss     = (losses * valid_masks).sum(-1) / valid_masks.sum(-1).clamp(min=1)  # (B,)
    fail_loss    = (fail_mask.float() * seq_loss).sum()
    succ_loss    = ((1 - fail_mask.float()) * seq_loss).sum()
    monitor_loss = (weights[0] * fail_loss + weights[1] * succ_loss) / B
    return monitor_loss


# ════════════════════════════════════════════════════════════════════════════
# 5. Evaluation helpers
# ════════════════════════════════════════════════════════════════════════════

def get_scores(model, rollouts, device, batch_size=32):
    """Returns scores_list, labels_list, feats_list."""
    model.eval()
    scores_all, labels_all, feats_all, masks_all = [], [], [], []
    with torch.no_grad():
        for start in range(0, len(rollouts), batch_size):
            chunk = rollouts[start:start + batch_size]
            batch = collate_rollouts(chunk, device)
            s = model(batch).squeeze(-1)                # (b, T)
            scores_all.append(s.cpu())
            masks_all.append(batch["valid_masks"].cpu())
            labels_all.append(batch["success_labels"].cpu())
            feats_all.append(batch["features"].cpu())

    scores_cat = torch.cat(scores_all, dim=0)
    masks_cat  = torch.cat(masks_all,  dim=0)
    labels_cat = torch.cat(labels_all, dim=0)
    feats_cat  = torch.cat(feats_all,  dim=0)
    seq_len    = masks_cat.sum(-1).long()

    scores_list = [scores_cat[i, :seq_len[i]].numpy() for i in range(len(seq_len))]
    labels_list = labels_cat.numpy()
    feats_list  = [feats_cat[i, :seq_len[i]].mean(0).numpy() for i in range(len(seq_len))]
    return scores_list, labels_list, feats_list


# ════════════════════════════════════════════════════════════════════════════
# 6. Main
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

    optimizer  = torch.optim.Adam(model.parameters(), lr=HP["lr"])
    scheduler  = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=HP["lr_step_size"], gamma=HP["lr_gamma"])

    weights = datasets["train"].get_class_weights()
    print(f"\nClass weights: fail={weights[0]:.3f}  succ={weights[1]:.3f}")

    print("\n=== Baseline training (time-weighted BCE) ===")
    for epoch in range(HP["n_epochs"]):
        model.train()
        losses = []
        for batch in loaders["train"]:
            scores = model(batch).squeeze(-1)        # (B, T)
            loss   = time_weighted_bce_loss(
                        scores,
                        batch["valid_masks"],
                        batch["success_labels"],
                        weights,
                        use_time_weighting=HP["use_time_weighting"],
                     )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), HP["grad_max_norm"])
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{HP['n_epochs']}  loss={np.mean(losses):.4f}")

    torch.save(model.state_dict(), "baseline_model.ckpt")
    print("Saved baseline_model.ckpt")

    print("\n=== Evaluating ===")
    val_scores, val_features = {}, {}
    for split_name, ds in datasets.items():
        s_list, l_list, f_list = get_scores(model, ds.rollouts, DEVICE, HP["batch_size"])
        val_scores[split_name]   = (s_list, l_list)
        val_features[split_name] = (f_list, l_list)
        print(f"  {split_name:12s}: {len(s_list)} rollouts")

    with open("baseline_val_scores.pkl",  "wb") as f: pickle.dump(val_scores,   f)
    with open("baseline_features.pkl",    "wb") as f: pickle.dump(val_features, f)
    print("\nSaved baseline_val_scores.pkl  baseline_features.pkl\nDone.")


if __name__ == "__main__":
    main()
