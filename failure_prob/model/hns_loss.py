"""
Hide-and-Seek losses, implemented against SAFE's (B, T) score / valid_mask convention.

Reference: "Hide-and-Seek in Trajectories: Discovering Failure Signals for VLA
Runtime Monitoring", arXiv:2605.30834.

    L = L_inter + lambda * L_intra

    L_inter = mean_{tau_f, tau_s} relu( m_r - (max_t s_t^{f} - max_t s_t^{s}) )
    L_intra = mean_{tau_f}        relu( m_o - (postmean(s^f) - premean(s^f)) )
    t_onset = argmax_t (s_t - s_{t-1})

Three implementation details that the equations alone do not give you, and that
matter for whether this trains at all:

1. `t_onset` is a function of the model's own scores. If gradients flow through
   the argmax-selected split point, the model can drive the loss down by making
   one arbitrary step-difference large instead of by localizing failure. The
   onset is therefore computed under `no_grad` and treated as a constant target.

2. At initialization the scores are noise, so the onset is noise. Applying
   L_intra from step 0 lets the model lock onto a spurious split. We support a
   warmup during which only L_inter is applied (`intra_warmup_frac`).

3. L_intra only constrains the *difference* between the post-onset and pre-onset
   means inside failure trajectories. It is invariant to a global additive shift
   and says nothing whatsoever about success trajectories. Trained alone it is
   degenerate. L_inter is what anchors the two classes apart. The `loss_type`
   switch below makes this testable rather than assumed.
"""

import torch


def _graph_zero(scores: torch.Tensor) -> torch.Tensor:
    """An exact zero that stays connected to the autograd graph.

    A batch may contain only successes or only failures, in which case a
    contrastive term is undefined. Returning a detached zero makes the total
    loss lose its grad_fn and .backward() raises. Multiplying by zero keeps the
    graph intact and contributes no gradient.
    """
    return scores.sum() * 0.0


def masked_max(scores: torch.Tensor, valid_masks: torch.Tensor, beta: float | None = None) -> torch.Tensor:
    """Max over valid timesteps. If `beta` is given, a logsumexp soft-max instead.

    Args:
        scores: (B, T)
        valid_masks: (B, T), 1 for valid timesteps
        beta: soft-max sharpness. None -> hard max.
    Returns:
        (B,)
    """
    neg_inf = torch.finfo(scores.dtype).min
    masked = torch.where(valid_masks > 0.5, scores, torch.full_like(scores, neg_inf))
    if beta is None:
        return masked.max(dim=1).values
    return (1.0 / beta) * torch.logsumexp(beta * masked, dim=1)


def estimate_onset(
    scores: torch.Tensor,
    valid_masks: torch.Tensor,
    min_side: int = 1,
) -> torch.Tensor:
    """Proxy failure onset: t_onset = argmax_t (s_t - s_{t-1}), computed WITHOUT gradient.

    The onset is clamped into [min_side, L - min_side] so that both the pre-onset
    and post-onset windows are non-empty (the paper's 1/(t_onset - 1) and
    1/(N - t_onset + 1) normalizers both require this).

    Args:
        scores: (B, T)
        valid_masks: (B, T)
        min_side: minimum number of timesteps on each side of the split.
    Returns:
        (B,) long tensor. Index t means: pre-onset is [0, t), post-onset is [t, L).
    """
    with torch.no_grad():
        B, T = scores.shape
        lengths = valid_masks.sum(dim=1).long()                       # (B,)

        diffs = scores[:, 1:] - scores[:, :-1]                        # (B, T-1)
        # A diff at index i corresponds to the transition into timestep i+1, so it
        # is a candidate onset t = i + 1. Valid candidates satisfy
        # min_side <= t <= L - min_side.
        t_cand = torch.arange(1, T, device=scores.device).unsqueeze(0).expand(B, -1)
        ok = (t_cand >= min_side) & (t_cand <= (lengths.unsqueeze(1) - min_side))

        neg_inf = torch.finfo(scores.dtype).min
        diffs = torch.where(ok, diffs, torch.full_like(diffs, neg_inf))

        onset = diffs.argmax(dim=1) + 1                               # (B,)

        # Sequences too short for a valid split fall back to the midpoint.
        too_short = lengths < (2 * min_side + 1)
        onset = torch.where(too_short, torch.clamp(lengths // 2, min=1), onset)
        onset = torch.clamp(onset, min=1)
    return onset


def intra_trajectory_loss(
    scores: torch.Tensor,
    valid_masks: torch.Tensor,
    fail_mask: torch.Tensor,
    margin_o: float,
    onset: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hinge that pushes the post-onset mean score above the pre-onset mean.

    Applied to failure trajectories only.

    Args:
        scores: (B, T)
        valid_masks: (B, T)
        fail_mask: (B,) bool, True for failure trajectories
        margin_o: required margin m_o
        onset: (B,) precomputed onset, or None to estimate it here
    Returns:
        loss: scalar
        onset: (B,) the onset actually used
    """
    if onset is None:
        onset = estimate_onset(scores, valid_masks)

    B, T = scores.shape
    t_idx = torch.arange(T, device=scores.device).unsqueeze(0).expand(B, -1)
    pre = ((t_idx < onset.unsqueeze(1)) & (valid_masks > 0.5)).to(scores)
    post = ((t_idx >= onset.unsqueeze(1)) & (valid_masks > 0.5)).to(scores)

    eps = 1e-8
    pre_mean = (scores * pre).sum(1) / (pre.sum(1) + eps)             # (B,)
    post_mean = (scores * post).sum(1) / (post.sum(1) + eps)          # (B,)

    per_traj = torch.relu(margin_o - (post_mean - pre_mean))          # (B,)

    if fail_mask.sum() == 0:
        return _graph_zero(scores), onset
    loss = (per_traj * fail_mask.to(scores)).sum() / fail_mask.to(scores).sum()
    return loss, onset


def inter_trajectory_loss(
    scores: torch.Tensor,
    valid_masks: torch.Tensor,
    fail_mask: torch.Tensor,
    margin_r: float,
    beta: float | None = None,
) -> torch.Tensor:
    """Hinge over all (failure, success) pairs on the per-trajectory max score.

    Vectorized over the full pair set rather than looped: for the hinge
    relu(m_r - (max_f - max_s)) the pair matrix is (n_fail, n_succ).
    """
    peaks = masked_max(scores, valid_masks, beta=beta)                # (B,)
    succ_mask = ~fail_mask

    if fail_mask.sum() == 0 or succ_mask.sum() == 0:
        return _graph_zero(scores)

    peak_f = peaks[fail_mask]                                         # (n_f,)
    peak_s = peaks[succ_mask]                                         # (n_s,)
    pairs = torch.relu(margin_r - (peak_f.unsqueeze(1) - peak_s.unsqueeze(0)))
    return pairs.mean()


def hns_loss(
    scores: torch.Tensor,
    valid_masks: torch.Tensor,
    fail_mask: torch.Tensor,
    margin_r: float = 0.5,
    margin_o: float = 0.3,
    lambda_intra: float = 1.0,
    beta: float | None = None,
    use_inter: bool = True,
    use_intra: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Total Hide-and-Seek loss. Returns (loss, logs)."""
    logs = {}
    total = _graph_zero(scores)

    if use_inter:
        l_inter = inter_trajectory_loss(scores, valid_masks, fail_mask, margin_r, beta=beta)
        total = total + l_inter
        logs["inter_loss"] = float(l_inter)

    if use_intra:
        l_intra, onset = intra_trajectory_loss(scores, valid_masks, fail_mask, margin_o)
        total = total + lambda_intra * l_intra
        logs["intra_loss"] = float(l_intra)
        if fail_mask.sum() > 0:
            lengths = valid_masks.sum(1).clamp(min=1)
            logs["onset_rel_mean"] = float((onset[fail_mask] / lengths[fail_mask]).mean())

    logs["monitor_loss"] = float(total)
    return total, logs
