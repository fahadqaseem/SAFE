"""
A probe that is architecturally identical to SAFE's `IndepModel` but exposes the
loss and the temporal aggregation as two independent switches.

The point of this class is to make the comparison against SAFE a controlled one.
SAFE's MLP probe couples two things that Hide-and-Seek changes together:

  * the LOSS      - a per-timestep hinge driven by the trajectory-level label,
                    optionally reweighted over time (`use_time_weighting`)
  * the AGGREGATION - `cumsum` over the per-timestep sigmoid, on by default
                    (IndepModelConfig.cumsum = True)

Because a cumulative sum of strictly positive increments is monotonically
non-decreasing, the `cumsum` head *cannot* produce a score that stays flat and
then spikes, whatever the loss is. So any "SAFE smears, ours spikes" figure that
also flips `cumsum` off is confounded: it is not evidence about the loss.

Switching `loss_type` and `agg` separately on one shared architecture is what
separates the two effects.

    agg:        "cumsum" (SAFE default) | "rmean" | "none"
    loss_type:  "safe"  - SAFE's hinge, byte-for-byte the same code path
                "inter" - Hide-and-Seek inter-trajectory term only
                "intra" - Hide-and-Seek intra-trajectory term only (expected to
                          be degenerate; included so that it is measured rather
                          than asserted)
                "hns"   - L_inter + lambda * L_intra
"""

import torch
import torch.nn as nn

from .base import BaseModel
from .utils import get_time_weight, aggregate_monitor_loss
from .hns_loss import hns_loss

from failure_prob.conf import Config


def get_model(cfg: Config, input_dim: int) -> BaseModel:
    return HnsModel(cfg, input_dim)


def build_projector(cfg: Config, total_input_dim: int) -> nn.Sequential:
    """Identical construction to IndepModel.__init__, kept in sync deliberately."""
    projector = []
    if cfg.model.n_layers == 1:
        projector.append(nn.Linear(total_input_dim, 1))
    else:
        projector.append(nn.Linear(total_input_dim, cfg.model.hidden_dim))
        projector.append(nn.ReLU())
        for _ in range(cfg.model.n_layers - 2):
            projector.append(nn.Linear(cfg.model.hidden_dim, cfg.model.hidden_dim))
            projector.append(nn.ReLU())
        projector.append(nn.Linear(cfg.model.hidden_dim, 1))

    if cfg.model.final_act_layer == "sigmoid":
        projector.append(nn.Sigmoid())
    elif cfg.model.final_act_layer == "relu":
        projector.append(nn.ReLU())
    elif cfg.model.final_act_layer == "none":
        pass
    else:
        raise ValueError(f"Unknown final activation: {cfg.model.final_act_layer}")

    return nn.Sequential(*projector)


class HnsModel(BaseModel):
    def __init__(self, cfg: Config, input_dim: int):
        super().__init__(cfg, input_dim)
        self.total_input_dim = input_dim * cfg.model.n_history_steps
        self.projector = build_projector(cfg, self.total_input_dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch["features"]
        assert x.ndim == 3, f"Input dim mismatch: {x.ndim} != 3"
        assert x.shape[-1] == self.input_dim, f"Input dim mismatch: {x.shape[-1]} != {self.input_dim}"

        x = self.projector(x)                                   # (B, T, 1)

        agg = self.cfg.model.agg
        if agg == "cumsum":
            x = torch.cumsum(x, dim=-2)
        elif agg == "rmean":
            x = torch.cumsum(x, dim=-2)
            x = x / torch.arange(1, x.shape[1] + 1, device=x.device).view(1, -1, 1)
        elif agg == "none":
            pass
        else:
            raise ValueError(f"Unknown aggregation: {agg}")

        return x

    def forward_compute_loss(
        self,
        batch: dict[str, torch.Tensor],
        weights: list[float] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        valid_masks = batch["valid_masks"]
        labels = batch["success_labels"]

        scores = self(batch).squeeze(-1)                         # (B, T)
        loss_type = self.cfg.model.loss_type

        if loss_type == "safe":
            # Reproduces IndepModel.forward_compute_loss exactly.
            time_weights = get_time_weight(self.cfg.model.use_time_weighting, valid_masks).to(scores)
            higher_thresh = self.cfg.model.threshold
            lower_thresh = 0
            seq_loss_success = torch.relu(scores - lower_thresh)
            if self.cfg.model.use_threshold:
                seq_loss_fail = time_weights * torch.relu(higher_thresh - scores)
            else:
                seq_loss_fail = time_weights * (-scores)
            losses = (labels == 1).float()[:, None] * seq_loss_success + \
                (labels == 0).float()[:, None] * seq_loss_fail
            monitor_loss, success_loss, fail_loss = aggregate_monitor_loss(
                losses, valid_masks, labels, weights
            )
            return monitor_loss, {
                "monitor_loss": monitor_loss.item(),
                "success_loss": success_loss.item(),
                "fail_loss": fail_loss.item(),
            }

        # Hide-and-Seek variants. SAFE labels 1 = success, so failure is label == 0.
        fail_mask = (labels == 0)
        return hns_loss(
            scores,
            valid_masks,
            fail_mask,
            margin_r=self.cfg.model.margin_r,
            margin_o=self.cfg.model.margin_o,
            lambda_intra=self.cfg.model.lambda_intra,
            beta=(self.cfg.model.hns_beta if self.cfg.model.hns_beta > 0 else None),
            use_inter=loss_type in ("inter", "hns"),
            use_intra=loss_type in ("intra", "hns"),
        )
