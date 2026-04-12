"""
focal_loss.py — Focal Loss for class-imbalanced classification

Focal Loss down-weights easy examples and focuses training on hard,
misclassified ones.  Especially effective for medical imaging where
class imbalance is extreme.

Reference: Lin et al., "Focal Loss for Dense Object Detection" (2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma:  focusing parameter (default 2.0). Higher = more focus on hard examples.
        alpha:  per-class weight tensor, or scalar for binary. None = uniform.
        label_smoothing: optional label smoothing (0.0 = none).
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs:  raw logits [B, C]
            targets: class indices [B]
        """
        num_classes = inputs.size(1)

        # Compute log-softmax for numerical stability
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)

        # Optional label smoothing
        if self.label_smoothing > 0:
            smooth = self.label_smoothing / num_classes
            one_hot = torch.zeros_like(log_probs).scatter(1, targets.unsqueeze(1), 1.0)
            one_hot = one_hot * (1 - self.label_smoothing) + smooth
        else:
            one_hot = torch.zeros_like(log_probs).scatter(1, targets.unsqueeze(1), 1.0)

        # Gather probabilities for the true class
        pt = (probs * one_hot).sum(dim=1)

        # Focal modulating factor
        focal_weight = (1 - pt) ** self.gamma

        # Cross-entropy component
        ce = -(one_hot * log_probs).sum(dim=1)

        # Alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)[targets]
            focal_weight = focal_weight * alpha_t

        loss = focal_weight * ce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
