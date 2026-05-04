from __future__ import annotations
from typing import Dict
import torch
import torch.nn.functional as F


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
    reduction: str = 'mean',
) -> torch.Tensor:
    """Sigmoid Focal Loss — lebih efektif dari BCE untuk class imbalance extreme.

    Focal Loss mengurangi bobot pada easy negatives yang mendominasi dataset,
    supaya model fokus belajar dari hard examples (frame di sekitar note boundary).

    Args:
        logits: raw logits sebelum sigmoid, shape [B, T]
        targets: ground truth 0/1, shape [B, T]
        alpha: weight untuk positive class (0.75 = positif 3x lebih penting)
        gamma: focusing parameter (2.0 = standar, semakin besar = semakin fokus ke hard examples)
    """
    probs = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

    # p_t = probability of correct class
    p_t = probs * targets + (1 - probs) * (1 - targets)

    # Focal modulating factor: (1 - p_t)^gamma
    # Easy examples (p_t tinggi) → weight kecil
    # Hard examples (p_t rendah) → weight besar
    focal_weight = (1 - p_t) ** gamma

    # Alpha weighting: positive class gets alpha, negative gets (1-alpha)
    alpha_weight = alpha * targets + (1 - alpha) * (1 - targets)

    focal_loss = alpha_weight * focal_weight * ce_loss

    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    return focal_loss


def compute_losses(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    cfg: Dict,
    pos_weight: torch.Tensor
) -> Dict[str, torch.Tensor]:
    event_target = batch['event'].float().clamp(0, 1)
    lane_target = batch['lane']

    event_logits = outputs['event_logits']
    lane_logits = outputs['lane_logits']

    # =========================
    # EVENT LOSS
    # =========================
    use_focal = cfg['train'].get('use_focal_loss', True)

    if use_focal:
        focal_alpha = float(cfg['train'].get('focal_alpha', 0.75))
        focal_gamma = float(cfg['train'].get('focal_gamma', 2.0))
        event_loss = sigmoid_focal_loss(
            event_logits,
            event_target,
            alpha=focal_alpha,
            gamma=focal_gamma,
        )
    else:
        # Fallback ke BCE dengan pos_weight
        pw = torch.nan_to_num(pos_weight, nan=1.0, posinf=100.0, neginf=1.0)
        event_loss = F.binary_cross_entropy_with_logits(
            event_logits,
            event_target,
            pos_weight=pw,
        )

    # =========================
    # LANE LOSS
    # =========================
    label_smoothing = float(cfg['train'].get('label_smoothing', 0.0))

    if (lane_target != -100).any():
        lane_loss = F.cross_entropy(
            lane_logits.transpose(1, 2),
            lane_target,
            ignore_index=-100,
            label_smoothing=label_smoothing,
        )
    else:
        lane_loss = torch.tensor(0.0, device=lane_logits.device)

    # =========================
    # TOTAL LOSS
    # =========================
    total = (
        cfg['train']['event_loss_weight'] * event_loss
        + cfg['train']['lane_loss_weight'] * lane_loss
    )

    return {
        'loss': total,
        'event_loss': event_loss,
        'lane_loss': lane_loss,
    }