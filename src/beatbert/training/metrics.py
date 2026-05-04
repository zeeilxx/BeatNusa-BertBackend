from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch


def frame_classification_metrics(event_logits: torch.Tensor, lane_logits: torch.Tensor, event_target: torch.Tensor, lane_target: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    event_prob = torch.sigmoid(event_logits)
    event_pred = (event_prob >= threshold).long()
    event_true = event_target.long()

    tp = int(((event_pred == 1) & (event_true == 1)).sum().item())
    fp = int(((event_pred == 1) & (event_true == 0)).sum().item())
    fn = int(((event_pred == 0) & (event_true == 1)).sum().item())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)

    lane_pred = lane_logits.argmax(dim=-1)
    mask = lane_target != -100
    lane_acc = float((lane_pred[mask] == lane_target[mask]).float().mean().item()) if mask.any() else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1, 'lane_acc': lane_acc}


def decode_events(event_logits: np.ndarray, lane_logits: np.ndarray, frame_times_ms: np.ndarray, threshold: float) -> List[Tuple[float, int]]:
    event_prob = 1.0 / (1.0 + np.exp(-event_logits))
    idxs = np.where((event_prob >= threshold) & (frame_times_ms >= 0))[0]
    lanes = lane_logits.argmax(axis=-1)
    return [(float(frame_times_ms[i]), int(lanes[i])) for i in idxs]


def event_level_metrics(pred_events: List[Tuple[float, int]], true_events: List[Tuple[float, int]], tolerance_ms: float) -> Dict[str, float]:
    matched_true = set()
    tp = 0
    lane_tp = 0
    offsets = []
    for pt, plane in pred_events:
        best_j = None
        best_dt = None
        for j, (tt, tlane) in enumerate(true_events):
            if j in matched_true:
                continue
            dt = abs(pt - tt)
            if dt <= tolerance_ms and (best_dt is None or dt < best_dt):
                best_dt = dt
                best_j = j
        if best_j is not None:
            matched_true.add(best_j)
            tp += 1
            offsets.append(best_dt)
            if pred_events and plane == true_events[best_j][1]:
                lane_tp += 1
    fp = len(pred_events) - tp
    fn = len(true_events) - tp
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    median_offset = float(np.median(offsets)) if offsets else float('nan')
    mae_offset = float(np.mean(offsets)) if offsets else float('nan')
    lane_acc = lane_tp / max(1, tp)
    return {
        'event_precision': precision,
        'event_recall': recall,
        'event_f1': f1,
        'lane_event_acc': lane_acc,
        'median_offset_ms': median_offset,
        'mae_offset_ms': mae_offset,
    }
