from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch


def _clone_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def aggregate_deltas(
    global_state: Dict[str, torch.Tensor],
    deltas: List[Tuple[int, Dict[str, torch.Tensor]]],
    trust_scores: Dict[int, float],
    weights: Dict[int, float],
    clip_norm: float = 2.0,
    use_trust: bool = True,
) -> Dict[str, torch.Tensor]:
    if not deltas:
        return _clone_state_dict(global_state)

    total_weight = 0.0
    accum = {k: torch.zeros_like(v) for k, v in global_state.items()}
    for node_id, delta in deltas:
        flat_sq = 0.0
        for v in delta.values():
            if torch.is_floating_point(v):
                flat_sq += float(torch.sum(v * v).item())
        norm = max(np.sqrt(flat_sq), 1e-9)
        scale = min(1.0, clip_norm / norm)
        w = float(weights.get(node_id, 1.0))
        if use_trust:
            w *= float(trust_scores.get(node_id, 1.0))
        total_weight += w
        for k, v in delta.items():
            if torch.is_floating_point(v):
                accum[k] += v * scale * w
    if total_weight <= 0:
        return _clone_state_dict(global_state)

    new_state = {}
    for k, base in global_state.items():
        if torch.is_floating_point(base):
            new_state[k] = base + accum[k] / total_weight
        else:
            new_state[k] = base.clone()
    return new_state


def detect_update_anomaly(delta: Dict[str, torch.Tensor], reference_scale: float) -> bool:
    flat_sq = 0.0
    for v in delta.values():
        if torch.is_floating_point(v):
            flat_sq += float(torch.sum(v * v).item())
    norm = np.sqrt(flat_sq)
    return bool(norm > max(1.0, 3.0 * reference_scale))
