from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


ACTION_DUTY = {
    0: 0.8, 1: 0.8, 2: 0.5, 3: 0.5,
    4: 0.2, 5: 0.2, 6: 0.5, 7: 0.8,
}
ACTION_PARTICIPATE = {
    0: True, 1: True, 2: True, 3: True,
    4: True, 5: False, 6: True, 7: True,
}


def heuristic_action(local_view: Dict[str, float]) -> int:
    if local_view["trust"] < 0.5:
        return 6
    if local_view["energy"] < 0.3 or local_view["queue"] < 0.2:
        return 4
    if local_view["pdr_best"] > 0.85:
        return 1
    if local_view["hop_best"] < local_view["hop_current"]:
        return 3
    return 2 if local_view["neighbor_energy_best"] > local_view["energy"] else 0


@dataclass
class CentralizedPolicy:
    duty_mode: str = "balanced"

    def choose_action(self, global_state: np.ndarray) -> int:
        avg_energy, avg_queue, avg_trust, avg_key = global_state[:4]
        if avg_trust < 0.6:
            return 6
        if avg_energy < 0.35:
            return 4
        if avg_queue > 0.55:
            return 1
        if avg_key < 0.3:
            return 5
        return 2


def action_to_tuple(action_id: int) -> Tuple[float, bool]:
    return ACTION_DUTY[int(action_id)], ACTION_PARTICIPATE[int(action_id)]
