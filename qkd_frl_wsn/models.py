from __future__ import annotations

import numpy as np
import torch
from torch import nn


class DQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_units: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters())


def to_numpy_state(state: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(state, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    return arr.reshape(-1).astype(np.float32)
