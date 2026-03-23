from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Set

import numpy as np

from .config import AttackConfig


@dataclass
class AttackState:
    compromised_forwarders: Set[int]
    false_metric_nodes: Set[int]
    poisoned_clients: Set[int]
    free_riders: Set[int]

    @classmethod
    def build(cls, node_ids: Iterable[int], cfg: AttackConfig, rng: np.random.Generator) -> "AttackState":
        node_ids = list(node_ids)
        n = len(node_ids)
        def choose(frac: float) -> Set[int]:
            k = int(round(frac * n))
            if k <= 0:
                return set()
            return set(int(x) for x in rng.choice(node_ids, size=min(k, n), replace=False))
        compromised_forwarders = choose(max(cfg.selective_forwarding_fraction, cfg.false_metric_fraction))
        false_metric_nodes = choose(cfg.false_metric_fraction)
        poisoned_clients = choose(cfg.model_poisoning_fraction)
        free_riders = choose(cfg.free_riding_fraction)
        return cls(compromised_forwarders, false_metric_nodes, poisoned_clients, free_riders)

    def is_forwarder_compromised(self, node_id: int) -> bool:
        return int(node_id) in self.compromised_forwarders

    def is_false_metric(self, node_id: int) -> bool:
        return int(node_id) in self.false_metric_nodes

    def is_poisoned_client(self, node_id: int) -> bool:
        return int(node_id) in self.poisoned_clients

    def is_free_rider(self, node_id: int) -> bool:
        return int(node_id) in self.free_riders


def poison_update(delta: Dict[str, object], scale: float) -> Dict[str, object]:
    import torch
    out = {}
    for k, v in delta.items():
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
            noise = torch.randn_like(v) * scale * (v.std().item() + 1e-3)
            out[k] = v + noise
        else:
            out[k] = v
    return out
