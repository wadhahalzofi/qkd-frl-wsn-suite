from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import math

from .config import QKDConfig


def binary_entropy(q: float) -> float:
    q = max(1e-12, min(1 - 1e-12, q))
    return -(q * math.log2(q) + (1 - q) * math.log2(1 - q))


def qkd_penalty(qber: float) -> float:
    return max(0.0, 1.0 - 2.0 * binary_entropy(qber))


@dataclass
class QKDEngine:
    cfg: QKDConfig
    key_pool_bits: float = 0.0
    last_refresh_slot: int = 0
    secure_messages: int = 0
    insecure_messages: int = 0
    bits_used_frl: int = 0
    bits_used_bcast: int = 0
    bits_used_ctrl: int = 0

    def __post_init__(self) -> None:
        self.key_pool_bits = float(self.cfg.key_pool_capacity_bits * 0.6)

    def current_rate_bps(self, qber_override: float | None = None) -> float:
        if not self.cfg.enabled:
            return 0.0
        qber = self.cfg.nominal_qber if qber_override is None else qber_override
        rate = self.cfg.baseline_secret_key_rate_bps * (10 ** (-(self.cfg.attenuation_db_per_km * self.cfg.link_length_km) / 10.0))
        rate *= qkd_penalty(qber)
        return max(0.0, rate)

    def replenish(self, dt_s: float, current_slot: int, qber_override: float | None = None) -> float:
        rate = self.current_rate_bps(qber_override=qber_override)
        self.key_pool_bits = min(float(self.cfg.key_pool_capacity_bits), self.key_pool_bits + rate * dt_s)
        if current_slot - self.last_refresh_slot >= self.cfg.target_refresh_interval_s:
            self.last_refresh_slot = current_slot
        return rate

    def consume(self, frl_participants: int, control_messages: int, secure: bool = True) -> int:
        if not secure or not self.cfg.enabled:
            self.insecure_messages += frl_participants + 1 + control_messages
            return 0
        required = (
            frl_participants * self.cfg.session_key_bits_frl
            + self.cfg.session_key_bits_broadcast
            + control_messages * self.cfg.session_key_bits_control
        )
        available = int(self.key_pool_bits)
        used = min(required, available)
        self.key_pool_bits -= used
        self.bits_used_frl += min(frl_participants * self.cfg.session_key_bits_frl, used)
        rem = max(0, used - frl_participants * self.cfg.session_key_bits_frl)
        self.bits_used_bcast += min(self.cfg.session_key_bits_broadcast, rem)
        rem2 = max(0, rem - self.cfg.session_key_bits_broadcast)
        self.bits_used_ctrl += min(control_messages * self.cfg.session_key_bits_control, rem2)
        if used >= required:
            self.secure_messages += frl_participants + 1 + control_messages
        else:
            secured_messages = frl_participants + 1
            self.secure_messages += secured_messages
            self.insecure_messages += control_messages
        return used

    def secure_fraction(self) -> float:
        total = self.secure_messages + self.insecure_messages
        if total == 0:
            return 1.0 if self.cfg.enabled else 0.0
        return self.secure_messages / total

    def snapshot(self) -> Dict[str, float]:
        return {
            "key_pool_bits": float(self.key_pool_bits),
            "secure_fraction": float(self.secure_fraction()),
            "bits_used_frl": float(self.bits_used_frl),
            "bits_used_bcast": float(self.bits_used_bcast),
            "bits_used_ctrl": float(self.bits_used_ctrl),
        }
