from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import copy
import yaml


@dataclass
class RewardConfig:
    alpha_energy: float = 0.40
    alpha_qos: float = 0.35
    alpha_security: float = 0.15
    alpha_penalty: float = 0.10


@dataclass
class NetworkConfig:
    n_nodes: int = 100
    n_cluster_heads: int = 10
    field_width_m: float = 100.0
    field_height_m: float = 100.0
    min_node_separation_m: float = 5.0
    communication_radius_m: float = 25.0
    fallback_k_nearest: int = 2
    sink_position: Tuple[float, float] = (50.0, 0.0)
    sink_position_mode: str = "edge_center"  # edge_center or field_center
    cluster_rotation_period_slots: int = 300
    cluster_head_probability: float = 0.10
    queue_capacity_packets: int = 32
    service_rate_packets_per_slot: int = 2
    shadowing_sigma_db: float = 4.0
    pdr_floor: float = 0.05
    pdr_ceiling: float = 0.995


@dataclass
class EnergyConfig:
    initial_energy_j: float = 2.0
    death_threshold_j: float = 0.01
    e_elec_j_per_bit: float = 50e-9
    epsilon_amp_j_per_bit_m_alpha: float = 100e-12
    path_loss_exponent: float = 2.5
    sensing_energy_j_per_packet: float = 5e-6
    dqn_inference_energy_j: float = 2e-6
    dqn_train_step_energy_j: float = 8e-6
    control_cpu_energy_j: float = 1e-6


@dataclass
class TrafficConfig:
    slot_duration_s: float = 1.0
    data_packet_bits: int = 512
    control_packet_bits: int = 4096
    sense_interval_s: int = 10
    hotspot_cluster_fraction: float = 0.30
    hotspot_burst_interval_s: int = 2
    burst_on_slots: int = 60
    burst_off_slots: int = 300
    control_message_probability_per_slot: float = 0.01


@dataclass
class RLConfig:
    state_dim: int = 14
    action_dim: int = 8
    hidden_units: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.98
    replay_buffer_size: int = 50000
    batch_size: int = 64
    target_update_period: int = 250
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5000
    local_training_frequency: int = 1
    weight_decay: float = 1e-6
    gradient_clip_norm: float = 1.0
    local_steps_per_round: int = 12
    reward: RewardConfig = field(default_factory=RewardConfig)


@dataclass
class FederatedConfig:
    global_rounds: int = 300
    frl_round_interval_s: int = 60
    client_fraction: float = 0.30
    eligibility_energy_threshold: float = 0.25
    eligibility_key_threshold: float = 0.30
    eligibility_trust_threshold: float = 0.60
    aggregation_rule: str = "trust_weighted_fedavg"
    norm_clip_value: float = 2.0
    model_quantization_bits: int = 16
    model_compression_ratio: float = 0.20
    qkd_replenish_every_m_rounds: int = 1
    use_trust_weights: bool = True
    use_key_status_features: bool = True


@dataclass
class QKDConfig:
    enabled: bool = True
    link_length_km: float = 10.0
    attenuation_db_per_km: float = 0.25
    baseline_secret_key_rate_bps: float = 50_000.0
    nominal_qber: float = 0.02
    error_correction_efficiency: float = 1.15
    key_pool_capacity_bits: int = 10_000_000
    target_refresh_interval_s: int = 300
    session_key_bits_frl: int = 256
    session_key_bits_broadcast: int = 256
    session_key_bits_control: int = 128
    min_secure_fraction: float = 0.80
    classical_key_compromise_fraction: float = 0.0


@dataclass
class AttackConfig:
    selective_forwarding_fraction: float = 0.0
    selective_forwarding_drop_probability: float = 0.50
    false_metric_fraction: float = 0.0
    false_metric_bias: float = 0.30
    model_poisoning_fraction: float = 0.0
    model_poisoning_scale: float = 0.20
    free_riding_fraction: float = 0.0
    elevated_qber: Optional[float] = None
    classical_key_compromise_fraction: float = 0.0


@dataclass
class ScenarioConfig:
    name: str = "S1_nominal"
    description: str = ""
    traffic_mode: str = "periodic"  # periodic or bursty
    enable_hotspots: bool = False
    total_slots: int = 18000
    n_seeds_default: int = 10
    schemes: List[str] = field(default_factory=lambda: ["frl_qkd", "heuristic", "frl_noqkd"])
    network: NetworkConfig = field(default_factory=NetworkConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    traffic: TrafficConfig = field(default_factory=TrafficConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    qkd: QKDConfig = field(default_factory=QKDConfig)
    attacks: AttackConfig = field(default_factory=AttackConfig)
    assumptions: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_time_s(self) -> float:
        return self.total_slots * self.traffic.slot_duration_s

    @property
    def frl_interval_slots(self) -> int:
        return max(1, int(round(self.federated.frl_round_interval_s / self.traffic.slot_duration_s)))

    @property
    def derived_global_rounds(self) -> int:
        return max(1, self.total_slots // self.frl_interval_slots)


def _update_dataclass(obj: Any, payload: Dict[str, Any]) -> Any:
    clone = copy.deepcopy(obj)
    for key, value in payload.items():
        current = getattr(clone, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            setattr(clone, key, _update_dataclass(current, value))
        else:
            setattr(clone, key, value)
    return clone


def load_config(path: str | Path) -> ScenarioConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    cfg = ScenarioConfig()
    cfg = _update_dataclass(cfg, payload)
    return cfg


def dump_config(cfg: ScenarioConfig, path: str | Path) -> None:
    path = Path(path)
    path.write_text(yaml.safe_dump(asdict(cfg), sort_keys=False), encoding="utf-8")


def as_dict(cfg: ScenarioConfig) -> Dict[str, Any]:
    return asdict(cfg)
