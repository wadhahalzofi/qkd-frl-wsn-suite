from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import math
import random

import numpy as np
import torch

from .agents import DQNAgent
from .attacks import AttackState, poison_update
from .baselines import CentralizedPolicy, action_to_tuple, heuristic_action
from .config import ScenarioConfig
from .federated import aggregate_deltas, detect_update_anomaly
from .models import count_parameters
from .qkd import QKDEngine


@dataclass
class Node:
    node_id: int
    x: float
    y: float
    is_cluster_head: bool = False
    cluster_id: Optional[int] = None
    energy_j: float = 0.0
    alive: bool = True


@dataclass
class ClusterHeadRuntime:
    node_id: int
    agent: Optional[DQNAgent] = None
    trust_score: float = 1.0
    current_next_hop: Optional[int] = None
    advertised_pdr: Dict[int, float] = field(default_factory=dict)
    action_id: int = 0
    duty_cycle_ratio: float = 0.8
    participate_flag: bool = True
    queue: Deque[int] = field(default_factory=deque)
    recent_tx_success: Deque[int] = field(default_factory=lambda: deque(maxlen=50))
    recent_participation: Deque[int] = field(default_factory=lambda: deque(maxlen=20))
    arrival_ema: float = 0.0
    reward_ema: float = 0.0
    last_key_refresh_slot: int = 0
    last_round_loss: float = 0.0
    delivered_packets: int = 0


class ExperimentSimulator:
    def __init__(self, cfg: ScenarioConfig, scheme: str, seed: int = 0, device: str = "cpu") -> None:
        self.cfg = cfg
        self.scheme = scheme
        self.seed = int(seed)
        self.device = device
        self.rng = np.random.default_rng(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.nodes: Dict[int, Node] = {}
        self.cluster_heads: Dict[int, ClusterHeadRuntime] = {}
        self.cluster_members: Dict[int, List[int]] = defaultdict(list)
        self.ch_graph: Dict[int, List[int]] = defaultdict(list)
        self.ch_to_sink_distance: Dict[int, float] = {}
        self.current_slot = 0
        self.key_engine = QKDEngine(self._scheme_qkd_config())
        self.centralized_policy = CentralizedPolicy()
        self.attack_state: Optional[AttackState] = None
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.packet_delays: List[int] = []
        self.generated_packets = 0
        self.delivered_packets = 0
        self.dropped_packets = 0
        self.overflow_drops = 0
        self.control_messages = 0
        self.first_death_slot: Optional[int] = None
        self.l50_slot: Optional[int] = None
        self.l100_slot: Optional[int] = None
        self.reference_update_scale = 1.0
        self.model_size_params = 0
        self._build_topology()
        self._init_attacks()

    def _scheme_qkd_config(self):
        import copy
        qcfg = copy.deepcopy(self.cfg.qkd)
        if self.scheme == "frl_noqkd":
            qcfg.enabled = False
            qcfg.classical_key_compromise_fraction = max(qcfg.classical_key_compromise_fraction, self.cfg.attacks.classical_key_compromise_fraction)
        return qcfg

    def _sample_positions(self, n: int) -> List[Tuple[float, float]]:
        positions: List[Tuple[float, float]] = []
        attempts = 0
        while len(positions) < n and attempts < 100000:
            x = self.rng.uniform(0, self.cfg.network.field_width_m)
            y = self.rng.uniform(0, self.cfg.network.field_height_m)
            if all((x - px) ** 2 + (y - py) ** 2 >= self.cfg.network.min_node_separation_m ** 2 for px, py in positions):
                positions.append((x, y))
            attempts += 1
        if len(positions) < n:
            raise RuntimeError("Could not place all nodes with the minimum separation constraint.")
        return positions

    def _distance(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return float(math.dist(a, b))

    def _build_topology(self) -> None:
        positions = self._sample_positions(self.cfg.network.n_nodes)
        initial_energy = self.cfg.energy.initial_energy_j
        for node_id, (x, y) in enumerate(positions):
            self.nodes[node_id] = Node(node_id=node_id, x=x, y=y, energy_j=initial_energy)

        ch_ids = self._select_cluster_heads()
        for ch_id in ch_ids:
            self.nodes[ch_id].is_cluster_head = True
            self.cluster_heads[ch_id] = ClusterHeadRuntime(
                node_id=ch_id,
                agent=self._build_agent_if_needed(),
                last_key_refresh_slot=0,
            )

        for node_id, node in self.nodes.items():
            if node.is_cluster_head:
                node.cluster_id = node_id
                continue
            ch_id = self._nearest_cluster_head(node_id)
            node.cluster_id = ch_id
            self.cluster_members[ch_id].append(node_id)

        self._build_ch_graph()
        self._init_routes_and_views()

    def _select_cluster_heads(self) -> List[int]:
        candidate_ids = list(self.nodes.keys())
        self.rng.shuffle(candidate_ids)
        chosen: List[int] = []
        min_dist = self.cfg.network.communication_radius_m * 0.5
        for node_id in candidate_ids:
            pos = (self.nodes[node_id].x, self.nodes[node_id].y)
            if all(self._distance(pos, (self.nodes[c].x, self.nodes[c].y)) >= min_dist for c in chosen):
                chosen.append(node_id)
            if len(chosen) >= self.cfg.network.n_cluster_heads:
                break
        if len(chosen) < self.cfg.network.n_cluster_heads:
            remaining = [n for n in candidate_ids if n not in chosen]
            chosen.extend(remaining[: self.cfg.network.n_cluster_heads - len(chosen)])
        return chosen

    def _nearest_cluster_head(self, node_id: int) -> int:
        node = self.nodes[node_id]
        dists = []
        for ch_id in self.cluster_heads.keys():
            ch = self.nodes[ch_id]
            dists.append((self._distance((node.x, node.y), (ch.x, ch.y)), ch_id))
        dists.sort()
        return int(dists[0][1])

    def _build_ch_graph(self) -> None:
        rc = self.cfg.network.communication_radius_m
        sink_pos = self._sink_position()
        ch_ids = list(self.cluster_heads.keys())
        for ch_id in ch_ids:
            ch_pos = (self.nodes[ch_id].x, self.nodes[ch_id].y)
            self.ch_to_sink_distance[ch_id] = self._distance(ch_pos, sink_pos)

        for i in ch_ids:
            pos_i = (self.nodes[i].x, self.nodes[i].y)
            neighbors = []
            for j in ch_ids:
                if i == j:
                    continue
                d = self._distance(pos_i, (self.nodes[j].x, self.nodes[j].y))
                if d <= rc:
                    neighbors.append(j)
            if self.ch_to_sink_distance[i] <= rc:
                neighbors.append(-1)  # sink sentinel
            if len(neighbors) == 0:
                ranked = sorted(
                    [(self._distance(pos_i, (self.nodes[j].x, self.nodes[j].y)), j) for j in ch_ids if j != i],
                    key=lambda x: x[0],
                )
                neighbors.extend([j for _, j in ranked[: self.cfg.network.fallback_k_nearest]])
                if -1 not in neighbors and len(ranked) > 0 and self.ch_to_sink_distance[i] < ranked[0][0]:
                    neighbors.append(-1)
            self.ch_graph[i] = list(dict.fromkeys(neighbors))

    def _sink_position(self) -> Tuple[float, float]:
        if self.cfg.network.sink_position_mode == "field_center":
            return (self.cfg.network.field_width_m / 2.0, self.cfg.network.field_height_m / 2.0)
        return tuple(self.cfg.network.sink_position)

    def _build_agent_if_needed(self) -> Optional[DQNAgent]:
        if self.scheme in {"frl_qkd", "frl_noqkd"}:
            return DQNAgent(self.cfg.rl, device=self.device)
        return None

    def _init_routes_and_views(self) -> None:
        for ch_id, runtime in self.cluster_heads.items():
            runtime.current_next_hop = self._heuristic_next_hop(ch_id, security_aware=True)
            runtime.action_id = 0
            runtime.duty_cycle_ratio = 0.8
            runtime.participate_flag = True
        if self.scheme in {"frl_qkd", "frl_noqkd"}:
            any_agent = next(iter(self.cluster_heads.values())).agent
            if any_agent is not None:
                self.global_model = any_agent.get_state_dict()
                self.model_size_params = count_parameters(any_agent.policy)
        else:
            self.global_model = None
            self.model_size_params = 0

    def _init_attacks(self) -> None:
        node_ids = list(self.cluster_heads.keys())
        self.attack_state = AttackState.build(node_ids=node_ids, cfg=self.cfg.attacks, rng=self.rng)

    def _alive_fraction(self) -> float:
        alive = sum(1 for n in self.nodes.values() if n.alive)
        return alive / len(self.nodes)

    def _dead_counts(self) -> int:
        return sum(1 for n in self.nodes.values() if not n.alive)

    def _pdr_from_distance(self, d: float) -> float:
        rc = self.cfg.network.communication_radius_m
        base = max(0.0, 1.0 - (d / max(rc, 1e-9)) ** self.cfg.energy.path_loss_exponent)
        shadow = self.rng.normal(0.0, self.cfg.network.shadowing_sigma_db)
        penalty = 1.0 / (1.0 + math.exp((shadow - 1.0) / 2.0))
        pdr = 0.15 + 0.85 * base * penalty
        return float(np.clip(pdr, self.cfg.network.pdr_floor, self.cfg.network.pdr_ceiling))

    def _energy_tx(self, bits: int, distance_m: float) -> float:
        e = bits * self.cfg.energy.e_elec_j_per_bit
        e += bits * self.cfg.energy.epsilon_amp_j_per_bit_m_alpha * (distance_m ** self.cfg.energy.path_loss_exponent)
        return float(e)

    def _energy_rx(self, bits: int) -> float:
        return float(bits * self.cfg.energy.e_elec_j_per_bit)

    def _kill_if_needed(self, node_id: int) -> None:
        node = self.nodes[node_id]
        if node.alive and node.energy_j <= self.cfg.energy.death_threshold_j:
            node.alive = False
            if self.first_death_slot is None:
                self.first_death_slot = self.current_slot

    def _heuristic_next_hop(self, ch_id: int, security_aware: bool = True) -> int:
        current_hop = self.cluster_heads[ch_id].current_next_hop
        best_score = -1e18
        best = current_hop if current_hop is not None else self.ch_graph[ch_id][0]
        current_dist = self.ch_to_sink_distance[ch_id]
        for nb in self.ch_graph[ch_id]:
            if nb == -1:
                pdr = self._pdr_from_distance(current_dist)
                energy = 1.0
                trust = 1.0
                hop_bonus = 1.0
            else:
                d = self._distance((self.nodes[ch_id].x, self.nodes[ch_id].y), (self.nodes[nb].x, self.nodes[nb].y))
                pdr = self._advertised_pdr(nb, d)
                energy = self.nodes[nb].energy_j / self.cfg.energy.initial_energy_j
                trust = self.cluster_heads[nb].trust_score
                hop_bonus = max(0.0, 1.0 - self.ch_to_sink_distance[nb] / (current_dist + 1e-9))
            score = 0.5 * pdr + 0.3 * energy + 0.2 * hop_bonus
            if security_aware:
                score += 0.2 * trust
            if score > best_score:
                best_score = score
                best = nb
        return int(best)

    def _advertised_pdr(self, node_id: int, d: float) -> float:
        pdr = self._pdr_from_distance(d)
        if self.attack_state and self.attack_state.is_false_metric(node_id):
            pdr = min(1.0, pdr + self.cfg.attacks.false_metric_bias)
        return pdr

    def _hop_estimate(self, ch_id: int, visited: Optional[set] = None) -> int:
        if visited is None:
            visited = set()
        if ch_id in visited:
            return 5
        visited.add(ch_id)
        if -1 in self.ch_graph[ch_id]:
            return 1
        hops = []
        for nb in self.ch_graph[ch_id]:
            if nb == -1:
                hops.append(1)
            else:
                hops.append(1 + self._hop_estimate(nb, visited.copy()))
        return min(hops) if hops else 5

    def _build_local_state(self, ch_id: int) -> np.ndarray:
        runtime = self.cluster_heads[ch_id]
        node = self.nodes[ch_id]
        members = self.cluster_members.get(ch_id, [])
        member_energies = [self.nodes[m].energy_j / self.cfg.energy.initial_energy_j for m in members if self.nodes[m].alive]
        if not member_energies:
            member_energies = [0.0]
        q_norm = min(1.0, len(runtime.queue) / max(1, self.cfg.network.queue_capacity_packets))
        arr_norm = float(np.clip(runtime.arrival_ema / 3.0, 0.0, 1.0))
        delay_norm = float(np.clip((np.mean([self.current_slot - t for t in runtime.queue]) if runtime.queue else 0.0) / 20.0, 0.0, 1.0))
        recent_pdr = np.mean(runtime.recent_tx_success) if runtime.recent_tx_success else 0.8
        hop = min(1.0, self._hop_estimate(ch_id) / 5.0)
        n_nbr = min(1.0, len(self.ch_graph[ch_id]) / 6.0)
        k_pool_norm = min(1.0, self.key_engine.key_pool_bits / max(1.0, self.key_engine.cfg.key_pool_capacity_bits))
        tau_key = min(1.0, (self.current_slot - runtime.last_key_refresh_slot) / max(1, self.cfg.qkd.target_refresh_interval_s))
        rho_part = float(np.mean(runtime.recent_participation)) if runtime.recent_participation else 0.0
        reward_ema = float(np.clip(runtime.reward_ema, -1.0, 1.0))
        state = np.array(
            [
                node.energy_j / self.cfg.energy.initial_energy_j,
                float(np.mean(member_energies)),
                float(np.mean([e < 0.25 for e in member_energies])),
                q_norm,
                arr_norm,
                delay_norm,
                float(np.clip(recent_pdr, 0.0, 1.0)),
                hop,
                n_nbr,
                k_pool_norm if self.cfg.federated.use_key_status_features else 1.0,
                tau_key if self.cfg.federated.use_key_status_features else 0.0,
                float(np.clip(runtime.trust_score, 0.0, 1.0)),
                rho_part,
                reward_ema,
            ],
            dtype=np.float32,
        )
        return state

    def _local_view_for_heuristic(self, ch_id: int) -> Dict[str, float]:
        rt = self.cluster_heads[ch_id]
        current = rt.current_next_hop if rt.current_next_hop is not None else self.ch_graph[ch_id][0]
        pdr_best = 0.0
        neighbor_energy_best = 0.0
        hop_best = 1.0
        for nb in self.ch_graph[ch_id]:
            if nb == -1:
                pdr = self._pdr_from_distance(self.ch_to_sink_distance[ch_id])
                energy = 1.0
                hop = 0.0
            else:
                d = self._distance((self.nodes[ch_id].x, self.nodes[ch_id].y), (self.nodes[nb].x, self.nodes[nb].y))
                pdr = self._advertised_pdr(nb, d)
                energy = self.nodes[nb].energy_j / self.cfg.energy.initial_energy_j
                hop = self._hop_estimate(nb)
            pdr_best = max(pdr_best, pdr)
            neighbor_energy_best = max(neighbor_energy_best, energy)
            hop_best = min(hop_best, hop)
        return {
            "energy": self.nodes[ch_id].energy_j / self.cfg.energy.initial_energy_j,
            "queue": len(rt.queue) / max(1, self.cfg.network.queue_capacity_packets),
            "trust": rt.trust_score,
            "pdr_best": pdr_best,
            "neighbor_energy_best": neighbor_energy_best,
            "hop_best": hop_best,
            "hop_current": 0 if current == -1 else self._hop_estimate(current),
        }

    def _choose_actions(self) -> None:
        if self.scheme == "heuristic":
            for ch_id in self.cluster_heads.keys():
                action = heuristic_action(self._local_view_for_heuristic(ch_id))
                self._apply_action(ch_id, action)
            return

        if self.scheme == "centralized_rl":
            gstate = np.array([
                np.mean([self.nodes[c].energy_j / self.cfg.energy.initial_energy_j for c in self.cluster_heads]),
                np.mean([len(rt.queue) / self.cfg.network.queue_capacity_packets for rt in self.cluster_heads.values()]),
                np.mean([rt.trust_score for rt in self.cluster_heads.values()]),
                self.key_engine.key_pool_bits / self.key_engine.cfg.key_pool_capacity_bits if self.key_engine.cfg.key_pool_capacity_bits else 0.0,
            ], dtype=np.float32)
            action = self.centralized_policy.choose_action(gstate)
            for ch_id in self.cluster_heads.keys():
                self._apply_action(ch_id, action)
            return

        if self.scheme == "fl":
            for ch_id in self.cluster_heads.keys():
                action = heuristic_action(self._local_view_for_heuristic(ch_id))
                self._apply_action(ch_id, action)
            return

        for ch_id, runtime in self.cluster_heads.items():
            if not self.nodes[ch_id].alive:
                continue
            state = self._build_local_state(ch_id)
            action = runtime.agent.act(state) if runtime.agent is not None else 0
            self._apply_action(ch_id, action)

    def _apply_action(self, ch_id: int, action: int) -> None:
        rt = self.cluster_heads[ch_id]
        rt.action_id = int(action)
        duty, participate = action_to_tuple(action)
        if action == 0:
            next_hop = rt.current_next_hop if rt.current_next_hop is not None else self._heuristic_next_hop(ch_id)
        elif action == 1:
            next_hop = self._best_pdr_neighbor(ch_id)
        elif action == 2:
            next_hop = self._best_energy_neighbor(ch_id)
        elif action == 3:
            next_hop = self._min_hop_trusted_neighbor(ch_id)
        elif action == 4:
            next_hop = rt.current_next_hop if rt.current_next_hop is not None else self._heuristic_next_hop(ch_id)
            participate = bool(
                self.nodes[ch_id].energy_j / self.cfg.energy.initial_energy_j >= self.cfg.federated.eligibility_energy_threshold
                and self.key_engine.key_pool_bits / self.key_engine.cfg.key_pool_capacity_bits >= self.cfg.federated.eligibility_key_threshold
            )
        elif action == 5:
            next_hop = self._highest_trust_neighbor(ch_id)
            participate = False
        elif action == 6:
            next_hop = self._avoid_low_trust_neighbor(ch_id)
        else:
            next_hop = self._heuristic_next_hop(ch_id, security_aware=True)
            duty = 0.8
            if self.current_slot % 10 == 0:
                self._rotate_cluster_heads()
        rt.current_next_hop = int(next_hop)
        rt.duty_cycle_ratio = float(duty)
        rt.participate_flag = bool(participate)

    def _neighbor_positions(self, ch_id: int, nb: int) -> Tuple[Tuple[float, float], float]:
        pos_i = (self.nodes[ch_id].x, self.nodes[ch_id].y)
        if nb == -1:
            sink_pos = self._sink_position()
            return sink_pos, self._distance(pos_i, sink_pos)
        pos_j = (self.nodes[nb].x, self.nodes[nb].y)
        return pos_j, self._distance(pos_i, pos_j)

    def _best_pdr_neighbor(self, ch_id: int) -> int:
        best, best_pdr = -1, -1.0
        pos_i = (self.nodes[ch_id].x, self.nodes[ch_id].y)
        for nb in self.ch_graph[ch_id]:
            _, d = self._neighbor_positions(ch_id, nb)
            pdr = self._advertised_pdr(nb, d) if nb != -1 else self._pdr_from_distance(d)
            if pdr > best_pdr:
                best_pdr = pdr
                best = nb
        return int(best)

    def _best_energy_neighbor(self, ch_id: int) -> int:
        candidates = []
        for nb in self.ch_graph[ch_id]:
            if nb == -1:
                energy = 1.0
            else:
                energy = self.nodes[nb].energy_j / self.cfg.energy.initial_energy_j
            candidates.append((energy, nb))
        candidates.sort(reverse=True)
        return int(candidates[0][1])

    def _min_hop_trusted_neighbor(self, ch_id: int) -> int:
        best, best_hop = -1, 1e9
        for nb in self.ch_graph[ch_id]:
            if nb == -1:
                return -1
            trust = self.cluster_heads[nb].trust_score
            if trust < 0.55:
                continue
            hop = self._hop_estimate(nb)
            if hop < best_hop:
                best_hop = hop
                best = nb
        return int(best if best != -1 else self._best_pdr_neighbor(ch_id))

    def _highest_trust_neighbor(self, ch_id: int) -> int:
        ranked = []
        for nb in self.ch_graph[ch_id]:
            trust = 1.0 if nb == -1 else self.cluster_heads[nb].trust_score
            ranked.append((trust, nb))
        ranked.sort(reverse=True)
        return int(ranked[0][1])

    def _avoid_low_trust_neighbor(self, ch_id: int) -> int:
        candidates = [nb for nb in self.ch_graph[ch_id] if nb == -1 or self.cluster_heads[nb].trust_score >= 0.5]
        if not candidates:
            return int(self._best_pdr_neighbor(ch_id))
        best, best_score = candidates[0], -1e9
        for nb in candidates:
            _, d = self._neighbor_positions(ch_id, nb)
            pdr = self._pdr_from_distance(d)
            trust = 1.0 if nb == -1 else self.cluster_heads[nb].trust_score
            score = 0.6 * pdr + 0.4 * trust
            if score > best_score:
                best_score = score
                best = nb
        return int(best)

    def _rotate_cluster_heads(self) -> None:
        # Faithful to thesis intent but kept lightweight: reassign members to current CH set.
        for node_id, node in self.nodes.items():
            if node.is_cluster_head:
                continue
            node.cluster_id = self._nearest_cluster_head(node_id)
        self.cluster_members = defaultdict(list)
        for node_id, node in self.nodes.items():
            if not node.is_cluster_head:
                self.cluster_members[node.cluster_id].append(node_id)

    def _sensor_generation_rate(self, ch_id: int) -> int:
        sense_interval = self.cfg.traffic.sense_interval_s
        if self.cfg.enable_hotspots:
            hotspot_fraction = self.cfg.traffic.hotspot_cluster_fraction
            hot_count = max(1, int(round(hotspot_fraction * len(self.cluster_heads))))
            sorted_ch = sorted(self.cluster_heads.keys())
            hotspot_set = set(sorted_ch[:hot_count])
            if ch_id in hotspot_set:
                cycle = self.cfg.traffic.burst_on_slots + self.cfg.traffic.burst_off_slots
                phase = self.current_slot % cycle
                if phase < self.cfg.traffic.burst_on_slots:
                    sense_interval = self.cfg.traffic.hotspot_burst_interval_s
        return sense_interval

    def _sensing_and_intra_cluster(self) -> None:
        bits = self.cfg.traffic.data_packet_bits
        for ch_id, member_ids in self.cluster_members.items():
            rt = self.cluster_heads[ch_id]
            arrivals = 0
            rate = self._sensor_generation_rate(ch_id)
            if self.current_slot % max(1, rate) != 0:
                rt.arrival_ema = 0.95 * rt.arrival_ema
                continue
            for node_id in member_ids:
                node = self.nodes[node_id]
                if not node.alive:
                    continue
                if self.rng.random() > rt.duty_cycle_ratio:
                    continue
                self.generated_packets += 1
                node.energy_j -= self.cfg.energy.sensing_energy_j_per_packet
                ch = self.nodes[ch_id]
                d = self._distance((node.x, node.y), (ch.x, ch.y))
                node.energy_j -= self._energy_tx(bits, d)
                ch.energy_j -= self._energy_rx(bits)
                pdr = self._pdr_from_distance(d)
                if self.rng.random() <= pdr:
                    if len(rt.queue) < self.cfg.network.queue_capacity_packets:
                        rt.queue.append(self.current_slot)
                        arrivals += 1
                    else:
                        self.overflow_drops += 1
                        self.dropped_packets += 1
                else:
                    self.dropped_packets += 1
                self._kill_if_needed(node_id)
                self._kill_if_needed(ch_id)
            rt.arrival_ema = 0.95 * rt.arrival_ema + 0.05 * arrivals

    def _forwarding(self) -> None:
        bits = self.cfg.traffic.data_packet_bits
        for ch_id, rt in self.cluster_heads.items():
            if not self.nodes[ch_id].alive:
                continue
            service = min(len(rt.queue), self.cfg.network.service_rate_packets_per_slot)
            next_hop = rt.current_next_hop if rt.current_next_hop is not None else self._heuristic_next_hop(ch_id)
            for _ in range(service):
                if not rt.queue:
                    break
                created_at = rt.queue.popleft()
                _, d = self._neighbor_positions(ch_id, next_hop)
                tx_energy = self._energy_tx(bits, d)
                self.nodes[ch_id].energy_j -= tx_energy
                pdr = self._pdr_from_distance(d)
                if next_hop != -1 and self.attack_state and self.attack_state.is_forwarder_compromised(next_hop):
                    if self.rng.random() < self.cfg.attacks.selective_forwarding_drop_probability:
                        pdr = 0.0
                success = self.rng.random() <= pdr
                rt.recent_tx_success.append(1 if success else 0)
                if success:
                    if next_hop == -1:
                        self.delivered_packets += 1
                        rt.delivered_packets += 1
                        self.packet_delays.append(self.current_slot - created_at + 1)
                    else:
                        self.nodes[next_hop].energy_j -= self._energy_rx(bits)
                        if len(self.cluster_heads[next_hop].queue) < self.cfg.network.queue_capacity_packets:
                            self.cluster_heads[next_hop].queue.append(created_at)
                        else:
                            self.overflow_drops += 1
                            self.dropped_packets += 1
                        # Trust update on actual success
                        self.cluster_heads[next_hop].trust_score = min(1.0, self.cluster_heads[next_hop].trust_score + 0.003)
                else:
                    self.dropped_packets += 1
                    if next_hop != -1:
                        self.cluster_heads[next_hop].trust_score = max(0.0, self.cluster_heads[next_hop].trust_score - 0.05)
                self._kill_if_needed(ch_id)
                if next_hop != -1:
                    self._kill_if_needed(next_hop)

    def _control_messages(self) -> int:
        count = 0
        for _ in self.cluster_heads:
            if self.rng.random() < self.cfg.traffic.control_message_probability_per_slot:
                count += 1
        self.control_messages += count
        return count

    def _reward_for(self, ch_id: int, before_energy: float, delivered_increment: int, queue_before: int) -> float:
        rt = self.cluster_heads[ch_id]
        node = self.nodes[ch_id]
        delta_e = max(0.0, before_energy - node.energy_j) / max(1e-6, self.cfg.energy.initial_energy_j)
        f_e = -delta_e
        local_delay = (np.mean([self.current_slot - t for t in rt.queue]) if rt.queue else 0.0)
        f_q = float(np.clip(0.5 * delivered_increment + 0.5 * (1.0 - min(1.0, local_delay / 20.0)), 0.0, 1.0))
        key_norm = self.key_engine.key_pool_bits / max(1.0, self.key_engine.cfg.key_pool_capacity_bits)
        f_s = 0.5 * float(rt.trust_score) + 0.5 * float(np.clip(key_norm, 0.0, 1.0))
        penalty = 0.0
        penalty += 0.5 if len(rt.queue) > queue_before else 0.0
        penalty += 1.0 if rt.trust_score < 0.5 else 0.0
        penalty += 0.5 if key_norm < self.cfg.federated.eligibility_key_threshold else 0.0
        rw = (
            self.cfg.rl.reward.alpha_energy * f_e
            + self.cfg.rl.reward.alpha_qos * f_q
            + self.cfg.rl.reward.alpha_security * f_s
            - self.cfg.rl.reward.alpha_penalty * penalty
        )
        rt.reward_ema = 0.95 * rt.reward_ema + 0.05 * rw
        return float(rw)

    def _train_local_agents(self) -> None:
        if self.scheme not in {"frl_qkd", "frl_noqkd"}:
            return
        for ch_id, rt in self.cluster_heads.items():
            if rt.agent is None or not self.nodes[ch_id].alive:
                continue
            if self.current_slot % self.cfg.rl.local_training_frequency == 0:
                rt.agent.train_step()

    def _federated_round(self) -> None:
        if self.scheme not in {"frl_qkd", "frl_noqkd"}:
            return
        if self.global_model is None:
            return
        eligible = []
        key_norm = self.key_engine.key_pool_bits / max(1.0, self.key_engine.cfg.key_pool_capacity_bits)
        for ch_id, rt in self.cluster_heads.items():
            if not self.nodes[ch_id].alive or rt.agent is None:
                continue
            energy_norm = self.nodes[ch_id].energy_j / self.cfg.energy.initial_energy_j
            if energy_norm < self.cfg.federated.eligibility_energy_threshold:
                continue
            if self.cfg.federated.use_key_status_features and key_norm < self.cfg.federated.eligibility_key_threshold:
                continue
            if rt.trust_score < self.cfg.federated.eligibility_trust_threshold:
                continue
            if not rt.participate_flag:
                continue
            eligible.append(ch_id)

        target_count = max(1, int(math.ceil(self.cfg.federated.client_fraction * max(1, len(eligible)))))
        if len(eligible) > target_count:
            self.rng.shuffle(eligible)
            selected = eligible[:target_count]
        else:
            selected = eligible

        control_messages = self._control_messages()
        used_bits = self.key_engine.consume(
            frl_participants=len(selected),
            control_messages=control_messages,
            secure=(self.scheme == "frl_qkd"),
        )
        secure_fraction_now = self.key_engine.secure_fraction()

        deltas = []
        weights = {}
        for ch_id, rt in self.cluster_heads.items():
            rt.recent_participation.append(1 if ch_id in selected else 0)

        for ch_id in selected:
            rt = self.cluster_heads[ch_id]
            rt.agent.apply_global(self.global_model)
            stats = rt.agent.local_train(self.cfg.rl.local_steps_per_round)
            rt.last_round_loss = stats["last_loss"]
            delta = rt.agent.get_delta(self.global_model)
            if self.attack_state and self.attack_state.is_free_rider(ch_id):
                delta = {k: torch.zeros_like(v) if torch.is_floating_point(v) else v for k, v in delta.items()}
            if self.attack_state and self.attack_state.is_poisoned_client(ch_id):
                delta = poison_update(delta, scale=self.cfg.attacks.model_poisoning_scale)
            anomaly = detect_update_anomaly(delta, self.reference_update_scale)
            if anomaly:
                rt.trust_score = max(0.05, rt.trust_score - 0.10)
            else:
                rt.trust_score = min(1.0, rt.trust_score + 0.02)
            deltas.append((ch_id, delta))
            weights[ch_id] = max(1.0, len(self.cluster_members.get(ch_id, [])))
        if deltas:
            norms = []
            for _, delta in deltas:
                sq = 0.0
                for v in delta.values():
                    if torch.is_floating_point(v):
                        sq += float(torch.sum(v * v).item())
                norms.append(math.sqrt(sq))
            self.reference_update_scale = float(np.mean(norms)) if norms else self.reference_update_scale

        new_global = aggregate_deltas(
            global_state=self.global_model,
            deltas=deltas,
            trust_scores={cid: self.cluster_heads[cid].trust_score for cid in self.cluster_heads},
            weights=weights,
            clip_norm=self.cfg.federated.norm_clip_value,
            use_trust=self.cfg.federated.use_trust_weights,
        )
        self.global_model = new_global
        for ch_id, rt in self.cluster_heads.items():
            if rt.agent is not None:
                rt.agent.apply_global(self.global_model)

        # payload accounting
        uplink_bytes = int(self.model_size_params * (self.cfg.federated.model_quantization_bits / 8.0) * self.cfg.federated.model_compression_ratio)
        downlink_bytes = uplink_bytes
        self.metrics["uplink_bytes_round"].append(uplink_bytes * max(1, len(selected)))
        self.metrics["downlink_bytes_round"].append(downlink_bytes * len(self.cluster_heads))
        self.metrics["qkd_bits_round"].append(used_bits if self.scheme == "frl_qkd" else 0)
        self.metrics["client_participation_rate"].append(len(selected) / max(1, len(eligible) if eligible else len(self.cluster_heads)))
        self.metrics["secure_fraction"].append(secure_fraction_now)
        self.metrics["global_reward_moving_avg"].append(float(np.mean([rt.reward_ema for rt in self.cluster_heads.values()])))

    def _update_qkd(self) -> None:
        qber = self.cfg.attacks.elevated_qber if self.cfg.attacks.elevated_qber is not None else self.cfg.qkd.nominal_qber
        self.key_engine.replenish(self.cfg.traffic.slot_duration_s, self.current_slot, qber_override=qber if self.scheme == "frl_qkd" else None)

    def step(self) -> None:
        before_energy = {cid: self.nodes[cid].energy_j for cid in self.cluster_heads}
        queue_before = {cid: len(rt.queue) for cid, rt in self.cluster_heads.items()}
        delivered_before = self.delivered_packets

        self._choose_actions()
        self._sensing_and_intra_cluster()
        self._forwarding()
        self._train_local_agents()
        if self.current_slot > 0 and self.current_slot % self.cfg.frl_interval_slots == 0:
            self._federated_round()
        self._update_qkd()

        delivered_inc_total = self.delivered_packets - delivered_before
        for ch_id, rt in self.cluster_heads.items():
            state = self._build_local_state(ch_id)
            reward = self._reward_for(ch_id, before_energy[ch_id], max(0, delivered_inc_total // max(1, len(self.cluster_heads))), queue_before[ch_id])
            if self.scheme in {"frl_qkd", "frl_noqkd"} and rt.agent is not None:
                rt.agent.remember(state, rt.action_id, reward, self._build_local_state(ch_id), 0.0)
        alive_frac = self._alive_fraction()
        dead_count = self._dead_counts()
        if self.l50_slot is None and alive_frac <= 0.5:
            self.l50_slot = self.current_slot
        if self.l100_slot is None and dead_count == len(self.nodes):
            self.l100_slot = self.current_slot

        self.metrics["avg_residual_energy"].append(np.mean([n.energy_j for n in self.nodes.values() if n.alive]) if any(n.alive for n in self.nodes.values()) else 0.0)
        self.metrics["key_pool_bits"].append(self.key_engine.key_pool_bits)
        self.metrics["queue_occupancy"].append(np.mean([len(rt.queue) for rt in self.cluster_heads.values()]))
        self.metrics["alive_fraction"].append(alive_frac)
        self.current_slot += 1

    def run(self) -> Dict[str, float]:
        while self.current_slot < self.cfg.total_slots and (self.l100_slot is None):
            self.step()

        pdr = self.delivered_packets / self.generated_packets if self.generated_packets else 0.0
        latency = float(np.mean(self.packet_delays)) if self.packet_delays else float(self.cfg.total_slots)
        ttfnd = float(self.first_death_slot if self.first_death_slot is not None else self.cfg.total_slots)
        l50 = float(self.l50_slot if self.l50_slot is not None else self.cfg.total_slots)
        l100 = float(self.l100_slot if self.l100_slot is not None else self.cfg.total_slots)
        secure_fraction = self.key_engine.secure_fraction() if self.scheme == "frl_qkd" else 0.0
        leakage = self._compute_leakage_proxy(secure_fraction)
        result = {
            "scenario": self.cfg.name,
            "scheme": self.scheme,
            "seed": self.seed,
            "ttfnd": ttfnd,
            "l50": l50,
            "l100": l100,
            "pdr": pdr,
            "latency": latency,
            "avg_residual_energy_final": float(self.metrics["avg_residual_energy"][-1]) if self.metrics["avg_residual_energy"] else 0.0,
            "generated_packets": float(self.generated_packets),
            "delivered_packets": float(self.delivered_packets),
            "dropped_packets": float(self.dropped_packets),
            "overflow_drops": float(self.overflow_drops),
            "avg_queue_occupancy": float(np.mean(self.metrics["queue_occupancy"])) if self.metrics["queue_occupancy"] else 0.0,
            "global_reward_ma_final": float(self.metrics["global_reward_moving_avg"][-1]) if self.metrics["global_reward_moving_avg"] else 0.0,
            "client_participation_rate_mean": float(np.mean(self.metrics["client_participation_rate"])) if self.metrics["client_participation_rate"] else 0.0,
            "avg_key_pool_bits": float(np.mean(self.metrics["key_pool_bits"])) if self.metrics["key_pool_bits"] else 0.0,
            "secure_fraction": float(secure_fraction),
            "leakage_proxy": float(leakage),
            "uplink_bytes_round_mean": float(np.mean(self.metrics["uplink_bytes_round"])) if self.metrics["uplink_bytes_round"] else 0.0,
            "downlink_bytes_round_mean": float(np.mean(self.metrics["downlink_bytes_round"])) if self.metrics["downlink_bytes_round"] else 0.0,
            "qkd_bits_round_mean": float(np.mean(self.metrics["qkd_bits_round"])) if self.metrics["qkd_bits_round"] else 0.0,
            "model_size_params": float(self.model_size_params),
        }
        return result

    def _compute_leakage_proxy(self, secure_fraction: float) -> float:
        # Normalized proxy in [0,1], lower is better.
        classical_frac = 1.0 - secure_fraction
        qber = self.cfg.attacks.elevated_qber if self.cfg.attacks.elevated_qber is not None else self.cfg.qkd.nominal_qber
        qkd_penalty = 0.0 if self.scheme == "frl_qkd" else 0.6
        scarcity_penalty = 0.2 if (self.scheme == "frl_qkd" and secure_fraction < self.cfg.qkd.min_secure_fraction) else 0.0
        compromise_penalty = self.cfg.attacks.classical_key_compromise_fraction if self.scheme != "frl_qkd" else 0.0
        qber_penalty = max(0.0, (qber - 0.02) / 0.09) * 0.15 if self.scheme == "frl_qkd" else 0.0
        leakage = classical_frac * 0.5 + qkd_penalty + scarcity_penalty + compromise_penalty * 0.5 + qber_penalty
        return float(np.clip(leakage, 0.0, 1.0))
