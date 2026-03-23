# QKD-assisted FRL for WSNs — full experimental suite

The complete experimental environment described in the thesis:

> *Secure Energy-efficient Federated Reinforcement Learning for Wireless Sensor Networks via Quantum Key Distribution*

It includes:

- a modular Python simulator for clustered WSNs;
- DQN-based local agents and federated aggregation;
- a BB84-style parametric QKD key-rate / key-pool layer;
- attack injection for network, learning, and QKD-layer adversaries;
- baseline schemes (`heuristic`, `centralized_rl`, `fl`, `frl_noqkd`, `frl_qkd`);
- ready-to-run scenario configuration files for **S1–S6**;
- reproducibility and assumption notes.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run_suite.py --config configs/scenario_S1_nominal.yaml --schemes heuristic frl_noqkd frl_qkd --seeds 0 1
python run_suite.py --all-scenarios --config-dir configs --seeds 0
```

Outputs are written to `results/`.

## Scenario files

- `configs/scenario_S1_nominal.yaml`
- `configs/scenario_S2_high_traffic.yaml`
- `configs/scenario_S3_network_attacks.yaml`
- `configs/scenario_S4_model_poisoning.yaml`
- `configs/scenario_S5_qkd_scarcity.yaml`
- `configs/scenario_S6_combined.yaml`

## Package layout

- `qkd_frl_wsn/config.py` — dataclasses + YAML config loading
- `qkd_frl_wsn/models.py` — DQN definition
- `qkd_frl_wsn/agents.py` — replay buffer + DQN agent
- `qkd_frl_wsn/qkd.py` — BB84-style parametric key-rate and key-pool logic
- `qkd_frl_wsn/attacks.py` — attack state and poisoning helper
- `qkd_frl_wsn/baselines.py` — heuristic and centralized policy helpers
- `qkd_frl_wsn/federated.py` — trust-weighted FedAvg aggregation
- `qkd_frl_wsn/simulator.py` — integrated WSN/FRL/QKD simulator
- `qkd_frl_wsn/runner.py` — experiment orchestration and CSV export

## Output metrics

The suite exports raw and aggregated CSV tables for:
- TTFND, L50, L100
- PDR
- latency
- average residual energy
- queue occupancy
- client participation rate
- key-pool occupancy
- secure message fraction
- leakage proxy
- control-plane overhead and QKD bits consumed
