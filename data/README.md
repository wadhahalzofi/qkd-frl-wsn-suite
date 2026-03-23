# Data

## Data Sources and Generation

This directory holds data files used in the experiments. Currently, all experimental data is **synthetically generated** during simulation runs.

### Synthetic Data

- **Network topologies**: Generated programmatically in `qkd_frl_wsn/simulator.py` based on scenario configurations in `configs/`
- **Energy profiles**: Synthetic, based on realistic WSN parameters defined in configuration files
- **Attack traces**: Generated on-the-fly during simulation based on attack scenarios (S3–S6)
- **Training traces**: Generated during federated learning with DQN agents

No external datasets are included; all data is produced by `run_suite.py` and written to the `results/` directory.

### Scenario Configuration

Scenario parameters are defined in `configs/scenario_S*.yaml` files:
- **S1_nominal**: Standard baseline scenario
- **S2_high_traffic**: High network traffic conditions
- **S3_network_attacks**: Network-layer adversarial attacks
- **S4_model_poisoning**: Federated learning poisoning
- **S5_qkd_scarcity**: QKD key rate constraints
- **S6_combined**: Combined attacks and resource constraints

### Reproducibility

To regenerate all experimental data:

```bash
python run_suite.py --all-scenarios --config-dir configs --seeds 0 1 2
```

Results are written to `results/` with timestamped subdirectories.

### References

See `docs/ASSUMPTIONS.md` for parameter justifications and design choices.
