from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from .config import ScenarioConfig, as_dict, load_config
from .simulator import ExperimentSimulator


DEFAULT_SCHEME_ORDER = ["heuristic", "centralized_rl", "fl", "frl_noqkd", "frl_qkd"]


def normalize_against_heuristic(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for scenario in out["scenario"].unique():
        sub = out[out["scenario"] == scenario]
        heur = sub[sub["scheme"] == "heuristic"]
        if heur.empty:
            continue
        ttfnd_ref = float(heur["ttfnd"].mean())
        l50_ref = float(heur["l50"].mean())
        lat_ref = float(heur["latency"].mean())
        out.loc[out["scenario"] == scenario, "ttfnd_norm"] = out.loc[out["scenario"] == scenario, "ttfnd"] / max(1e-9, ttfnd_ref)
        out.loc[out["scenario"] == scenario, "l50_norm"] = out.loc[out["scenario"] == scenario, "l50"] / max(1e-9, l50_ref)
        out.loc[out["scenario"] == scenario, "latency_norm"] = out.loc[out["scenario"] == scenario, "latency"] / max(1e-9, lat_ref)
    return out


def make_scheme_config(base_cfg: ScenarioConfig, scheme: str) -> ScenarioConfig:
    cfg = copy.deepcopy(base_cfg)
    if scheme == "frl_noqkd":
        cfg.qkd.enabled = False
    if scheme in {"heuristic", "centralized_rl", "fl"}:
        cfg.qkd.enabled = False
    if scheme == "centralized_rl":
        cfg.federated.global_rounds = 0
    return cfg


def run_single(cfg: ScenarioConfig, scheme: str, seed: int, device: str = "cpu") -> Dict[str, float]:
    sim = ExperimentSimulator(cfg=make_scheme_config(cfg, scheme), scheme=scheme, seed=seed, device=device)
    return sim.run()


def run_experiment(cfg: ScenarioConfig, schemes: Iterable[str], seeds: Iterable[int], device: str = "cpu") -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    for seed in seeds:
        for scheme in schemes:
            records.append(run_single(cfg, scheme=scheme, seed=int(seed), device=device))
    df = pd.DataFrame.from_records(records)
    return normalize_against_heuristic(df)


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in df.columns if c not in {"scenario", "scheme", "seed"}]
    agg = df.groupby(["scenario", "scheme"], as_index=False)[metric_cols].agg(["mean", "std"])
    agg.columns = ["_".join([c for c in col if c]).strip("_") for col in agg.columns.to_flat_index()]
    agg = agg.rename(columns={"scenario_": "scenario", "scheme_": "scheme"})
    return agg


def _load_all_configs(config_dir: Path) -> List[Path]:
    return sorted(config_dir.glob("scenario_*.yaml"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reconstructed QKD-FRL-WSN experiments.")
    parser.add_argument("--config", type=str, default=None, help="Path to one scenario YAML.")
    parser.add_argument("--config-dir", type=str, default="configs", help="Directory with scenario YAML files.")
    parser.add_argument("--schemes", type=str, nargs="*", default=None, help="Schemes to run.")
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="Seeds to run.")
    parser.add_argument("--device", type=str, default="cpu", help="PyTorch device.")
    parser.add_argument("--outdir", type=str, default="results", help="Output directory.")
    parser.add_argument("--all-scenarios", action="store_true", help="Run every scenario config in --config-dir.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.all_scenarios:
        cfg_paths = _load_all_configs(Path(args.config_dir))
    elif args.config:
        cfg_paths = [Path(args.config)]
    else:
        raise SystemExit("Provide --config or --all-scenarios.")

    all_frames = []
    for cfg_path in cfg_paths:
        cfg = load_config(cfg_path)
        schemes = args.schemes or cfg.schemes
        seeds = args.seeds or list(range(cfg.n_seeds_default))
        df = run_experiment(cfg, schemes=schemes, seeds=seeds, device=args.device)
        summary = aggregate_results(df)
        raw_path = outdir / f"{cfg.name}_raw.csv"
        sum_path = outdir / f"{cfg.name}_summary.csv"
        cfg_json = outdir / f"{cfg.name}_resolved_config.json"
        df.to_csv(raw_path, index=False)
        summary.to_csv(sum_path, index=False)
        cfg_json.write_text(json.dumps(as_dict(cfg), indent=2), encoding="utf-8")
        all_frames.append(df)

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined.to_csv(outdir / "combined_raw.csv", index=False)
        aggregate_results(combined).to_csv(outdir / "combined_summary.csv", index=False)


if __name__ == "__main__":
    main()
