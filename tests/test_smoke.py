import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathlib import Path

from qkd_frl_wsn.config import load_config
from qkd_frl_wsn.runner import run_experiment


def test_smoke_run():
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "scenario_S1_nominal.yaml")
    cfg.total_slots = 120
    cfg.n_seeds_default = 1
    df = run_experiment(cfg, schemes=["heuristic", "frl_qkd"], seeds=[0], device="cpu")
    assert not df.empty
    assert set(df["scheme"]) == {"heuristic", "frl_qkd"}
