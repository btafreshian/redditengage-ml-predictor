import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_latest_run


def run_cli(*args: str) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "src.cli", *args]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return result


def test_end_to_end_smoke(tmp_path):
    overrides = ["--override", "data.synthetic_rows=200", "--override", "model_baseline.vectorizer.max_features=1000"]
    run_cli("prepare", *overrides)
    run_cli("train-baseline", *overrides)
    run_cli("evaluate")

    latest = load_latest_run()
    metrics = latest.get("evaluation_metrics") or latest.get("metrics")
    assert metrics
    for key in ["rmse", "mae", "median_ae", "r2"]:
        assert key in metrics

    figures_dir = Path("reports/figures")
    assert (figures_dir / "pred_vs_actual.png").exists()
    assert (figures_dir / "error_distribution.png").exists()
