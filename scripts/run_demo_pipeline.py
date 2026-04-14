"""
Demo pipeline runner.

Usage:
    python scripts/run_demo_pipeline.py                  # proxy demo only (default)
    python scripts/run_demo_pipeline.py --with-calibration  # proxy demo + APS 113 calibration
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

from src.reproducibility import set_global_seed
from src.demo_run_pipeline import main


def _run_calibration_pipeline() -> None:
    """Run APS 113 calibration pipeline for all products (calls run_calibration_pipeline.py)."""
    import subprocess
    import sys as _sys

    print("\n" + "=" * 60)
    print("Running APS 113 calibration pipeline (all products)...")
    print("=" * 60)
    result = subprocess.run(
        [_sys.executable, str(ROOT / "scripts" / "run_calibration_pipeline.py"),
         "--products", "all"],
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        print("WARNING: Calibration pipeline exited with non-zero status.")
    else:
        print("Calibration pipeline completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LGD demo pipeline with optional APS 113 calibration layer."
    )
    parser.add_argument(
        "--with-calibration",
        action="store_true",
        default=False,
        help=(
            "After the proxy demo pipeline, run the full APS 113 calibration "
            "pipeline (all 11 product modules). Default: disabled."
        ),
    )
    args = parser.parse_args()

    set_global_seed(42)
    main()

    if args.with_calibration:
        _run_calibration_pipeline()
