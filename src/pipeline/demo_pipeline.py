"""
Demo pipeline and source-aware pipeline runner.

Usage:
    python -m src.pipeline.demo_pipeline                    # proxy demo (generated data)
    python -m src.pipeline.demo_pipeline --with-calibration # proxy demo + APS 113 calibration
    python -m src.pipeline.demo_pipeline --source controlled --controlled-root data/controlled
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True

from src.demo_pipeline import main as _demo_main
from src.reproducibility import set_global_seed
from src.data_source_adapter import load_datasets
from src.lgd_calculation import run_full_pipeline


def _run_calibration_pipeline() -> None:
    """Run APS 113 calibration pipeline for all products."""
    import subprocess
    import sys as _sys

    print("\n" + "=" * 60)
    print("Running APS 113 calibration pipeline (all products)...")
    print("=" * 60)
    result = subprocess.run(
        [_sys.executable, "-m", "src.pipeline.calibration_pipeline", "--products", "all"],
        cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        print("WARNING: Calibration pipeline exited with non-zero status.")
    else:
        print("Calibration pipeline completed successfully.")


def _write_core_outputs(results: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for product in ["mortgage", "commercial", "development", "cashflow_lending"]:
        if product not in results:
            continue
        block = results[product]
        if "loans_with_overlays" in block:
            block["loans_with_overlays"].to_csv(out_dir / f"{product}_loan_level_output.csv", index=False)
        if "segment_summary" in block:
            block["segment_summary"].to_csv(out_dir / f"{product}_segment_summary.csv", index=False)
        if "weighted_output" in block:
            block["weighted_output"].to_csv(out_dir / f"{product}_weighted_output.csv", index=False)
    if "reporting_tables" in results:
        for name, df in results["reporting_tables"].items():
            if isinstance(df, pd.DataFrame):
                df.to_csv(out_dir / name, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LGD demo pipeline with optional calibration or source-adapter selection."
    )
    parser.add_argument(
        "--with-calibration",
        action="store_true",
        default=False,
        help="After the proxy demo pipeline, run the full APS 113 calibration pipeline.",
    )
    parser.add_argument(
        "--source",
        choices=["generated", "controlled"],
        default=None,
        help="Use source adapter instead of demo mode. Omit for standard demo.",
    )
    parser.add_argument(
        "--controlled-root",
        default="data/controlled",
        help="Folder with controlled input tables (used when --source controlled).",
    )
    parser.add_argument("--scenario-id", default="baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-reporting", action="store_true")
    parser.add_argument("--out-dir", default="outputs/tables")
    args = parser.parse_args()

    set_global_seed(args.seed)

    if args.source is not None:
        datasets = load_datasets(
            source=args.source,
            controlled_root=args.controlled_root,
            require_all_products=True,
        )
        results = run_full_pipeline(
            datasets,
            include_reporting=args.include_reporting,
            scenario_id=args.scenario_id,
            seed=args.seed,
        )
        _write_core_outputs(results, Path(args.out_dir))
        print(f"source={args.source}")
        print(f"out_dir={Path(args.out_dir).resolve()}")
    else:
        _demo_main()

    if args.with_calibration:
        _run_calibration_pipeline()


if __name__ == "__main__":
    main()
