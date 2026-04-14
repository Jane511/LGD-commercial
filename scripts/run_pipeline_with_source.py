from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_source_adapter import load_datasets
from src.lgd_calculation import run_full_pipeline


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
    parser = argparse.ArgumentParser(description="Run LGD pipeline with generated or controlled inputs.")
    parser.add_argument("--source", choices=["generated", "controlled"], default="generated")
    parser.add_argument("--controlled-root", default="data/controlled", help="Folder with controlled input tables.")
    parser.add_argument("--scenario-id", default="baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-reporting", action="store_true")
    parser.add_argument("--out-dir", default="outputs/tables")
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
