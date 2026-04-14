"""
Classify economic regimes across the 2014-2024 observation window.

Reads the upstream macro_regime_flags.parquet from the industry-analysis repo
(real RBA/ABS data). Falls back to built-in synthetic macro series when the
upstream file is not available.

Outputs:
    data/generated/historical/economic_regimes.parquet
    outputs/tables/economic_regime_classification.csv

APS 113 References:
  - s.43: Long-run LGD must span a full economic cycle including a downturn
  - s.46-50: Downturn periods must be formally identified for downturn LGD calibration

Usage:
    python scripts/classify_economic_regimes.py
    python scripts/classify_economic_regimes.py --upstream-path ../industry-analysis/data/exports/macro_regime_flags.parquet
    python scripts/classify_economic_regimes.py --synthetic   # force synthetic mode
"""
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.regime_classifier import (
    classify_economic_regime,
    export_regime_classification,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("classify_economic_regimes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify economic regimes for LGD downturn calibration."
    )
    parser.add_argument(
        "--upstream-path", type=str, default=None,
        help="Path to macro_regime_flags.parquet from industry-analysis repo.",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Force synthetic mode (ignore upstream parquet).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: data/generated/historical/).",
    )
    parser.add_argument(
        "--start-year", type=int, default=2014,
        help="First year to classify (default: 2014).",
    )
    parser.add_argument(
        "--end-year", type=int, default=2024,
        help="Last year to classify (default: 2024).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("=" * 65)
    logger.info("Economic Regime Classifier")
    logger.info("APS 113 s.43, s.46-50 — Downturn Period Identification")
    logger.info("=" * 65)

    method = "synthetic" if args.synthetic else "upstream_first"
    logger.info("Method: %s", method)

    regimes = classify_economic_regime(
        upstream_parquet_path=args.upstream_path,
        method=method,
    )

    # Filter to requested year range
    regimes = regimes[
        (regimes["year"] >= args.start_year) & (regimes["year"] <= args.end_year)
    ].copy()

    # Summary
    n_downturn = regimes["is_downturn_period"].sum()
    data_source = regimes["data_source"].iloc[0] if len(regimes) > 0 else "unknown"

    logger.info("")
    logger.info("Regime classification results (%d--%d):", args.start_year, args.end_year)
    logger.info("  Data source: %s", data_source)
    logger.info("  Total years classified: %d", len(regimes))
    logger.info("  Downturn years: %d", n_downturn)
    logger.info("")

    for _, row in regimes.iterrows():
        flag = " *** DOWNTURN ***" if row["is_downturn_period"] else ""
        logger.info("  %d: %-15s%s", int(row["year"]), row["regime"], flag)

    logger.info("")

    if n_downturn == 0:
        logger.error(
            "CRITICAL: No downturn years identified in %d--%d. "
            "APS 113 s.43 requires at least one downturn period. "
            "Check upstream macro data or extend observation window.",
            args.start_year, args.end_year,
        )
        sys.exit(1)

    # Write outputs
    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "data" / "generated" / "historical"
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / "economic_regimes.parquet"
    regimes.to_parquet(parquet_path, index=False)
    logger.info("Written: %s", parquet_path)

    csv_path = REPO_ROOT / "outputs" / "tables" / "economic_regime_classification.csv"
    export_regime_classification(regimes, csv_path)
    logger.info("Written: %s", csv_path)
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
