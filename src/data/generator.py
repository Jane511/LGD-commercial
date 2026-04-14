"""
Generate synthetic historical workout data for all 11 LGD product modules.

Writes Parquet files to data/generated/historical/.
Runs as an idempotent script: re-running with the same seed overwrites files
with identical output (deterministic generation).

Usage:
    python -m src.data.generator
    python -m src.data.generator --seed 42
    python -m src.data.generator --module mortgage
    python -m src.data.generator --products mortgage development_finance

Output:
    data/generated/historical/
        mortgage_workouts.parquet
        commercial_cashflow_workouts.parquet
        receivables_workouts.parquet
        trade_contingent_workouts.parquet
        asset_equipment_workouts.parquet
        development_finance_workouts.parquet
        cre_investment_workouts.parquet
        residual_stock_workouts.parquet
        land_subdivision_workouts.parquet
        bridging_workouts.parquet
        mezz_second_mortgage_workouts.parquet
        (+ corresponding _cashflows.parquet files)

APS 113 Reference: All generated data is for calibration demonstration only.
SYNTHETIC HISTORICAL CALIBRATION DATA — FOR DEMONSTRATION ONLY.
"""
import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent

from src.generators import generate_all_historical_workouts, GENERATOR_MAP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("data.generator")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic historical workout data for LGD calibration."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible generation (default: 42).",
    )
    parser.add_argument(
        "--module", "--product", dest="module", type=str, default=None,
        help="Run a single product module in isolation (e.g., --module mortgage).",
    )
    parser.add_argument(
        "--products", nargs="+", default=None,
        help="List of product names to generate (default: all 11).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for Parquet files (default: data/generated/historical/).",
    )
    parser.add_argument(
        "--no-parquet", action="store_true",
        help="Do not write Parquet files (dry-run mode, for testing).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_ts = datetime.now()

    logger.info("=" * 65)
    logger.info("LGD Historical Workout Data Generator")
    logger.info("SYNTHETIC DATA — FOR DEMONSTRATION ONLY")
    logger.info("=" * 65)
    logger.info("Seed: %d | Start: %s", args.seed, start_ts.strftime("%Y-%m-%d %H:%M:%S"))

    if args.module:
        products = [args.module]
        logger.info("Single-module mode: %s", args.module)
    elif args.products:
        products = args.products
        logger.info("Selected products: %s", products)
    else:
        products = list(GENERATOR_MAP.keys())
        logger.info("Generating all %d products.", len(products))

    unknown = [p for p in products if p not in GENERATOR_MAP]
    if unknown:
        logger.error("Unknown product(s): %s. Valid: %s", unknown, list(GENERATOR_MAP.keys()))
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "data" / "generated" / "historical"

    t0 = time.time()
    results = generate_all_historical_workouts(
        seed=args.seed,
        output_dir=output_dir,
        write_parquet=not args.no_parquet,
        products=products,
    )
    elapsed = time.time() - t0

    logger.info("")
    logger.info("Generation complete (%.1fs):", elapsed)
    total_loans = 0
    total_cashflows = 0
    for product, data in results.items():
        n_loans = len(data["loans"])
        n_cf = len(data["cashflows"])
        total_loans += n_loans
        total_cashflows += n_cf
        years_covered = data["loans"]["default_year"].nunique()
        downturn_pct = 100 * data["loans"]["downturn_flag"].mean()
        logger.info(
            "  %-25s %5d loans | %6d cashflow rows | %d years | %.0f%% downturn",
            product, n_loans, n_cf, years_covered, downturn_pct,
        )

    logger.info("")
    logger.info("Total: %d loans | %d cashflow rows", total_loans, total_cashflows)
    if not args.no_parquet:
        logger.info("Output dir: %s", output_dir)
        n_files = len(list(output_dir.glob("*.parquet")))
        logger.info("Parquet files written: %d", n_files)

    logger.info("=" * 65)


if __name__ == "__main__":
    main()
