"""
Synthetic historical workout data generators for all 11 product types.

All generators produce 10-year (2014-2024) synthetic workout histories
suitable for APS 113-aligned LGD calibration. Data is labelled
SYNTHETIC HISTORICAL CALIBRATION DATA — FOR DEMONSTRATION ONLY.
"""
import logging
from pathlib import Path

import pandas as pd

from .mortgage_generator import MortgageWorkoutGenerator
from .commercial_cashflow_generator import CommercialCashflowWorkoutGenerator
from .receivables_generator import ReceivablesWorkoutGenerator
from .trade_contingent_generator import TradeContingentWorkoutGenerator
from .asset_equipment_generator import AssetEquipmentWorkoutGenerator
from .development_finance_generator import DevelopmentFinanceWorkoutGenerator
from .cre_investment_generator import CREInvestmentWorkoutGenerator
from .residual_stock_generator import ResidualStockWorkoutGenerator
from .land_subdivision_generator import LandSubdivisionWorkoutGenerator
from .bridging_generator import BridgingWorkoutGenerator
from .mezz_second_mortgage_generator import MezzSecondMortgageWorkoutGenerator

logger = logging.getLogger(__name__)

GENERATOR_MAP = {
    "mortgage":              MortgageWorkoutGenerator,
    "commercial_cashflow":   CommercialCashflowWorkoutGenerator,
    "receivables":           ReceivablesWorkoutGenerator,
    "trade_contingent":      TradeContingentWorkoutGenerator,
    "asset_equipment":       AssetEquipmentWorkoutGenerator,
    "development_finance":   DevelopmentFinanceWorkoutGenerator,
    "cre_investment":        CREInvestmentWorkoutGenerator,
    "residual_stock":        ResidualStockWorkoutGenerator,
    "land_subdivision":      LandSubdivisionWorkoutGenerator,
    "bridging":              BridgingWorkoutGenerator,
    "mezz_second_mortgage":  MezzSecondMortgageWorkoutGenerator,
}


def generate_all_historical_workouts(
    seed: int = 42,
    output_dir: Path | None = None,
    write_parquet: bool = True,
    products: list[str] | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Generate synthetic historical workout data for all (or specified) products.

    Parameters
    ----------
    seed : random seed for reproducibility
    output_dir : directory for Parquet output. Defaults to data/generated/historical/.
    write_parquet : if True, write .parquet files to output_dir
    products : list of product names to generate; None = all 11

    Returns
    -------
    dict mapping product_name -> {'loans': DataFrame, 'cashflows': DataFrame}
    """
    products = products or list(GENERATOR_MAP.keys())
    results = {}

    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "data" / "generated" / "historical"

    if write_parquet:
        output_dir.mkdir(parents=True, exist_ok=True)

    for product in products:
        if product not in GENERATOR_MAP:
            logger.warning("Unknown product '%s', skipping.", product)
            continue

        GeneratorClass = GENERATOR_MAP[product]
        gen = GeneratorClass(seed=seed)
        loans, cashflows = gen.generate()
        results[product] = {"loans": loans, "cashflows": cashflows}

        if write_parquet:
            loans_path = output_dir / f"{product}_workouts.parquet"
            cashflows_path = output_dir / f"{product}_cashflows.parquet"
            loans.to_parquet(loans_path, index=False)
            cashflows.to_parquet(cashflows_path, index=False)
            logger.info("Written %s: %d loans -> %s", product, len(loans), loans_path.name)

    return results
