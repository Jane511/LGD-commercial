"""
Base class for all synthetic historical workout data generators.

All 11 product generators inherit from BaseWorkoutGenerator. Shared
utilities (_random_dates, _discount, _build_discount_rate, STATES, STATE_WEIGHTS)
are imported from src.data_generation to avoid duplication.

Observation period: 2014–2024 (10 years) — exceeds APS 113 Attachment A minimums:
  - Residential mortgage: minimum 7 years (this provides 10)
  - All other products: minimum 5 years (this provides 10)

The 2014–2024 window includes:
  - Expansion: 2014–2019 (with minor stress in 2015-2016 mining downturn)
  - Severe stress: 2020–2021 (COVID-19)
  - Normalisation: 2022–2024 (RBA rate cycle, mild stress)

SYNTHETIC DATA DISCLAIMER:
All data produced by these generators is synthetically generated for portfolio
demonstration purposes only. No real internal workout tape is included.
These datasets are labelled SYNTHETIC HISTORICAL CALIBRATION DATA — FOR DEMONSTRATION ONLY.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Import shared helpers from existing data_generation module (do not duplicate)
from src.data_generation import (
    STATES,
    STATE_WEIGHTS,
    _build_discount_rate,
    _discount,
    _random_dates,
)
from src.rba_rates_loader import load_rba_lending_rates, build_discount_rate_register

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Observation window constants
# ---------------------------------------------------------------------------

DATE_RANGE_START = datetime(2014, 1, 1)
DATE_RANGE_END = datetime(2024, 6, 30)

# Minimum observation periods per APS 113 Attachment A
OBSERVATION_PERIODS_BY_PRODUCT: dict[str, int] = {
    "mortgage":              10,   # APRA min 7y; provide 10y
    "commercial_cashflow":   10,
    "receivables":           10,
    "trade_contingent":      10,
    "asset_equipment":       10,
    "development_finance":   10,
    "cre_investment":        10,
    "residual_stock":        10,
    "land_subdivision":      10,
    "bridging":              10,
    "mezz_second_mortgage":  10,
}

# Macro regime by year (driven by real RBA/ABS history)
# Source: industry-analysis repo macro_regime_flags.parquet
MACRO_REGIME_BY_YEAR: dict[int, str] = {
    2014: "expansion",
    2015: "mild_stress",      # mining sector downturn, iron ore price collapse
    2016: "mild_stress",      # continued mining headwinds
    2017: "expansion",
    2018: "expansion",
    2019: "expansion",
    2020: "severe_stress",    # COVID-19 pandemic
    2021: "severe_stress",    # continued COVID disruption, economic recovery starting
    2022: "mild_stress",      # RBA rate hike cycle begins, cost-of-living pressure
    2023: "mild_stress",      # persistent inflation, rate plateau
    2024: "expansion",        # easing cycle beginning, stabilisation
}

DOWNTURN_YEARS = {yr for yr, regime in MACRO_REGIME_BY_YEAR.items() if regime == "severe_stress"}


# ---------------------------------------------------------------------------
# Base generator class
# ---------------------------------------------------------------------------

class BaseWorkoutGenerator(ABC):
    """
    Abstract base class for all 11 product workout generators.

    Subclasses must implement:
        - product_name (class attribute)
        - min_records (class attribute)
        - generate_loans() -> pd.DataFrame
        - generate_cashflows(loans: pd.DataFrame) -> pd.DataFrame

    Public interface:
        generator = SomeProductGenerator(n_defaults=1000, seed=42)
        loans, cashflows = generator.generate()

    All generated DataFrames include:
        - discount_rate sourced from RBA B6 rates (not RBA+300bps flat)
        - lip_costs (Loss Identification Period costs per APS 113 s.32)
        - macro_regime and downturn_flag
        - synthetic_data_flag = True
    """

    product_name: str = "base"
    min_records: int = 500

    def __init__(
        self,
        n_defaults: int | None = None,
        seed: int = 42,
        rates_df: pd.DataFrame | None = None,
    ) -> None:
        self.n = n_defaults or self.min_records
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Load RBA B6 rates once; shared across all generators
        self._rates_df = rates_df if rates_df is not None else load_rba_lending_rates()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def generate(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate loans and cashflows DataFrames.

        Returns
        -------
        (loans, cashflows) — both as pandas DataFrames.
        """
        logger.info("Generating %s workout data: n=%d, seed=%d", self.product_name, self.n, self.seed)

        loans = self.generate_loans()
        loans = self._add_common_fields(loans)
        loans = self._add_rba_discount_rates(loans)
        loans = self._add_lip_costs(loans)
        self._validate_loans(loans)

        cashflows = self.generate_cashflows(loans)
        self._validate_cashflows(cashflows)

        logger.info(
            "Generated %s: %d loans, %d cashflow rows",
            self.product_name, len(loans), len(cashflows),
        )
        return loans, cashflows

    # ------------------------------------------------------------------
    # Abstract methods (must implement in subclass)
    # ------------------------------------------------------------------

    @abstractmethod
    def generate_loans(self) -> pd.DataFrame:
        """Return raw loans DataFrame before common field enrichment."""
        ...

    @abstractmethod
    def generate_cashflows(self, loans: pd.DataFrame) -> pd.DataFrame:
        """Return cashflows DataFrame. Receives enriched loans DataFrame."""
        ...

    # ------------------------------------------------------------------
    # Shared enrichment methods
    # ------------------------------------------------------------------

    def _add_common_fields(self, loans: pd.DataFrame) -> pd.DataFrame:
        """Add fields shared by all products: dates, regime, geography, flags."""
        df = loans.copy()

        # Origination and default dates (2014-2024)
        if "default_date" not in df.columns:
            df["default_date"] = _random_dates(DATE_RANGE_START, DATE_RANGE_END, len(df), self.rng)
        if "origination_date" not in df.columns:
            df["origination_date"] = [
                d - pd.Timedelta(days=int(self.rng.uniform(180, 365 * 5)))
                for d in df["default_date"]
            ]

        # Year columns for regime classification
        df["default_year"] = pd.to_datetime(df["default_date"]).dt.year
        df["origination_year"] = pd.to_datetime(df["origination_date"]).dt.year

        # Macro regime
        df["macro_regime"] = df["default_year"].map(MACRO_REGIME_BY_YEAR).fillna("expansion")
        df["downturn_flag"] = df["default_year"].isin(DOWNTURN_YEARS).astype(int)

        # Geography
        if "geography" not in df.columns:
            df["geography"] = self.rng.choice(STATES, len(df), p=STATE_WEIGHTS)

        # Product labelling
        df["product"] = self.product_name
        df["synthetic_data_flag"] = True
        df["data_label"] = "SYNTHETIC HISTORICAL CALIBRATION DATA — FOR DEMONSTRATION ONLY"

        return df

    def _add_rba_discount_rates(self, loans: pd.DataFrame) -> pd.DataFrame:
        """
        Set discount_rate using real RBA B6 indicator lending rates.

        Overwrites any discount_rate column already present (generators should
        not set this themselves — let the base class handle it via RBA data).

        APS 113 s.50: use contractual interest rate; fallback to RBA cash + 300bps.
        """
        enriched = build_discount_rate_register(
            loans=loans,
            rates_df=self._rates_df,
            product_type=self.product_name,
            origination_year_col="origination_year",
        )
        return enriched

    def _add_lip_costs(self, loans: pd.DataFrame) -> pd.DataFrame:
        """
        Add Loss Identification Period costs per APS 113 s.32.

        LIP costs = direct costs incurred between actual default event and
        formal default identification (90-day window, typical AU bank practice).

        These are stored separately from direct_costs in the loans DataFrame
        so compute_realised_lgd() can treat them as a first charge against recovery.
        """
        df = loans.copy()
        if "lip_costs" not in df.columns:
            # LIP costs: ~0.5–2% of EAD, representing early legal and admin fees
            ead = pd.to_numeric(df.get("ead_at_default", df.get("ead", 0)), errors="coerce").fillna(0)
            lip_pct = self.rng.uniform(0.005, 0.020, len(df))
            df["lip_costs"] = (ead * lip_pct).round(2)
        return df

    # ------------------------------------------------------------------
    # Shared cashflow builder
    # ------------------------------------------------------------------

    def _build_cashflow_rows(
        self,
        loan_id: str | int,
        default_date: datetime,
        recovery_events: list[tuple[int, float]],   # (days_after_default, amount)
        cost_events: list[tuple[int, float]],        # (days_after_default, amount)
    ) -> list[dict]:
        """Build a list of cashflow dicts for a single loan."""
        rows = []
        for days, amount in recovery_events:
            if amount > 0:
                rows.append({
                    "loan_id": loan_id,
                    "cashflow_date": default_date + pd.Timedelta(days=days),
                    "cashflow_type": "recovery",
                    "recovery_amount": round(amount, 2),
                    "direct_costs": 0.0,
                    "indirect_costs": 0.0,
                })
        for days, amount in cost_events:
            if amount > 0:
                rows.append({
                    "loan_id": loan_id,
                    "cashflow_date": default_date + pd.Timedelta(days=days),
                    "cashflow_type": "cost",
                    "recovery_amount": 0.0,
                    "direct_costs": round(amount, 2),
                    "indirect_costs": round(amount * 0.20, 2),
                })
        return rows

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    REQUIRED_LOAN_COLUMNS = [
        "loan_id", "product", "ead_at_default", "default_date",
        "origination_date", "default_year", "origination_year",
        "macro_regime", "downturn_flag", "discount_rate",
        "lip_costs", "geography", "synthetic_data_flag",
    ]

    REQUIRED_CASHFLOW_COLUMNS = [
        "loan_id", "cashflow_date", "cashflow_type",
        "recovery_amount", "direct_costs", "indirect_costs",
    ]

    def _validate_loans(self, loans: pd.DataFrame) -> None:
        missing = [c for c in self.REQUIRED_LOAN_COLUMNS if c not in loans.columns]
        if missing:
            raise ValueError(
                f"{self.product_name} generator: loans DataFrame missing required columns: {missing}"
            )
        n_years = loans["default_year"].nunique()
        min_required = OBSERVATION_PERIODS_BY_PRODUCT.get(self.product_name, 5)
        if n_years < min_required:
            raise ValueError(
                f"{self.product_name}: observation period {n_years} years < "
                f"APS 113 Attachment A minimum {min_required} years."
            )
        if (loans["ead_at_default"] <= 0).any():
            raise ValueError(f"{self.product_name}: ead_at_default has non-positive values.")
        if len(loans) < self.min_records:
            logger.warning(
                "%s: generated %d loans, below recommended minimum %d.",
                self.product_name, len(loans), self.min_records,
            )

    def _validate_cashflows(self, cashflows: pd.DataFrame) -> None:
        missing = [c for c in self.REQUIRED_CASHFLOW_COLUMNS if c not in cashflows.columns]
        if missing:
            raise ValueError(
                f"{self.product_name} generator: cashflows DataFrame missing columns: {missing}"
            )

    # ------------------------------------------------------------------
    # Shared helpers for sub-generators
    # ------------------------------------------------------------------

    def _ead_array(self, mean: float, std: float) -> np.ndarray:
        """Generate realistic EAD values — lognormal distribution."""
        log_mean = np.log(mean) - 0.5 * np.log(1 + (std / mean) ** 2)
        log_std = np.sqrt(np.log(1 + (std / mean) ** 2))
        return np.exp(self.rng.normal(log_mean, log_std, self.n)).round(2)

    def _recovery_rate(
        self,
        base: float,
        std: float,
        downturn_flags: np.ndarray,
        downturn_haircut: float = 0.15,
    ) -> np.ndarray:
        """
        Generate recovery rates with downturn stress applied.

        base : mean recovery rate in non-downturn years
        downturn_haircut : absolute reduction in downturn (e.g., 0.15 = 15pp lower)
        """
        rates = self.rng.normal(base, std, self.n)
        rates = np.where(downturn_flags == 1, rates - downturn_haircut, rates)
        return np.clip(rates, 0.0, 1.0)
