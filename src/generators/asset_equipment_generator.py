"""
Synthetic Asset & Equipment Finance Workout Data Generator.

Produces 10-year (2014-2024) synthetic workout histories for asset and
equipment finance defaults (chattel mortgage, finance lease, hire purchase),
calibrated to Australian bank practice.

Key calibration references:
  - APS 113 Attachment A: minimum 5-year observation for non-mortgage exposures
    (this generator provides 10 years).
  - APS 113 s.50: collateral value haircuts must reflect forced-sale conditions
    and secondary market depth.
  - AFIA Equipment Finance Industry Report (2022): secondary market liquidity
    varies dramatically by asset class — vehicle markets are deep, aviation and
    specialist marine are illiquid.
  - COVID-19 impact: technology assets depreciated rapidly; aviation suffered
    prolonged liquidity drought (2020-2022).

SYNTHETIC DATA DISCLAIMER:
All data produced by this generator is synthetically generated for portfolio
demonstration and model-development purposes only. No real customer, loan, or
internal workout data is included. This dataset is labelled
SYNTHETIC HISTORICAL CALIBRATION DATA — FOR DEMONSTRATION ONLY.
It must not be used as a substitute for actual internal loss experience in
regulatory capital submissions.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from src.generators.base_generator import (
    BaseWorkoutGenerator,
    MACRO_REGIME_BY_YEAR,
    DOWNTURN_YEARS,
    DATE_RANGE_START,
    DATE_RANGE_END,
)
from src.data.data_generation import _random_dates, _discount, STATES, STATE_WEIGHTS

# Secondary market liquidity by asset class
_LIQUIDITY_MAP = {
    "Vehicles":          "High",
    "Heavy Equipment":   "Medium",
    "Plant & Machinery": "Medium",
    "Technology":        "Illiquid",
    "Marine":            "Low",
    "Aircraft":          "Low",
    "Other":             "Illiquid",
}

# Recovery ranges by liquidity tier (lo, hi) — normal and downturn
_RECOVERY_RANGES = {
    "High":     {"normal": (0.60, 0.85), "downturn": (0.45, 0.70)},
    "Medium":   {"normal": (0.45, 0.70), "downturn": (0.30, 0.55)},
    "Low":      {"normal": (0.30, 0.55), "downturn": (0.15, 0.40)},
    "Illiquid": {"normal": (0.20, 0.50), "downturn": (0.10, 0.35)},
}


class AssetEquipmentWorkoutGenerator(BaseWorkoutGenerator):
    """
    Asset and equipment finance workout data generator.

    Recovery mechanics:
      - Recovery driven by secondary market liquidity of asset class.
      - Downturn: remarketing discount widens 10-20pp additional haircut.
      - Repossession timeline: 1-12 months (extends in downturn).
      - Technology and Poor condition assets face highest discounts.
    """

    product_name = "asset_equipment"
    min_records = 500

    _EAD_MEAN = 300_000.0
    _EAD_STD = 320_000.0
    _EAD_MIN = 20_000.0
    _EAD_MAX = 2_000_000.0

    def generate_loans(self) -> pd.DataFrame:
        n = self.n
        rng = self.rng

        loan_ids = [f"AEF-{i:04d}" for i in range(1, n + 1)]

        default_dates = _random_dates(DATE_RANGE_START, DATE_RANGE_END, n, rng)
        origination_dates = [
            d - pd.Timedelta(days=int(rng.uniform(365, 365 * 7)))
            for d in default_dates
        ]
        default_years = [d.year for d in default_dates]
        downturn_flags = np.array([1 if y in DOWNTURN_YEARS else 0 for y in default_years])

        ead = self._ead_array(self._EAD_MEAN, self._EAD_STD)
        ead = np.clip(ead, self._EAD_MIN, self._EAD_MAX)

        asset_class = rng.choice(
            ["Heavy Equipment", "Vehicles", "Plant & Machinery",
             "Technology", "Marine", "Aircraft", "Other"],
            n, p=[0.20, 0.30, 0.20, 0.15, 0.05, 0.05, 0.05]
        )
        asset_condition_proxy = rng.choice(
            ["New", "Good", "Fair", "Poor"], n, p=[0.20, 0.40, 0.30, 0.10]
        )

        asset_age_years = rng.uniform(0.0, 15.0, n).round(1)

        # Repossession timeline — longer in downturn
        repossession_timeline = rng.randint(1, 13, n)
        repossession_timeline = np.where(
            downturn_flags == 1,
            np.clip(repossession_timeline + rng.randint(1, 5, n), 1, 18),
            repossession_timeline,
        ).astype(int)

        # Secondary market liquidity
        secondary_market_liquidity = [_LIQUIDITY_MAP[ac] for ac in asset_class]

        # Remarketing discount
        remarketing_discount = np.zeros(n)
        for i in range(n):
            base_disc = rng.uniform(0.10, 0.40)
            # Tech and Poor condition get higher discounts
            if asset_class[i] == "Technology" or asset_condition_proxy[i] == "Poor":
                base_disc = min(base_disc + rng.uniform(0.05, 0.15), 0.50)
            # Additional downturn haircut (10-20pp)
            if downturn_flags[i] == 1:
                base_disc = min(base_disc + rng.uniform(0.10, 0.20), 0.65)
            remarketing_discount[i] = round(base_disc, 4)

        residual_balloon_pct = rng.uniform(0.0, 0.40, n).round(4)

        # Recovery
        gross_recoveries = np.zeros(n)
        for i in range(n):
            liquidity = secondary_market_liquidity[i]
            dt = "downturn" if downturn_flags[i] == 1 else "normal"
            lo, hi = _RECOVERY_RANGES[liquidity][dt]
            rec_rate = rng.uniform(lo, hi)
            gross_recoveries[i] = round(ead[i] * rec_rate, 2)

        is_cured = rng.uniform(0, 1, n) < 0.06
        cure_recovery_amount = np.where(
            is_cured,
            (ead * rng.uniform(0.80, 0.98, n)).round(2),
            0.0,
        )
        gross_recoveries = np.where(is_cured, cure_recovery_amount, gross_recoveries)

        direct_costs = (ead * rng.uniform(0.02, 0.06, n)).round(2)

        # All assets: seniority is senior secured (PPSR registered)
        seniority = ["Senior Secured"] * n

        workout_months = repossession_timeline + rng.randint(1, 6, n)
        workout_months = workout_months.astype(int)

        resolution_dates = [
            dd + pd.Timedelta(days=int(wm * 30.5))
            for dd, wm in zip(default_dates, workout_months)
        ]

        df = pd.DataFrame({
            "loan_id":                      loan_ids,
            "ead_at_default":               ead,
            "origination_date":             origination_dates,
            "default_date":                 default_dates,
            "gross_recoveries":             gross_recoveries,
            "direct_costs":                 direct_costs,
            "is_cured":                     is_cured.tolist(),
            "cure_recovery_amount":         cure_recovery_amount.tolist(),
            "resolution_date":              resolution_dates,
            "workout_months":               workout_months.tolist(),
            "seniority":                    seniority,
            # Product-specific
            "asset_class":                  asset_class.tolist(),
            "asset_age_years":              asset_age_years.tolist(),
            "asset_condition_proxy":        asset_condition_proxy.tolist(),
            "repossession_timeline_months": repossession_timeline.tolist(),
            "remarketing_discount_pct":     remarketing_discount.tolist(),
            "residual_balloon_pct":         residual_balloon_pct.tolist(),
            "secondary_market_liquidity":   secondary_market_liquidity,
        })

        return df

    def generate_cashflows(self, loans: pd.DataFrame) -> pd.DataFrame:
        rows = []

        for _, row in loans.iterrows():
            loan_id = row["loan_id"]
            default_date = pd.Timestamp(row["default_date"])
            gross_rec = float(row["gross_recoveries"])
            direct_cost = float(row["direct_costs"])
            workout_months = int(row["workout_months"])
            is_cured = bool(row["is_cured"])
            repossession_months = int(row["repossession_timeline_months"])

            repossession_days = repossession_months * 30
            total_days = workout_months * 30

            if is_cured:
                recovery_events = [(60, gross_rec)]
                cost_events = [(30, direct_cost)]
            else:
                # Repossession costs upfront, sale proceeds at remarketing
                recovery_events = [(total_days, gross_rec)]
                cost_events = [
                    (min(30, repossession_days), round(direct_cost * 0.40, 2)),  # repossession
                    (repossession_days, round(direct_cost * 0.30, 2)),            # storage
                    (total_days, round(direct_cost * 0.30, 2)),                  # remarketing
                ]

            rows.extend(
                self._build_cashflow_rows(loan_id, default_date, recovery_events, cost_events)
            )

        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["loan_id", "cashflow_date", "cashflow_type",
                     "recovery_amount", "direct_costs", "indirect_costs"]
        )
