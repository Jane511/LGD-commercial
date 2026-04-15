"""
Synthetic Land & Subdivision Lending Workout Data Generator.

Produces 10-year (2014-2024) synthetic workout histories for land and
subdivision loan defaults (raw land, rezoning, titled lot production),
calibrated to Australian bank practice.

Key calibration references:
  - APS 113 Attachment A: minimum 5-year observation for non-mortgage exposures
    (this generator provides 10 years).
  - APS 113 s.50: vacant land collateral attracts the highest forced-sale
    discounts given illiquidity and limited buyer pool.
  - APRA Prudential Practice Guide APG 220 (2021): land exposures without
    approved permits are assigned the most conservative LGD treatment.
  - HIA Housing Industry Report (2020-2022): greenfield land markets froze
    briefly during COVID-19 before rebounding; outer-metro raw land suffered
    the most during the 2015-2016 mining downturn in WA/QLD.
  - CoreLogic Residential Land Report (2014-2024): serviced/approved lots
    showed far greater price stability than raw or rezoning-pending land.

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

# Recovery ranges by zoning stage (lo, hi) — normal and downturn
_RECOVERY_RANGES = {
    "Serviced":          {"normal": (0.55, 0.75), "downturn": (0.40, 0.62)},
    "Approved":          {"normal": (0.48, 0.68), "downturn": (0.33, 0.55)},
    "Rezoning Pending":  {"normal": (0.28, 0.52), "downturn": (0.15, 0.38)},
    "Raw":               {"normal": (0.15, 0.45), "downturn": (0.05, 0.25)},
}


class LandSubdivisionWorkoutGenerator(BaseWorkoutGenerator):
    """
    Land and subdivision lending workout data generator.

    Recovery mechanics:
      - Zoning stage is the primary driver: serviced lots are most liquid.
      - Raw land in downturn: 60-90% LGD range in severe stress cases.
      - Infrastructure complexity raises costs and extends workout.
      - Haircut severity calibrated against APS 113 Attachment A floor levels.
    """

    product_name = "land_subdivision"
    min_records = 500

    _EAD_MEAN = 2_500_000.0
    _EAD_STD = 2_000_000.0
    _EAD_MIN = 500_000.0
    _EAD_MAX = 10_000_000.0

    def generate_loans(self) -> pd.DataFrame:
        n = self.n
        rng = self.rng

        loan_ids = [f"LND-{i:04d}" for i in range(1, n + 1)]

        default_dates = _random_dates(DATE_RANGE_START, DATE_RANGE_END, n, rng)
        origination_dates = [
            d - pd.Timedelta(days=int(rng.uniform(365, 365 * 5)))
            for d in default_dates
        ]
        default_years = [d.year for d in default_dates]
        downturn_flags = np.array([1 if y in DOWNTURN_YEARS else 0 for y in default_years])

        ead = self._ead_array(self._EAD_MEAN, self._EAD_STD)
        ead = np.clip(ead, self._EAD_MIN, self._EAD_MAX)

        zoning_stage = rng.choice(
            ["Raw", "Rezoning Pending", "Approved", "Serviced"],
            n, p=[0.20, 0.20, 0.35, 0.25]
        )
        permit_stage = rng.choice(
            ["Pre-Permit", "Permitted", "Under Construction"],
            n, p=[0.30, 0.40, 0.30]
        )
        market_depth_proxy = rng.choice(
            ["Deep", "Medium", "Thin", "Illiquid"],
            n, p=[0.20, 0.35, 0.30, 0.15]
        )
        infrastructure_complexity = rng.choice(
            ["Low", "Medium", "High"], n, p=[0.40, 0.40, 0.20]
        )

        # Sale timeline: longer for Raw land and complex infrastructure
        base_sale_timeline = rng.randint(6, 37, n)
        # Raw or High complexity adds time
        raw_mask = zoning_stage == "Raw"
        complex_mask = infrastructure_complexity == "High"
        sale_timeline_months = np.where(
            raw_mask | complex_mask,
            np.clip(base_sale_timeline + rng.randint(6, 25, n), 12, 65),
            base_sale_timeline,
        )
        sale_timeline_months = np.where(
            downturn_flags == 1,
            np.clip(sale_timeline_months + rng.randint(3, 13, n), 6, 72),
            sale_timeline_months,
        ).astype(int)

        # Haircut severity (% of collateral value written off in realisation)
        haircut_severity_pct = np.array([
            rng.uniform(0.40, 0.70) if zoning_stage[i] == "Raw"
            else rng.uniform(0.25, 0.50) if zoning_stage[i] == "Rezoning Pending"
            else rng.uniform(0.15, 0.35) if zoning_stage[i] == "Approved"
            else rng.uniform(0.10, 0.28)   # Serviced
            for i in range(n)
        ]).round(4)

        seniority = ["Senior Secured"] * n

        # Recovery
        gross_recoveries = np.zeros(n)
        for i in range(n):
            stage = zoning_stage[i]
            dt = "downturn" if downturn_flags[i] == 1 else "normal"
            lo, hi = _RECOVERY_RANGES[stage][dt]
            rec_rate = rng.uniform(lo, hi)
            gross_recoveries[i] = round(ead[i] * rec_rate, 2)

        is_cured = rng.uniform(0, 1, n) < 0.06
        cure_recovery_amount = np.where(
            is_cured,
            (ead * rng.uniform(0.82, 0.98, n)).round(2),
            0.0,
        )
        gross_recoveries = np.where(is_cured, cure_recovery_amount, gross_recoveries)

        direct_costs = (ead * rng.uniform(0.03, 0.10, n)).round(2)
        workout_months = sale_timeline_months.copy()

        resolution_dates = [
            dd + pd.Timedelta(days=int(wm * 30.5))
            for dd, wm in zip(default_dates, workout_months)
        ]

        df = pd.DataFrame({
            "loan_id":                    loan_ids,
            "ead_at_default":             ead,
            "origination_date":           origination_dates,
            "default_date":               default_dates,
            "gross_recoveries":           gross_recoveries,
            "direct_costs":               direct_costs,
            "is_cured":                   is_cured.tolist(),
            "cure_recovery_amount":       cure_recovery_amount.tolist(),
            "resolution_date":            resolution_dates,
            "workout_months":             workout_months.tolist(),
            "seniority":                  seniority,
            # Product-specific
            "zoning_stage":               zoning_stage.tolist(),
            "permit_stage":               permit_stage.tolist(),
            "market_depth_proxy":         market_depth_proxy.tolist(),
            "sale_timeline_months":       sale_timeline_months.tolist(),
            "haircut_severity_pct":       haircut_severity_pct.tolist(),
            "infrastructure_complexity":  infrastructure_complexity.tolist(),
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
            zoning_stage = str(row["zoning_stage"])

            total_days = max(workout_months * 30, 60)

            if is_cured:
                recovery_events = [(90, gross_rec)]
                cost_events = [(30, direct_cost)]
            elif zoning_stage in ("Serviced", "Approved"):
                # Staged lot sales
                recovery_events = [
                    (int(total_days * 0.40), round(gross_rec * 0.45, 2)),
                    (int(total_days * 0.75), round(gross_rec * 0.35, 2)),
                    (total_days, round(gross_rec * 0.20, 2)),
                ]
                cost_events = [
                    (60, round(direct_cost * 0.25, 2)),
                    (int(total_days * 0.40), round(direct_cost * 0.45, 2)),
                    (total_days, round(direct_cost * 0.30, 2)),
                ]
            else:
                # Raw/Rezoning: bulk land sale after lengthy process
                recovery_events = [(total_days, gross_rec)]
                cost_events = [
                    (60, round(direct_cost * 0.30, 2)),
                    (int(total_days * 0.50), round(direct_cost * 0.40, 2)),
                    (total_days, round(direct_cost * 0.30, 2)),
                ]

            rows.extend(
                self._build_cashflow_rows(loan_id, default_date, recovery_events, cost_events)
            )

        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["loan_id", "cashflow_date", "cashflow_type",
                     "recovery_amount", "direct_costs", "indirect_costs"]
        )
