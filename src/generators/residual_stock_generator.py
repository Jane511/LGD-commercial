"""
Synthetic Residual Stock Lending Workout Data Generator.

Produces 10-year (2014-2024) synthetic workout histories for residual stock
lending defaults (completed but unsold property inventory finance), calibrated
to Australian bank practice.

Key calibration references:
  - APS 113 Attachment A: minimum 5-year observation for non-mortgage exposures
    (this generator provides 10 years).
  - APS 113 s.50: residual stock collateral must be valued at a discounted
    bulk-sale basis in stress; orderly absorption rate is not applicable.
  - Property Council of Australia / CoreLogic (2020-2022): apartment and
    townhouse stock absorption rates fell sharply during COVID-19 lockdowns in
    VIC, NSW; developers faced prolonged holding cost drag.
  - APRA Quarterly ADI Property Exposure Statistics (2014-2024): concentration
    in apartment/unit product elevated in 2017-2020 cohort.

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

# Recovery ranges by market depth (lo, hi) — normal and downturn
_RECOVERY_RANGES = {
    "Deep":     {"normal": (0.65, 0.85), "downturn": (0.50, 0.72)},
    "Medium":   {"normal": (0.50, 0.72), "downturn": (0.38, 0.58)},
    "Thin":     {"normal": (0.38, 0.60), "downturn": (0.22, 0.45)},
    "Illiquid": {"normal": (0.25, 0.50), "downturn": (0.12, 0.35)},
}


class ResidualStockWorkoutGenerator(BaseWorkoutGenerator):
    """
    Residual stock lending workout data generator.

    Recovery mechanics:
      - Deep market (e.g., inner-city houses): 65-85% recovery.
      - Illiquid market (e.g., regional apartments): 35-60%, 20-45% downturn.
      - Holding costs reduce net recovery over extended sale timelines.
      - Discount-to-clear widens with market depth and downturn stress.
    """

    product_name = "residual_stock"
    min_records = 500

    _EAD_MEAN = 3_000_000.0
    _EAD_STD = 2_500_000.0
    _EAD_MIN = 500_000.0
    _EAD_MAX = 15_000_000.0

    def generate_loans(self) -> pd.DataFrame:
        n = self.n
        rng = self.rng

        loan_ids = [f"RSS-{i:04d}" for i in range(1, n + 1)]

        default_dates = _random_dates(DATE_RANGE_START, DATE_RANGE_END, n, rng)
        origination_dates = [
            d - pd.Timedelta(days=int(rng.uniform(180, 365 * 3)))
            for d in default_dates
        ]
        default_years = [d.year for d in default_dates]
        downturn_flags = np.array([1 if y in DOWNTURN_YEARS else 0 for y in default_years])

        ead = self._ead_array(self._EAD_MEAN, self._EAD_STD)
        ead = np.clip(ead, self._EAD_MIN, self._EAD_MAX)

        project_type = rng.choice(
            ["Apartments", "Townhouses", "Houses", "Commercial Units"],
            n, p=[0.45, 0.25, 0.20, 0.10]
        )
        market_depth_proxy = rng.choice(
            ["Deep", "Medium", "Thin", "Illiquid"],
            n, p=[0.30, 0.35, 0.25, 0.10]
        )

        stock_age_months = rng.randint(1, 37, n).astype(int)
        absorption_rate_pct = rng.uniform(0.01, 0.15, n).round(4)
        discount_to_clear = rng.uniform(0.05, 0.45, n).round(4)

        # Holding costs: monthly % of remaining value
        holding_cost_monthly_pct = rng.uniform(0.003, 0.012, n).round(5)

        # Sale timeline: longer for thin/illiquid markets and in downturn
        base_sale_timeline = rng.randint(3, 25, n)
        sale_timeline_months = np.where(
            market_depth_proxy == "Illiquid",
            np.clip(base_sale_timeline + rng.randint(5, 13, n), 6, 40),
            np.where(
                market_depth_proxy == "Thin",
                np.clip(base_sale_timeline + rng.randint(3, 9, n), 4, 36),
                base_sale_timeline,
            )
        )
        sale_timeline_months = np.where(
            downturn_flags == 1,
            np.clip(sale_timeline_months + rng.randint(3, 9, n), 3, 48),
            sale_timeline_months,
        ).astype(int)

        # Seniority: senior secured over the completed stock
        seniority = ["Senior Secured"] * n

        # Recovery
        gross_recoveries = np.zeros(n)
        for i in range(n):
            depth = market_depth_proxy[i]
            dt = "downturn" if downturn_flags[i] == 1 else "normal"
            lo, hi = _RECOVERY_RANGES[depth][dt]
            # Apply holding cost drag over sale timeline
            holding_drag = holding_cost_monthly_pct[i] * sale_timeline_months[i]
            rec_rate = max(0.0, rng.uniform(lo, hi) - holding_drag)
            gross_recoveries[i] = round(ead[i] * rec_rate, 2)

        is_cured = rng.uniform(0, 1, n) < 0.07
        cure_recovery_amount = np.where(
            is_cured,
            (ead * rng.uniform(0.85, 1.00, n)).round(2),
            0.0,
        )
        gross_recoveries = np.where(is_cured, cure_recovery_amount, gross_recoveries)

        direct_costs = (ead * rng.uniform(0.03, 0.08, n)).round(2)
        workout_months = sale_timeline_months.copy()

        resolution_dates = [
            dd + pd.Timedelta(days=int(wm * 30.5))
            for dd, wm in zip(default_dates, workout_months)
        ]

        df = pd.DataFrame({
            "loan_id":                  loan_ids,
            "ead_at_default":           ead,
            "origination_date":         origination_dates,
            "default_date":             default_dates,
            "gross_recoveries":         gross_recoveries,
            "direct_costs":             direct_costs,
            "is_cured":                 is_cured.tolist(),
            "cure_recovery_amount":     cure_recovery_amount.tolist(),
            "resolution_date":          resolution_dates,
            "workout_months":           workout_months.tolist(),
            "seniority":                seniority,
            # Product-specific
            "stock_age_months":         stock_age_months.tolist(),
            "absorption_rate_pct":      absorption_rate_pct.tolist(),
            "discount_to_clear_pct":    discount_to_clear.tolist(),
            "holding_cost_monthly_pct": holding_cost_monthly_pct.tolist(),
            "sale_timeline_months":     sale_timeline_months.tolist(),
            "project_type":             project_type.tolist(),
            "market_depth_proxy":       market_depth_proxy.tolist(),
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
            market_depth = str(row["market_depth_proxy"])

            total_days = max(workout_months * 30, 30)

            if is_cured:
                recovery_events = [(90, gross_rec)]
                cost_events = [(30, direct_cost)]
            elif market_depth in ("Deep", "Medium"):
                # Staged unit sales
                recovery_events = [
                    (int(total_days * 0.35), round(gross_rec * 0.35, 2)),
                    (int(total_days * 0.65), round(gross_rec * 0.40, 2)),
                    (total_days, round(gross_rec * 0.25, 2)),
                ]
                cost_events = [
                    (30, round(direct_cost * 0.30, 2)),
                    (int(total_days * 0.50), round(direct_cost * 0.40, 2)),
                    (total_days, round(direct_cost * 0.30, 2)),
                ]
            else:
                # Thin/Illiquid: bulk discount sale or forced clearance
                recovery_events = [(total_days, gross_rec)]
                cost_events = [
                    (30, round(direct_cost * 0.40, 2)),
                    (total_days, round(direct_cost * 0.60, 2)),
                ]

            rows.extend(
                self._build_cashflow_rows(loan_id, default_date, recovery_events, cost_events)
            )

        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["loan_id", "cashflow_date", "cashflow_type",
                     "recovery_amount", "direct_costs", "indirect_costs"]
        )
