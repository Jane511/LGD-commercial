"""
Synthetic Construction & Development Finance Workout Data Generator.

Produces 10-year (2014-2024) synthetic workout histories for property
development finance defaults, calibrated to Australian bank practice.

Key calibration references:
  - APS 113 Attachment A: minimum 5-year observation for non-mortgage exposures
    (this generator provides 10 years).
  - APS 113 s.50: development finance collateral must be valued at forced-sale
    basis accounting for incomplete construction.
  - APRA Prudential Practice Guide APG 223 — Residential Mortgage Lending (2019):
    pre-sale cover and LTC ratios are primary risk drivers for construction loans.
  - RBA Financial Stability Review (2020-H2): construction sector stress elevated
    during COVID-19; developer insolvencies rose sharply in 2020-2021.
  - Property Council of Australia Market Outlook (2022): recovery lags in
    residential development significantly extended post-2020.

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
from src.data_generation import _random_dates, _discount, STATES, STATE_WEIGHTS

# Cost-to-complete by construction stage (fraction of total project cost remaining)
_COST_TO_COMPLETE = {
    "Pre-Start": (0.85, 0.95),
    "Early":     (0.60, 0.80),
    "Mid":       (0.35, 0.55),
    "Late":      (0.05, 0.20),
    "Complete":  (0.00, 0.05),
}

# Recovery ranges by stage (lo, hi) — normal and downturn
_RECOVERY_RANGES = {
    "Complete":  {"normal": (0.60, 0.90), "downturn": (0.45, 0.75)},
    "Late":      {"normal": (0.50, 0.80), "downturn": (0.35, 0.65)},
    "Mid":       {"normal": (0.35, 0.65), "downturn": (0.20, 0.50)},
    "Early":     {"normal": (0.20, 0.50), "downturn": (0.10, 0.35)},
    "Pre-Start": {"normal": (0.15, 0.40), "downturn": (0.05, 0.25)},
}


class DevelopmentFinanceWorkoutGenerator(BaseWorkoutGenerator):
    """
    Property construction and development finance workout data generator.

    Recovery mechanics:
      - Completion stage is the primary recovery driver.
      - Complete/Late: orderly sale possible; near-term recovery.
      - Pre-Start/Early: minimal improvements; land value underpins recovery.
      - Downturn: cap-rate expansion and market absorption slowdown compress GRV.
      - Recovery lag 12-48 months (longer in downturn / complex projects).
    """

    product_name = "development_finance"
    min_records = 500

    _EAD_MEAN = 4_000_000.0
    _EAD_STD = 3_500_000.0
    _EAD_MIN = 500_000.0
    _EAD_MAX = 20_000_000.0

    def generate_loans(self) -> pd.DataFrame:
        n = self.n
        rng = self.rng

        loan_ids = [f"DEV-{i:04d}" for i in range(1, n + 1)]

        default_dates = _random_dates(DATE_RANGE_START, DATE_RANGE_END, n, rng)
        origination_dates = [
            d - pd.Timedelta(days=int(rng.uniform(180, 365 * 4)))
            for d in default_dates
        ]
        default_years = [d.year for d in default_dates]
        downturn_flags = np.array([1 if y in DOWNTURN_YEARS else 0 for y in default_years])

        ead = self._ead_array(self._EAD_MEAN, self._EAD_STD)
        ead = np.clip(ead, self._EAD_MIN, self._EAD_MAX)

        loan_to_cost_pct = rng.uniform(0.55, 0.85, n)
        # GRV = ead / ltc * (1.1 to 1.3) — gross realisation value
        grv_multiplier = rng.uniform(1.10, 1.30, n)
        grv_estimate = (ead / loan_to_cost_pct * grv_multiplier).round(2)

        completion_stage = rng.choice(
            ["Pre-Start", "Early", "Mid", "Late", "Complete"],
            n, p=[0.15, 0.20, 0.30, 0.25, 0.10]
        )

        cost_to_complete_pct = np.array([
            rng.uniform(*_COST_TO_COMPLETE[s]) for s in completion_stage
        ])

        presale_cover_pct = rng.uniform(0.0, 1.20, n).round(4)
        sell_through_rate = rng.uniform(0.10, 1.00, n).round(4)

        resolution_type = rng.choice(
            ["Refinanced", "Forced Sale", "Completed and Sold"],
            n, p=[0.25, 0.50, 0.25]
        )

        # Recovery lag: longer in downturn
        base_lag = rng.randint(12, 37, n)
        recovery_lag_months = np.where(
            downturn_flags == 1,
            np.clip(base_lag + rng.randint(6, 13, n), 12, 48),
            base_lag,
        ).astype(int)

        # Construction stress flag: cost overruns > 10%
        construction_stress_flag = (rng.uniform(0, 1, n) < 0.25).astype(int)

        # Recovery
        gross_recoveries = np.zeros(n)
        for i in range(n):
            stage = completion_stage[i]
            dt = "downturn" if downturn_flags[i] == 1 else "normal"
            lo, hi = _RECOVERY_RANGES[stage][dt]
            # Severe stress further widens loss for Pre-Start in downturn
            if stage == "Pre-Start" and downturn_flags[i] == 1:
                hi = max(hi, 0.30)  # cap upside at 30% for pre-start downturn
            rec_rate = rng.uniform(lo, hi)
            gross_recoveries[i] = round(ead[i] * rec_rate, 2)

        is_cured = rng.uniform(0, 1, n) < 0.07
        cure_recovery_amount = np.where(
            is_cured,
            (ead * rng.uniform(0.85, 1.00, n)).round(2),
            0.0,
        )
        gross_recoveries = np.where(is_cured, cure_recovery_amount, gross_recoveries)

        direct_costs = (ead * rng.uniform(0.03, 0.10, n)).round(2)

        seniority = ["Senior Secured"] * n
        workout_months = recovery_lag_months.copy()

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
            "loan_to_cost_pct":           loan_to_cost_pct.round(4).tolist(),
            "grv_estimate":               grv_estimate.tolist(),
            "completion_stage_at_default": completion_stage.tolist(),
            "cost_to_complete_pct":       cost_to_complete_pct.round(4).tolist(),
            "presale_cover_pct":          presale_cover_pct.tolist(),
            "sell_through_rate":          sell_through_rate.tolist(),
            "resolution_type":            resolution_type.tolist(),
            "recovery_lag_months":        recovery_lag_months.tolist(),
            "construction_stress_flag":   construction_stress_flag.tolist(),
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
            resolution_type = str(row["resolution_type"])

            total_days = workout_months * 30

            if is_cured:
                recovery_events = [(90, gross_rec)]
                cost_events = [(30, direct_cost)]
            elif resolution_type == "Refinanced":
                # Refinance: lump-sum paydown
                recovery_events = [(int(total_days * 0.70), gross_rec)]
                cost_events = [
                    (30, round(direct_cost * 0.30, 2)),
                    (int(total_days * 0.70), round(direct_cost * 0.70, 2)),
                ]
            elif resolution_type == "Completed and Sold":
                # Completion then staged lot/unit sales
                recovery_events = [
                    (int(total_days * 0.60), round(gross_rec * 0.40, 2)),
                    (int(total_days * 0.80), round(gross_rec * 0.35, 2)),
                    (total_days, round(gross_rec * 0.25, 2)),
                ]
                cost_events = [
                    (60, round(direct_cost * 0.20, 2)),
                    (int(total_days * 0.50), round(direct_cost * 0.50, 2)),
                    (total_days, round(direct_cost * 0.30, 2)),
                ]
            else:
                # Forced sale: lower recovery, faster but impaired
                recovery_events = [
                    (int(total_days * 0.50), round(gross_rec * 0.70, 2)),
                    (total_days, round(gross_rec * 0.30, 2)),
                ]
                cost_events = [
                    (30, round(direct_cost * 0.35, 2)),
                    (int(total_days * 0.50), round(direct_cost * 0.40, 2)),
                    (total_days, round(direct_cost * 0.25, 2)),
                ]

            rows.extend(
                self._build_cashflow_rows(loan_id, default_date, recovery_events, cost_events)
            )

        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["loan_id", "cashflow_date", "cashflow_type",
                     "recovery_amount", "direct_costs", "indirect_costs"]
        )
