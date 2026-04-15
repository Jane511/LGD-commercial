"""
Synthetic Commercial Real Estate (CRE) Investment Lending Workout Data Generator.

Produces 10-year (2014-2024) synthetic workout histories for CRE investment
loan defaults (income-producing real estate), calibrated to Australian bank
practice.

Key calibration references:
  - APS 113 Attachment A: minimum 5-year observation for non-mortgage exposures
    (this generator provides 10 years).
  - APS 113 s.50: CRE collateral haircuts must reflect forced-sale conditions,
    cap-rate expansion and vacancy assumptions in stress.
  - APRA Information Paper — Commercial Property Lending (2022): LVR and DSCR
    are primary risk drivers; LVR > 70% cohorts show materially higher LGD.
  - RBA Financial Stability Review (2020, 2022): COVID-19 drove significant
    cap-rate expansion for CBD office and retail; industrial remained resilient.
  - MSCI/IPD Australia Quarterly Property Index (2014-2024): cap rates by
    sector used to calibrate downturn GRV compression.

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


class CREInvestmentWorkoutGenerator(BaseWorkoutGenerator):
    """
    Commercial real estate investment lending workout data generator.

    Recovery mechanics:
      - Low LVR (<60%) + high DSCR (>1.5): 65-85% recovery.
      - High LVR (>70%) or low DSCR (<1.2): 40-65% recovery.
      - Downturn: cap-rate expansion shock (150bps) compresses GRV 10-20%.
      - Office and Retail CRE suffer most in downturn; Industrial resilient.
      - Resolution timeline 12-48 months.
    """

    product_name = "cre_investment"
    min_records = 500

    _EAD_MEAN = 6_000_000.0
    _EAD_STD = 5_000_000.0
    _EAD_MIN = 1_000_000.0
    _EAD_MAX = 30_000_000.0

    def generate_loans(self) -> pd.DataFrame:
        n = self.n
        rng = self.rng

        loan_ids = [f"CRE-{i:04d}" for i in range(1, n + 1)]

        default_dates = _random_dates(DATE_RANGE_START, DATE_RANGE_END, n, rng)
        origination_dates = [
            d - pd.Timedelta(days=int(rng.uniform(365, 365 * 7)))
            for d in default_dates
        ]
        default_years = [d.year for d in default_dates]
        downturn_flags = np.array([1 if y in DOWNTURN_YEARS else 0 for y in default_years])

        ead = self._ead_array(self._EAD_MEAN, self._EAD_STD)
        ead = np.clip(ead, self._EAD_MIN, self._EAD_MAX)

        asset_class_cre = rng.choice(
            ["Office", "Retail", "Industrial", "Hotel", "Mixed"],
            n, p=[0.25, 0.25, 0.30, 0.10, 0.10]
        )

        lvr_at_origination = rng.uniform(0.45, 0.80, n)
        # LVR drift at default: worsens for stressed assets
        lvr_drift = rng.uniform(-0.05, 0.20, n)
        lvr_at_default = np.clip(lvr_at_origination + lvr_drift, 0.20, 1.30)

        dscr_at_origination = rng.uniform(1.10, 2.50, n)
        dscr_multiplier = rng.uniform(0.50, 1.00, n)
        dscr_at_default = (dscr_at_origination * dscr_multiplier).round(4)

        wale_years = rng.uniform(0.5, 8.0, n).round(2)
        vacancy_rate_pct = rng.uniform(0.0, 0.50, n).round(4)
        tenant_concentration_pct = rng.uniform(0.10, 0.90, n).round(4)

        refinance_outcome = rng.choice(
            ["Successful", "Failed"], n, p=[0.60, 0.40]
        )

        # Forced sale haircut: larger in downturn and for distressed properties
        base_haircut = rng.uniform(0.10, 0.35, n)
        forced_sale_haircut = np.where(
            downturn_flags == 1,
            np.clip(base_haircut + rng.uniform(0.05, 0.15, n), 0.10, 0.50),
            base_haircut,
        ).round(4)

        resolution_timeline_months = rng.randint(12, 49, n).astype(int)
        resolution_timeline_months = np.where(
            downturn_flags == 1,
            np.clip(resolution_timeline_months + rng.randint(3, 9, n), 12, 60),
            resolution_timeline_months,
        ).astype(int)

        # Seniority: CRE investment loans are senior secured (registered mortgage)
        seniority = ["Senior Secured"] * n

        # Recovery logic
        gross_recoveries = np.zeros(n)
        for i in range(n):
            dt = downturn_flags[i]
            lvr = lvr_at_default[i]
            dscr = dscr_at_default[i]

            # Assign recovery tier
            if lvr < 0.60 and dscr > 1.50:
                lo, hi = (0.50, 0.72) if dt else (0.65, 0.85)
            elif lvr > 0.70 or dscr < 1.20:
                lo, hi = (0.30, 0.52) if dt else (0.40, 0.65)
            else:
                lo, hi = (0.40, 0.62) if dt else (0.52, 0.75)

            # Downturn: cap-rate expansion shock 10-20% additional GRV compression
            rec_rate = rng.uniform(lo, hi)
            if dt:
                cap_rate_shock = rng.uniform(0.10, 0.20)
                rec_rate = max(0.0, rec_rate * (1.0 - cap_rate_shock))

            gross_recoveries[i] = round(ead[i] * rec_rate, 2)

        is_cured = rng.uniform(0, 1, n) < 0.08
        cure_recovery_amount = np.where(
            is_cured,
            (ead * rng.uniform(0.85, 1.00, n)).round(2),
            0.0,
        )
        gross_recoveries = np.where(is_cured, cure_recovery_amount, gross_recoveries)

        direct_costs = (ead * rng.uniform(0.02, 0.06, n)).round(2)
        workout_months = resolution_timeline_months.copy()

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
            "lvr_at_origination":         lvr_at_origination.round(4).tolist(),
            "lvr_at_default":             lvr_at_default.round(4).tolist(),
            "dscr_at_origination":        dscr_at_origination.round(4).tolist(),
            "dscr_at_default":            dscr_at_default.tolist(),
            "wale_years":                 wale_years.tolist(),
            "vacancy_rate_pct":           vacancy_rate_pct.tolist(),
            "tenant_concentration_pct":   tenant_concentration_pct.tolist(),
            "refinance_outcome":          refinance_outcome.tolist(),
            "forced_sale_haircut_pct":    forced_sale_haircut.tolist(),
            "resolution_timeline_months": resolution_timeline_months.tolist(),
            "asset_class_cre":            asset_class_cre.tolist(),
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
            refinance_outcome = str(row["refinance_outcome"])

            total_days = workout_months * 30

            if is_cured:
                recovery_events = [(120, gross_rec)]
                cost_events = [(30, direct_cost)]
            elif refinance_outcome == "Successful":
                # Refinance: bulk recovery mid-workout
                recovery_events = [(int(total_days * 0.65), gross_rec)]
                cost_events = [
                    (60, round(direct_cost * 0.40, 2)),
                    (int(total_days * 0.65), round(direct_cost * 0.60, 2)),
                ]
            else:
                # Forced sale / receivership: staged recovery
                recovery_events = [
                    (int(total_days * 0.55), round(gross_rec * 0.60, 2)),
                    (total_days, round(gross_rec * 0.40, 2)),
                ]
                cost_events = [
                    (30, round(direct_cost * 0.20, 2)),
                    (int(total_days * 0.40), round(direct_cost * 0.50, 2)),
                    (total_days, round(direct_cost * 0.30, 2)),
                ]

            rows.extend(
                self._build_cashflow_rows(loan_id, default_date, recovery_events, cost_events)
            )

        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["loan_id", "cashflow_date", "cashflow_type",
                     "recovery_amount", "direct_costs", "indirect_costs"]
        )
