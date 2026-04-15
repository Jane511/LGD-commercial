"""
Synthetic Receivables Financing Workout Data Generator.

Produces 10-year (2014-2024) synthetic workout histories for receivables
financing (debtor finance / invoice discounting / factoring) defaults,
calibrated to Australian bank practice.

Key calibration references:
  - APS 113 Attachment A: minimum 5-year observation for non-mortgage exposures
    (this generator provides 10 years).
  - APS 113 s.43-45: eligible receivables and dilution risk are explicit
    components of LGD estimation for trade receivables facilities.
  - IFSA/AFIA Debtor Finance Industry Guidelines: collections control flag
    materially improves recovery outcomes.
  - COVID-19 impact on SME receivables (RBA Bulletin, 2021): dilution rates
    spiked 2-3x in sectors most affected by lockdowns.

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


class ReceivablesWorkoutGenerator(BaseWorkoutGenerator):
    """
    Receivables financing workout data generator.

    Recovery mechanics:
      - Full recourse + strong collections control: 60-85% recovery.
      - Non-recourse or no collections control: 30-65% recovery.
      - Dilution reduces effective recovery by dilution_rate * eligible_pct.
      - Downturn: dilution rates elevated; recoveries compressed 10-20pp.
    """

    product_name = "receivables"
    min_records = 500

    _EAD_MEAN = 500_000.0
    _EAD_STD = 450_000.0
    _EAD_MIN = 100_000.0
    _EAD_MAX = 3_000_000.0

    def generate_loans(self) -> pd.DataFrame:
        n = self.n
        rng = self.rng

        loan_ids = [f"REC-{i:04d}" for i in range(1, n + 1)]

        default_dates = _random_dates(DATE_RANGE_START, DATE_RANGE_END, n, rng)
        origination_dates = [
            d - pd.Timedelta(days=int(rng.uniform(180, 365 * 4)))
            for d in default_dates
        ]
        default_years = [d.year for d in default_dates]
        downturn_flags = np.array([1 if y in DOWNTURN_YEARS else 0 for y in default_years])

        ead = self._ead_array(self._EAD_MEAN, self._EAD_STD)
        ead = np.clip(ead, self._EAD_MIN, self._EAD_MAX)

        # Receivables characteristics
        eligible_receivables_pct = rng.uniform(0.50, 0.95, n)
        # Dilution higher in downturn
        dilution_rate = np.where(
            downturn_flags == 1,
            rng.uniform(0.05, 0.20, n),
            rng.uniform(0.02, 0.15, n),
        )
        avg_debtor_age_days = rng.randint(30, 91, n)
        debtor_concentration_pct = rng.uniform(0.10, 0.60, n)

        # Collections control: 60% probability lender controls collections
        collections_control_flag = (rng.uniform(0, 1, n) < 0.60).astype(int)
        advance_rate_pct = rng.uniform(0.70, 0.90, n)
        recourse_flag = (rng.uniform(0, 1, n) < 0.40).astype(int)

        # Seniority: receivables are typically senior claims against debtor pool
        seniority = ["Senior Secured"] * n

        # Recovery logic
        gross_recoveries = np.zeros(n)
        for i in range(n):
            dt = downturn_flags[i]
            full_recourse = recourse_flag[i] == 1
            strong_collections = collections_control_flag[i] == 1

            if full_recourse and strong_collections:
                lo, hi = (0.45, 0.70) if dt else (0.60, 0.85)
            elif full_recourse or strong_collections:
                lo, hi = (0.30, 0.55) if dt else (0.40, 0.70)
            else:
                lo, hi = (0.15, 0.40) if dt else (0.30, 0.65)

            # Apply dilution haircut to eligible pool
            dilution_haircut = dilution_rate[i] * eligible_receivables_pct[i]
            rec_rate = max(0.0, rng.uniform(lo, hi) - dilution_haircut)
            gross_recoveries[i] = round(ead[i] * rec_rate, 2)

        # Cure via borrower restructure: rare for receivables
        is_cured = rng.uniform(0, 1, n) < 0.08
        cure_recovery_amount = np.where(
            is_cured,
            (ead * rng.uniform(0.80, 0.95, n)).round(2),
            0.0,
        )
        gross_recoveries = np.where(is_cured, cure_recovery_amount, gross_recoveries)

        direct_costs = (ead * rng.uniform(0.01, 0.05, n)).round(2)

        workout_months = rng.randint(3, 18, n).astype(int)
        workout_months = np.where(
            downturn_flags == 1,
            np.clip(workout_months + rng.randint(1, 5, n), 3, 24),
            workout_months,
        ).astype(int)

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
            "eligible_receivables_pct":     eligible_receivables_pct.round(4).tolist(),
            "dilution_rate":                dilution_rate.round(4).tolist(),
            "avg_debtor_age_days":          avg_debtor_age_days.tolist(),
            "debtor_concentration_pct":     debtor_concentration_pct.round(4).tolist(),
            "collections_control_flag":     collections_control_flag.tolist(),
            "advance_rate_pct":             advance_rate_pct.round(4).tolist(),
            "recourse_flag":                recourse_flag.tolist(),
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

            total_days = workout_months * 30

            if is_cured:
                recovery_events = [(45, gross_rec)]
                cost_events = [(15, direct_cost)]
            else:
                # Receivables typically collected in rapid tranches
                recovery_events = [
                    (int(total_days * 0.40), round(gross_rec * 0.50, 2)),
                    (int(total_days * 0.70), round(gross_rec * 0.30, 2)),
                    (total_days, round(gross_rec * 0.20, 2)),
                ]
                cost_events = [
                    (15, round(direct_cost * 0.50, 2)),
                    (int(total_days * 0.50), round(direct_cost * 0.30, 2)),
                    (total_days, round(direct_cost * 0.20, 2)),
                ]

            rows.extend(
                self._build_cashflow_rows(loan_id, default_date, recovery_events, cost_events)
            )

        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["loan_id", "cashflow_date", "cashflow_type",
                     "recovery_amount", "direct_costs", "indirect_costs"]
        )
