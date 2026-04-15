"""
Synthetic Mezzanine Finance & Second Mortgage Workout Data Generator.

Produces 10-year (2014-2024) synthetic workout histories for mezzanine debt
and second mortgage loan defaults, calibrated to Australian bank and non-bank
lender practice.

Key calibration references:
  - APS 113 Attachment A: minimum 5-year observation for non-mortgage exposures
    (this generator provides 10 years).
  - APS 113 s.50: second-ranking security interests must reflect the priority
    claim of senior lenders; residual collateral coverage after senior balance
    is the primary recovery driver.
  - APRA Prudential Standard APS 112 s.30: mezzanine positions are treated
    as subordinated exposures with separate LGD floors.
  - Basel III — Credit Risk Standard (BIS 2017): for subordinated debt,
    collateral coverage is assessed net of senior claims; severe stress
    scenarios for mezzanine approach 80-100% LGD.
  - Carey & Gordy (2021) "Measuring Systemic Risk in the Finance Sector":
    mezzanine loss distributions are bimodal — near-full recovery when
    collateral value > (senior + mezz), near-total loss when it falls below.

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


class MezzSecondMortgageWorkoutGenerator(BaseWorkoutGenerator):
    """
    Mezzanine finance and second mortgage workout data generator.

    Recovery mechanics:
      - Residual collateral coverage (net of senior balance) is the key driver.
      - High coverage (>30%): 40-70% recovery.
      - Low coverage (<10%): approach 0-20% recovery.
      - Downturn: 20-35% additional collateral haircut → coverage collapses.
      - Recovery lag 12-60 months due to dual-tranche enforcement complexity.
    """

    product_name = "mezz_second_mortgage"
    min_records = 500

    _EAD_MEAN = 1_000_000.0
    _EAD_STD = 900_000.0
    _EAD_MIN = 100_000.0
    _EAD_MAX = 5_000_000.0

    def generate_loans(self) -> pd.DataFrame:
        n = self.n
        rng = self.rng

        loan_ids = [f"MEZ-{i:04d}" for i in range(1, n + 1)]

        default_dates = _random_dates(DATE_RANGE_START, DATE_RANGE_END, n, rng)
        origination_dates = [
            d - pd.Timedelta(days=int(rng.uniform(365, 365 * 5)))
            for d in default_dates
        ]
        default_years = [d.year for d in default_dates]
        downturn_flags = np.array([1 if y in DOWNTURN_YEARS else 0 for y in default_years])

        ead = self._ead_array(self._EAD_MEAN, self._EAD_STD)
        ead = np.clip(ead, self._EAD_MIN, self._EAD_MAX)

        # Collateral value at origination: larger than EAD (mezz lends on top of senior)
        collateral_value_at_origination = (ead * rng.uniform(2.0, 4.5, n)).round(2)

        # Collateral value at default: 70-110% of origination
        collateral_drift = rng.uniform(0.70, 1.10, n)
        collateral_value_at_default = (
            collateral_value_at_origination * collateral_drift
        ).round(2)

        # Downturn: apply additional 20-35% collateral haircut
        downturn_haircut = np.where(
            downturn_flags == 1,
            rng.uniform(0.20, 0.35, n),
            0.0,
        )
        collateral_value_at_default_stressed = np.where(
            downturn_flags == 1,
            collateral_value_at_default * (1.0 - downturn_haircut),
            collateral_value_at_default,
        ).round(2)

        # Senior balance at default: 40-75% of (stressed) collateral value
        senior_balance_pct = rng.uniform(0.40, 0.75, n)
        senior_balance_at_default = (
            collateral_value_at_default_stressed * senior_balance_pct
        ).round(2)

        # Mezz attachment point
        mezz_attachment_point_pct = (
            senior_balance_at_default / np.maximum(collateral_value_at_default_stressed, 1.0)
        ).round(4)

        # Residual coverage after senior and mezz
        residual_collateral_coverage_pct = np.maximum(
            0.0,
            (collateral_value_at_default_stressed - senior_balance_at_default - ead)
            / np.maximum(ead, 1.0)
        ).round(4)

        resolution_path = rng.choice(
            ["Cure", "Restructure", "Enforcement"],
            n, p=[0.15, 0.25, 0.60]
        )

        # Recovery lag: longer for enforcement; extended in downturn
        base_lag = np.where(
            resolution_path == "Enforcement",
            rng.randint(24, 61, n),
            np.where(
                resolution_path == "Restructure",
                rng.randint(12, 37, n),
                rng.randint(6, 19, n),
            )
        )
        recovery_lag_months = np.where(
            downturn_flags == 1,
            np.clip(base_lag + rng.randint(6, 13, n), 12, 72),
            base_lag,
        ).astype(int)

        # Seniority: always second mortgage or mezzanine debt
        seniority_choices = rng.choice(
            ["Second Mortgage", "Mezzanine Debt"], n, p=[0.55, 0.45]
        )

        # Recovery logic — bimodal distribution tied to residual coverage
        gross_recoveries = np.zeros(n)
        for i in range(n):
            coverage = residual_collateral_coverage_pct[i]
            path = resolution_path[i]

            if path == "Cure":
                rec_rate = rng.uniform(0.80, 1.00)
            elif path == "Restructure":
                if coverage > 0.20:
                    rec_rate = rng.uniform(0.45, 0.75)
                else:
                    rec_rate = rng.uniform(0.15, 0.45)
            else:
                # Enforcement: recovery is purely driven by residual coverage
                if coverage > 0.30:
                    rec_rate = rng.uniform(0.40, 0.70)
                elif coverage > 0.10:
                    rec_rate = rng.uniform(0.15, 0.40)
                else:
                    # Near-zero residual: approach total loss
                    rec_rate = rng.uniform(0.00, 0.20)

            gross_recoveries[i] = round(ead[i] * rec_rate, 2)

        is_cured = resolution_path == "Cure"
        cure_recovery_amount = np.where(
            is_cured,
            (ead * rng.uniform(0.88, 1.02, n)).round(2),
            0.0,
        )
        gross_recoveries = np.where(is_cured, cure_recovery_amount, gross_recoveries)

        direct_costs = (ead * rng.uniform(0.03, 0.08, n)).round(2)
        workout_months = recovery_lag_months.copy()

        resolution_dates = [
            dd + pd.Timedelta(days=int(wm * 30.5))
            for dd, wm in zip(default_dates, workout_months)
        ]

        df = pd.DataFrame({
            "loan_id":                          loan_ids,
            "ead_at_default":                   ead,
            "origination_date":                 origination_dates,
            "default_date":                     default_dates,
            "gross_recoveries":                 gross_recoveries,
            "direct_costs":                     direct_costs,
            "is_cured":                         is_cured.tolist(),
            "cure_recovery_amount":             cure_recovery_amount.tolist(),
            "resolution_date":                  resolution_dates,
            "workout_months":                   workout_months.tolist(),
            "seniority":                        seniority_choices.tolist(),
            # Product-specific
            "collateral_value_at_origination":  collateral_value_at_origination.tolist(),
            "collateral_value_at_default":      collateral_value_at_default.tolist(),
            "senior_balance_at_default":        senior_balance_at_default.tolist(),
            "mezz_attachment_point_pct":        mezz_attachment_point_pct.tolist(),
            "residual_collateral_coverage_pct": residual_collateral_coverage_pct.tolist(),
            "resolution_path":                  resolution_path.tolist(),
            "recovery_lag_months":              recovery_lag_months.tolist(),
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
            resolution_path = str(row["resolution_path"])
            coverage = float(row["residual_collateral_coverage_pct"])

            total_days = max(workout_months * 30, 60)

            if is_cured or resolution_path == "Cure":
                recovery_events = [(90, gross_rec)]
                cost_events = [(30, direct_cost)]
            elif resolution_path == "Restructure":
                # Restructure: staged repayment over workout period
                recovery_events = [
                    (int(total_days * 0.40), round(gross_rec * 0.45, 2)),
                    (int(total_days * 0.75), round(gross_rec * 0.35, 2)),
                    (total_days, round(gross_rec * 0.20, 2)),
                ]
                cost_events = [
                    (60, round(direct_cost * 0.30, 2)),
                    (int(total_days * 0.50), round(direct_cost * 0.40, 2)),
                    (total_days, round(direct_cost * 0.30, 2)),
                ]
            else:
                # Enforcement: lump-sum from collateral realisation (minus senior)
                if coverage > 0.20:
                    recovery_events = [
                        (int(total_days * 0.70), round(gross_rec * 0.80, 2)),
                        (total_days, round(gross_rec * 0.20, 2)),
                    ]
                else:
                    # Near-zero residual: single small recovery at the end
                    recovery_events = [(total_days, gross_rec)]
                cost_events = [
                    (60, round(direct_cost * 0.20, 2)),
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
