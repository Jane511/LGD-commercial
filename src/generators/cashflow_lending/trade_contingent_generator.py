"""
Synthetic Trade Finance & Contingent Liability Workout Data Generator.

Produces 10-year (2014-2024) synthetic workout histories for trade finance
and contingent liability defaults (performance bonds, letters of credit,
guarantees), calibrated to Australian bank practice.

Key calibration references:
  - APS 113 Attachment A: minimum 5-year observation for non-mortgage exposures
    (this generator provides 10 years).
  - APS 113 s.60-62: Credit Conversion Factor (CCF) estimation for off-balance-
    sheet contingent exposures.
  - Basel III — Credit Risk Standard (BIS 2017): claim conversion rates for
    contingent instruments vary by instrument type and credit quality.
  - Australian Trade Finance Market Study (EY/AFIA 2021): performance bonds and
    L/Cs typically show high recovery where cash collateral backs the exposure.

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


class TradeContingentWorkoutGenerator(BaseWorkoutGenerator):
    """
    Trade finance and contingent liability workout data generator.

    Recovery mechanics:
      - Cash-backed (cash_collateral_pct > 0.5): 75-98% recovery.
      - Unsecured contingent: 20-55% (normal), 10-40% (downturn).
      - Drawn amount at default = contingent_exposure * claim_conversion_rate.
    """

    product_name = "trade_contingent"
    min_records = 500

    _EAD_MEAN = 400_000.0
    _EAD_STD = 380_000.0
    _EAD_MIN = 50_000.0
    _EAD_MAX = 2_000_000.0

    def generate_loans(self) -> pd.DataFrame:
        n = self.n
        rng = self.rng

        loan_ids = [f"TRD-{i:04d}" for i in range(1, n + 1)]

        default_dates = _random_dates(DATE_RANGE_START, DATE_RANGE_END, n, rng)
        origination_dates = [
            d - pd.Timedelta(days=int(rng.uniform(90, 365 * 4)))
            for d in default_dates
        ]
        default_years = [d.year for d in default_dates]
        downturn_flags = np.array([1 if y in DOWNTURN_YEARS else 0 for y in default_years])

        ead = self._ead_array(self._EAD_MEAN, self._EAD_STD)
        ead = np.clip(ead, self._EAD_MIN, self._EAD_MAX)

        facility_type = rng.choice(
            ["Performance Bond", "Letter of Credit", "Trade Finance", "Guarantee"],
            n, p=[0.25, 0.30, 0.30, 0.15]
        )

        contingent_exposure = (ead * rng.uniform(0.80, 1.20, n)).round(2)
        claim_conversion_rate = rng.uniform(0.30, 1.00, n)
        drawn_at_default = (contingent_exposure * claim_conversion_rate).round(2)

        tenor_months = rng.randint(3, 49, n).astype(int)
        cash_collateral_pct = rng.uniform(0.0, 0.80, n)

        # Recovery months post claim
        recovery_months_post_claim = rng.randint(3, 25, n).astype(int)
        recovery_months_post_claim = np.where(
            downturn_flags == 1,
            np.clip(recovery_months_post_claim + rng.randint(2, 7, n), 3, 30),
            recovery_months_post_claim,
        ).astype(int)

        # Seniority
        seniority = np.where(
            cash_collateral_pct > 0.50, "Senior Secured", "Unsecured"
        ).tolist()

        # Recovery logic
        gross_recoveries = np.zeros(n)
        for i in range(n):
            dt = downturn_flags[i]
            cash_backed = cash_collateral_pct[i] > 0.50

            if cash_backed:
                # Cash collateral provides near-full recovery
                rec_rate = rng.uniform(0.75, 0.98)
            else:
                lo, hi = (0.10, 0.40) if dt else (0.20, 0.55)
                rec_rate = rng.uniform(lo, hi)

            gross_recoveries[i] = round(drawn_at_default[i] * rec_rate, 2)

        # Cure: very rare for contingent claims
        is_cured = rng.uniform(0, 1, n) < 0.05
        cure_recovery_amount = np.where(
            is_cured,
            (ead * rng.uniform(0.85, 0.98, n)).round(2),
            0.0,
        )
        gross_recoveries = np.where(is_cured, cure_recovery_amount, gross_recoveries)

        direct_costs = (drawn_at_default * rng.uniform(0.01, 0.04, n)).round(2)

        workout_months = recovery_months_post_claim.copy()
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
            "facility_type":                facility_type.tolist(),
            "contingent_exposure":          contingent_exposure.tolist(),
            "claim_conversion_rate":        claim_conversion_rate.round(4).tolist(),
            "drawn_at_default":             drawn_at_default.tolist(),
            "tenor_months":                 tenor_months.tolist(),
            "cash_collateral_pct":          cash_collateral_pct.round(4).tolist(),
            "recovery_months_post_claim":   recovery_months_post_claim.tolist(),
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
            cash_collateral_pct = float(row["cash_collateral_pct"])
            is_cured = bool(row["is_cured"])

            total_days = max(workout_months * 30, 30)

            if is_cured:
                recovery_events = [(60, gross_rec)]
                cost_events = [(30, direct_cost)]
            elif cash_collateral_pct > 0.50:
                # Cash-backed: quick realisation
                recovery_events = [(int(total_days * 0.30), gross_rec)]
                cost_events = [(15, direct_cost)]
            else:
                # Unsecured contingent: slower, uncertain
                recovery_events = [
                    (int(total_days * 0.60), round(gross_rec * 0.80, 2)),
                    (total_days, round(gross_rec * 0.20, 2)),
                ]
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
