"""
Synthetic Commercial Cash Flow (Business Lending) Workout Data Generator.

Produces 10-year (2014-2024) synthetic workout histories for commercial
cash flow lending defaults, calibrated to Australian bank SME/corporate
lending practice.

Key calibration references:
  - APS 113 Attachment A: minimum 5-year observation for non-mortgage exposures
    (this generator provides 10 years).
  - APS 113 s.50: LGD estimation must reflect economic loss including direct and
    indirect costs of recovery.
  - APRA Information Paper — LGD Estimation (2022): commercial recovery rates
    vary significantly by security type and borrower size.
  - RBA Small Business Finance Report (2020): SME stress during COVID-19
    resulted in materially lower recovery rates on unsecured and partially
    secured facilities.

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


class CommercialCashflowWorkoutGenerator(BaseWorkoutGenerator):
    """
    Commercial cash flow lending workout data generator.

    Security-driven recovery tiers (APS 113 Attachment A):
      - Fully Secured:    55-80% (normal), 40-65% (downturn)
      - Partially Secured: 35-60% (normal), 25-45% (downturn)
      - Unsecured:        15-40% (normal),  5-25% (downturn)

    Revolving/OD facilities apply a realised CCF of 0.6-1.0.
    """

    product_name = "commercial_cashflow"
    min_records = 500

    _EAD_MEAN = 800_000.0
    _EAD_STD = 900_000.0
    _EAD_MIN = 50_000.0
    _EAD_MAX = 5_000_000.0

    def generate_loans(self) -> pd.DataFrame:
        n = self.n
        rng = self.rng

        loan_ids = [f"CCF-{i:04d}" for i in range(1, n + 1)]

        default_dates = _random_dates(DATE_RANGE_START, DATE_RANGE_END, n, rng)
        origination_dates = [
            d - pd.Timedelta(days=int(rng.uniform(365, 365 * 6)))
            for d in default_dates
        ]
        default_years = [d.year for d in default_dates]
        downturn_flags = np.array([1 if y in DOWNTURN_YEARS else 0 for y in default_years])

        ead = self._ead_array(self._EAD_MEAN, self._EAD_STD)
        ead = np.clip(ead, self._EAD_MIN, self._EAD_MAX)

        facility_type = rng.choice(
            ["Term Loan", "Revolving Credit", "Overdraft"],
            n, p=[0.40, 0.30, 0.30]
        )
        security_type = rng.choice(
            ["Unsecured", "Partially Secured", "Fully Secured"],
            n, p=[0.30, 0.40, 0.30]
        )
        borrower_size = rng.choice(
            ["SME", "Mid Corp", "Large Corp"], n, p=[0.50, 0.35, 0.15]
        )
        industry_risk_band = rng.choice(
            ["Low", "Medium", "High"], n, p=[0.30, 0.40, 0.30]
        )

        # Seniority derived from security type
        seniority = np.where(
            security_type == "Unsecured", "Unsecured", "Senior Secured"
        ).tolist()

        # CCF for revolving/OD; term loans always fully drawn
        realised_ccf = np.ones(n)
        for i in range(n):
            if facility_type[i] in ("Revolving Credit", "Overdraft"):
                realised_ccf[i] = rng.uniform(0.60, 1.00)

        # Undrawn at default
        undrawn_at_default = np.where(
            np.isin(facility_type, ["Revolving Credit", "Overdraft"]),
            rng.uniform(0.0, 0.5, n) * ead,
            0.0,
        ).round(2)

        # Recovery rates by security type and downturn
        gross_recoveries = np.zeros(n)
        for i in range(n):
            sec = security_type[i]
            dt = downturn_flags[i]
            if sec == "Fully Secured":
                lo, hi = (0.40, 0.65) if dt else (0.55, 0.80)
            elif sec == "Partially Secured":
                lo, hi = (0.25, 0.45) if dt else (0.35, 0.60)
            else:  # Unsecured
                lo, hi = (0.05, 0.25) if dt else (0.15, 0.40)
            rec_rate = rng.uniform(lo, hi)
            gross_recoveries[i] = round(ead[i] * rec_rate, 2)

        # Cure logic: small % cure via restructure
        cure_prob = np.where(downturn_flags == 1, 0.08, 0.15)
        is_cured = rng.uniform(0, 1, n) < cure_prob
        cure_recovery_amount = np.where(
            is_cured,
            (ead * rng.uniform(0.85, 1.00, n)).round(2),
            0.0,
        )
        # For cured loans, override gross recoveries
        gross_recoveries = np.where(is_cured, cure_recovery_amount, gross_recoveries)

        direct_costs = (ead * rng.uniform(0.02, 0.08, n)).round(2)

        # Workout months: longer for large, unsecured facilities
        base_wm = rng.randint(6, 36, n)
        workout_months = np.where(
            downturn_flags == 1, np.clip(base_wm + rng.randint(3, 9, n), 6, 48), base_wm
        ).astype(int)

        resolution_dates = [
            dd + pd.Timedelta(days=int(wm * 30.5))
            for dd, wm in zip(default_dates, workout_months)
        ]

        df = pd.DataFrame({
            "loan_id":                loan_ids,
            "ead_at_default":         ead,
            "origination_date":       origination_dates,
            "default_date":           default_dates,
            "gross_recoveries":       gross_recoveries,
            "direct_costs":           direct_costs,
            "is_cured":               is_cured.tolist(),
            "cure_recovery_amount":   cure_recovery_amount.tolist(),
            "resolution_date":        resolution_dates,
            "workout_months":         workout_months.tolist(),
            "seniority":              seniority,
            # Product-specific
            "facility_type":          facility_type.tolist(),
            "security_type":          security_type.tolist(),
            "borrower_size_proxy":    borrower_size.tolist(),
            "industry_risk_band":     industry_risk_band.tolist(),
            "undrawn_at_default":     undrawn_at_default.tolist(),
            "realised_ccf":           realised_ccf.round(4).tolist(),
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
                recovery_events = [(min(90, total_days), gross_rec)]
                cost_events = [(30, direct_cost)]
            else:
                # Two-tranche recovery: partial realisation then final
                recovery_events = [
                    (int(total_days * 0.5), round(gross_rec * 0.70, 2)),
                    (total_days, round(gross_rec * 0.30, 2)),
                ]
                cost_events = [
                    (30, round(direct_cost * 0.30, 2)),
                    (int(total_days * 0.5), round(direct_cost * 0.40, 2)),
                    (total_days, round(direct_cost * 0.30, 2)),
                ]

            rows.extend(
                self._build_cashflow_rows(loan_id, default_date, recovery_events, cost_events)
            )

        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["loan_id", "cashflow_date", "cashflow_type",
                     "recovery_amount", "direct_costs", "indirect_costs"]
        )
