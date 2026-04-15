"""
Synthetic Residential Mortgage Workout Data Generator.

Produces 10-year (2014-2024) synthetic workout histories for residential
mortgage defaults, calibrated to Australian Big-4 IRB bank experience.

Key calibration references:
  - APS 113 Attachment A: minimum 7-year observation period for residential
    mortgage (this generator provides 10 years).
  - APS 113 s.32: LIP costs handled by BaseWorkoutGenerator.
  - RBA Financial Stability Review 2020-2021: COVID stress period recovery
    distributions show significant deterioration in housing markets.
  - APRA Discussion Paper — Revisions to APS 113 (Dec 2022): LGD floors and
    collateral haircut requirements for residential exposures.

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


class MortgageWorkoutGenerator(BaseWorkoutGenerator):
    """
    Residential mortgage workout data generator.

    Recovery mechanics:
      - Cured loans (25% base, 15% downturn): recover 90-105% of EAD.
      - Non-cured: recovery driven by property sale net of a sale discount;
        discount wider in downturn (15-35%) vs normal (8-25%).
      - LMI applies on LVR > 80% loans and caps net loss to lender.

    APS 113 Attachment A minimum: 7 years (this provides 10 years, 2014-2024).
    """

    product_name = "mortgage"
    min_records = 1000

    # EAD distribution: lognormal, mean ~$600k, range $150k-$2.5m
    _EAD_MEAN = 600_000.0
    _EAD_STD = 350_000.0
    _EAD_MIN = 150_000.0
    _EAD_MAX = 2_500_000.0

    def generate_loans(self) -> pd.DataFrame:
        """Generate residential mortgage defaulted loan records."""
        n = self.n
        rng = self.rng

        # ------------------------------------------------------------------ #
        # Loan identifiers
        # ------------------------------------------------------------------ #
        loan_ids = [f"MTG-{i:04d}" for i in range(1, n + 1)]

        # ------------------------------------------------------------------ #
        # Dates — default dates spread across 2014-2024
        # ------------------------------------------------------------------ #
        default_dates = _random_dates(DATE_RANGE_START, DATE_RANGE_END, n, rng)
        origination_dates = [
            d - pd.Timedelta(days=int(rng.uniform(365, 365 * 7)))
            for d in default_dates
        ]
        default_years = [d.year for d in default_dates]
        downturn_flags = np.array([1 if y in DOWNTURN_YEARS else 0 for y in default_years])

        # ------------------------------------------------------------------ #
        # EAD — lognormal, clipped to realistic range
        # ------------------------------------------------------------------ #
        ead = self._ead_array(self._EAD_MEAN, self._EAD_STD)
        ead = np.clip(ead, self._EAD_MIN, self._EAD_MAX)

        # ------------------------------------------------------------------ #
        # Mortgage classification
        # ------------------------------------------------------------------ #
        mortgage_class = rng.choice(
            ["Standard", "Non-Standard"], n, p=[0.75, 0.25]
        )
        property_type = rng.choice(
            ["House", "Unit", "Townhouse"], n, p=[0.55, 0.35, 0.10]
        )
        occupancy_type = rng.choice(
            ["Owner-Occupier", "Investor"], n, p=[0.65, 0.35]
        )

        # ------------------------------------------------------------------ #
        # LVR at origination and drift to default
        # ------------------------------------------------------------------ #
        lvr_at_origination = rng.uniform(0.50, 0.95, n)
        lmi_flag = (lvr_at_origination > 0.80).astype(int)
        lvr_drift = rng.uniform(-0.10, 0.20, n)
        lvr_at_default = np.clip(lvr_at_origination + lvr_drift, 0.10, 1.50)

        # ------------------------------------------------------------------ #
        # Arrears and foreclosure timeline
        # ------------------------------------------------------------------ #
        arrears_months = rng.randint(3, 37, n)
        # Foreclosure takes longer in downturn due to court backlogs
        base_foreclosure = rng.randint(6, 25, n)
        foreclosure_timeline = np.where(
            downturn_flags == 1,
            np.clip(base_foreclosure + rng.randint(3, 9, n), 6, 30),
            base_foreclosure,
        ).astype(int)

        # ------------------------------------------------------------------ #
        # Cure logic
        # ------------------------------------------------------------------ #
        cure_prob = np.where(downturn_flags == 1, 0.15, 0.25)
        cure_draws = rng.uniform(0, 1, n)
        is_cured = cure_draws < cure_prob

        # ------------------------------------------------------------------ #
        # Recovery logic
        # ------------------------------------------------------------------ #
        gross_recoveries = np.zeros(n)
        cure_recovery_amount = np.zeros(n)

        for i in range(n):
            if is_cured[i]:
                # Cured: recover 90-105% of EAD
                cure_recovery_amount[i] = round(
                    ead[i] * rng.uniform(0.90, 1.05), 2
                )
                gross_recoveries[i] = cure_recovery_amount[i]
            else:
                # Property sale recovery
                # Property value estimated from LVR at origination
                property_value = ead[i] / max(lvr_at_origination[i], 0.01)
                if downturn_flags[i] == 1:
                    sale_discount = rng.uniform(0.15, 0.35)
                else:
                    sale_discount = rng.uniform(0.08, 0.25)
                net_sale_proceeds = property_value * (1.0 - sale_discount)
                # Recovery = min(net sale proceeds, EAD)
                gross_recoveries[i] = round(
                    min(net_sale_proceeds, ead[i] * 1.05), 2
                )

        # Direct costs: legal, agent commission, preservation fees (3-8% EAD)
        direct_cost_pct = rng.uniform(0.03, 0.08, n)
        direct_costs = (ead * direct_cost_pct).round(2)

        # ------------------------------------------------------------------ #
        # Resolution dates and workout months
        # ------------------------------------------------------------------ #
        workout_months_arr = np.where(
            is_cured,
            rng.randint(3, 12, n),          # cured: shorter resolution
            foreclosure_timeline,
        ).astype(int)

        resolution_dates = [
            dd + pd.Timedelta(days=int(wm * 30.5))
            for dd, wm in zip(default_dates, workout_months_arr)
        ]

        # ------------------------------------------------------------------ #
        # Seniority — residential mortgages are senior secured
        # ------------------------------------------------------------------ #
        seniority = ["Senior Secured"] * n

        df = pd.DataFrame({
            "loan_id":                    loan_ids,
            "ead_at_default":             ead,
            "origination_date":           origination_dates,
            "default_date":               default_dates,
            "gross_recoveries":           gross_recoveries,
            "direct_costs":               direct_costs,
            "is_cured":                   is_cured.tolist(),
            "cure_recovery_amount":       cure_recovery_amount,
            "resolution_date":            resolution_dates,
            "workout_months":             workout_months_arr.tolist(),
            "seniority":                  seniority,
            # Mortgage-specific
            "mortgage_class":             mortgage_class.tolist(),
            "lvr_at_origination":         lvr_at_origination.round(4).tolist(),
            "lvr_at_default":             lvr_at_default.round(4).tolist(),
            "lmi_flag":                   lmi_flag.tolist(),
            "occupancy_type":             occupancy_type.tolist(),
            "arrears_months_at_default":  arrears_months.tolist(),
            "foreclosure_timeline_months": foreclosure_timeline.tolist(),
            "cure_flag":                  is_cured.astype(int).tolist(),
            "property_type":              property_type.tolist(),
        })

        return df

    def generate_cashflows(self, loans: pd.DataFrame) -> pd.DataFrame:
        """Generate cashflow events for each defaulted mortgage."""
        rows = []

        for _, row in loans.iterrows():
            loan_id = row["loan_id"]
            default_date = pd.Timestamp(row["default_date"])
            ead = float(row["ead_at_default"])
            is_cured = bool(row["is_cured"])
            gross_rec = float(row["gross_recoveries"])
            direct_cost = float(row["direct_costs"])
            workout_months = int(row["workout_months"])

            if is_cured:
                # Cured: single recovery event early in workout
                recovery_day = int(self.rng.uniform(30, 90))
                recovery_events = [(recovery_day, gross_rec)]
                cost_day = int(self.rng.uniform(10, 30))
                cost_events = [(cost_day, direct_cost * 0.3)]
            else:
                # Non-cured: recovery arrives at foreclosure completion
                # Split into partial sale proceeds + final settlement
                total_days = workout_months * 30
                mid_day = int(total_days * 0.6)
                final_day = total_days

                # Interim proceeds (agent fees paid upfront)
                recovery_events = [
                    (mid_day, round(gross_rec * 0.85, 2)),
                    (final_day, round(gross_rec * 0.15, 2)),
                ]
                # Costs spread across workout
                cost_events = [
                    (30, round(direct_cost * 0.25, 2)),          # early legal
                    (mid_day, round(direct_cost * 0.50, 2)),     # agent/valuation
                    (final_day, round(direct_cost * 0.25, 2)),   # settlement
                ]

            rows.extend(
                self._build_cashflow_rows(
                    loan_id, default_date, recovery_events, cost_events
                )
            )

        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["loan_id", "cashflow_date", "cashflow_type",
                     "recovery_amount", "direct_costs", "indirect_costs"]
        )
