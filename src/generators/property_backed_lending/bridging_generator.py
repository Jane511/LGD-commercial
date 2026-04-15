"""
Synthetic Bridging Finance Workout Data Generator.

Produces 10-year (2014-2024) synthetic workout histories for bridging finance
loan defaults (short-term property bridging, development bridging, equity
release bridging), calibrated to Australian non-bank and bank lender practice.

Key calibration references:
  - APS 113 Attachment A: minimum 5-year observation for non-mortgage exposures
    (this generator provides 10 years).
  - APS 113 s.50: exit risk is the dominant LGD driver for bridging; failed
    exits require collateral-based recovery under forced-sale conditions.
  - MFAA/CAFBA Bridging Finance Market Report (2022): refinance-dependent
    bridging exits are particularly vulnerable to credit tightening cycles.
  - RBA Monetary Policy Tightening (2022-2023): rising rates caused a material
    increase in bridging refinance failures as borrowers could not exit.
  - Basel III: short-term bridge facilities carry elevated LGD where exit
    strategy is speculative or dependent on illiquid secondary markets.

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


class BridgingWorkoutGenerator(BaseWorkoutGenerator):
    """
    Bridging finance workout data generator.

    Recovery mechanics:
      - Confirmed exit: 70-90% recovery (exit proceeds >= loan balance).
      - Speculative/failed exit: 30-60% (normal), 15-45% (downturn).
      - Failed exits occur when actual_exit_months > 1.5x planned OR
        exit_certainty_band = Speculative.
      - Refinance risk flag elevates LGD for Refinance exits outside Confirmed.
    """

    product_name = "bridging"
    min_records = 500

    _EAD_MEAN = 1_200_000.0
    _EAD_STD = 1_000_000.0
    _EAD_MIN = 200_000.0
    _EAD_MAX = 5_000_000.0

    def generate_loans(self) -> pd.DataFrame:
        n = self.n
        rng = self.rng

        loan_ids = [f"BRG-{i:04d}" for i in range(1, n + 1)]

        default_dates = _random_dates(DATE_RANGE_START, DATE_RANGE_END, n, rng)
        origination_dates = [
            d - pd.Timedelta(days=int(rng.uniform(90, 365 * 2)))
            for d in default_dates
        ]
        default_years = [d.year for d in default_dates]
        downturn_flags = np.array([1 if y in DOWNTURN_YEARS else 0 for y in default_years])

        ead = self._ead_array(self._EAD_MEAN, self._EAD_STD)
        ead = np.clip(ead, self._EAD_MIN, self._EAD_MAX)

        exit_type = rng.choice(
            ["Refinance", "Sale", "Development Completion", "Other"],
            n, p=[0.40, 0.35, 0.15, 0.10]
        )
        exit_certainty_band = rng.choice(
            ["Confirmed", "Likely", "Uncertain", "Speculative"],
            n, p=[0.25, 0.35, 0.30, 0.10]
        )

        # Refinance risk: Refinance exit but not Confirmed certainty
        refinance_risk_flag = np.where(
            (exit_type == "Refinance") & (exit_certainty_band != "Confirmed"),
            1, 0
        ).astype(int)

        valuation_volatility_proxy = rng.choice(
            ["Stable", "Moderate", "High"], n, p=[0.30, 0.45, 0.25]
        )

        planned_exit_months = rng.randint(3, 25, n).astype(int)
        actual_exit_multiplier = rng.uniform(0.80, 2.50, n)
        actual_exit_months = np.clip(
            (planned_exit_months * actual_exit_multiplier).astype(int), 1, 60
        )

        # Failed exit: actual > 1.5x planned OR Speculative certainty
        failed_exit_flag = np.where(
            (actual_exit_months > planned_exit_months * 1.5) |
            (exit_certainty_band == "Speculative"),
            1, 0
        ).astype(int)

        # Seniority: bridging loans are senior secured (registered mortgage)
        seniority = ["Senior Secured"] * n

        # Recovery logic
        gross_recoveries = np.zeros(n)
        for i in range(n):
            dt = downturn_flags[i]
            certainty = exit_certainty_band[i]
            failed = failed_exit_flag[i]

            if certainty == "Confirmed" and not failed:
                lo, hi = (0.60, 0.82) if dt else (0.70, 0.90)
            elif certainty in ("Confirmed", "Likely") and not failed:
                lo, hi = (0.45, 0.68) if dt else (0.55, 0.78)
            elif failed or certainty == "Speculative":
                lo, hi = (0.15, 0.45) if dt else (0.30, 0.60)
            else:
                lo, hi = (0.35, 0.60) if dt else (0.45, 0.70)

            gross_recoveries[i] = round(ead[i] * rng.uniform(lo, hi), 2)

        is_cured = rng.uniform(0, 1, n) < 0.10
        cure_recovery_amount = np.where(
            is_cured,
            (ead * rng.uniform(0.85, 1.00, n)).round(2),
            0.0,
        )
        gross_recoveries = np.where(is_cured, cure_recovery_amount, gross_recoveries)

        direct_costs = (ead * rng.uniform(0.02, 0.07, n)).round(2)

        workout_months = actual_exit_months + rng.randint(1, 7, n)
        workout_months = workout_months.astype(int)

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
            "exit_type":                  exit_type.tolist(),
            "exit_certainty_band":        exit_certainty_band.tolist(),
            "refinance_risk_flag":        refinance_risk_flag.tolist(),
            "valuation_volatility_proxy": valuation_volatility_proxy.tolist(),
            "planned_exit_months":        planned_exit_months.tolist(),
            "actual_exit_months":         actual_exit_months.tolist(),
            "failed_exit_flag":           failed_exit_flag.tolist(),
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
            failed_exit = int(row["failed_exit_flag"]) == 1
            exit_type = str(row["exit_type"])

            total_days = max(workout_months * 30, 30)

            if is_cured:
                recovery_events = [(60, gross_rec)]
                cost_events = [(30, direct_cost)]
            elif not failed_exit and exit_type in ("Sale", "Refinance"):
                # Clean exit: bulk settlement proceeds
                recovery_events = [(int(total_days * 0.80), gross_rec)]
                cost_events = [
                    (30, round(direct_cost * 0.35, 2)),
                    (int(total_days * 0.80), round(direct_cost * 0.65, 2)),
                ]
            else:
                # Failed exit: enforced sale or mortgagee-in-possession
                recovery_events = [
                    (int(total_days * 0.60), round(gross_rec * 0.65, 2)),
                    (total_days, round(gross_rec * 0.35, 2)),
                ]
                cost_events = [
                    (30, round(direct_cost * 0.25, 2)),
                    (int(total_days * 0.40), round(direct_cost * 0.45, 2)),
                    (total_days, round(direct_cost * 0.30, 2)),
                ]

            rows.extend(
                self._build_cashflow_rows(loan_id, default_date, recovery_events, cost_events)
            )

        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["loan_id", "cashflow_date", "cashflow_type",
                     "recovery_amount", "direct_costs", "indirect_costs"]
        )
