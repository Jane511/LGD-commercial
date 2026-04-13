from __future__ import annotations

import numpy as np
import pandas as pd


COMMERCIAL_REQUIRED_FIELDS = [
    "loan_id",
    "facility_type",
    "security_type",
    "facility_limit",
    "drawn_balance",
    "ead",
    "default_date",
]


def assign_framework_segment(df: pd.DataFrame) -> pd.Series:
    """
    Standard commercial framework segment mapping used across notebooks.
    """
    x = df.copy()
    return pd.Series(
        np.select(
            [
                x["security_type"].eq("PPSR - Receivables"),
                x["security_type"].eq("PPSR - P&E"),
                x["facility_type"].isin(["Overdraft", "Revolving Credit"])
                & x["security_type"].eq("GSR Only"),
                x["facility_type"].isin(["Overdraft", "Revolving Credit"]),
            ],
            [
                "Receivables / Invoice Finance",
                "Asset / Equipment Finance",
                "Trade / Contingent Facilities (Proxy)",
                "Overdraft / Revolver",
            ],
            default="SME / Middle-Market Term Lending",
        ),
        index=x.index,
        name="framework_segment",
    )


def run_commercial_data_controls(
    loans: pd.DataFrame,
    cashflows: pd.DataFrame | None = None,
    *,
    segment_col: str = "framework_segment",
    extra_probability_cols: list[str] | None = None,
    extra_haircut_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Bank-style data control checks for commercial framework modules.
    """
    work = loans.copy()
    rows: list[dict] = []

    missing = [c for c in COMMERCIAL_REQUIRED_FIELDS if c not in work.columns]
    rows.append(
        {
            "check_name": "required_fields_present",
            "passed": len(missing) == 0,
            "failed_rows": int(len(missing)),
            "detail": "none" if len(missing) == 0 else f"missing={missing}",
        }
    )

    if len(missing) > 0:
        return pd.DataFrame(rows)

    for col in ["facility_limit", "drawn_balance", "ead"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    impossible_bal = (
        work["facility_limit"].isna()
        | work["drawn_balance"].isna()
        | work["ead"].isna()
        | (work["facility_limit"] < 0)
        | (work["drawn_balance"] < 0)
        | (work["ead"] < 0)
    )
    rows.append(
        {
            "check_name": "no_impossible_balances",
            "passed": (~impossible_bal).all(),
            "failed_rows": int(impossible_bal.sum()),
            "detail": "facility_limit/drawn_balance/ead must be numeric and non-negative",
        }
    )

    ead_below_drawn = work["ead"] + 1e-9 < work["drawn_balance"]
    rows.append(
        {
            "check_name": "ead_not_below_drawn_balance",
            "passed": (~ead_below_drawn).all(),
            "failed_rows": int(ead_below_drawn.sum()),
            "detail": "ead >= drawn_balance",
        }
    )

    drawn_above_limit = work["drawn_balance"] - work["facility_limit"] > 1e-9
    rows.append(
        {
            "check_name": "drawn_not_above_facility_limit",
            "passed": (~drawn_above_limit).all(),
            "failed_rows": int(drawn_above_limit.sum()),
            "detail": "drawn_balance <= facility_limit",
        }
    )

    if segment_col in work.columns:
        expected = assign_framework_segment(work)
        inconsistent = expected.ne(work[segment_col].astype(str))
        rows.append(
            {
                "check_name": "segment_assignment_consistency",
                "passed": (~inconsistent).all(),
                "failed_rows": int(inconsistent.sum()),
                "detail": "framework segment matches standard mapping",
            }
        )

    prob_cols = [
        "claim_probability_base",
        "claim_probability_downturn",
        "conversion_factor_base",
        "conversion_factor_downturn",
        "eligible_receivables_pct",
        "ineligible_receivables_pct",
        "dilution_proxy",
        "cash_security_pct",
        "collateral_support_pct",
        "advance_rate",
        "secondary_market_liquidity",
    ]
    if extra_probability_cols:
        prob_cols.extend(extra_probability_cols)
    prob_cols = list(dict.fromkeys(prob_cols))

    for col in [c for c in prob_cols if c in work.columns]:
        bad = ~pd.to_numeric(work[col], errors="coerce").between(0, 1, inclusive="both")
        bad = bad.fillna(True)
        rows.append(
            {
                "check_name": f"range_check_{col}",
                "passed": (~bad).all(),
                "failed_rows": int(bad.sum()),
                "detail": f"{col} in [0,1]",
            }
        )

    haircut_cols = [
        "remarketing_discount_pct",
        "asset_haircut_proxy",
        "residual_balloon_pct",
    ]
    if extra_haircut_cols:
        haircut_cols.extend(extra_haircut_cols)
    haircut_cols = list(dict.fromkeys(haircut_cols))

    for col in [c for c in haircut_cols if c in work.columns]:
        bad = ~pd.to_numeric(work[col], errors="coerce").between(0, 1, inclusive="both")
        bad = bad.fillna(True)
        rows.append(
            {
                "check_name": f"range_check_{col}",
                "passed": (~bad).all(),
                "failed_rows": int(bad.sum()),
                "detail": f"{col} in [0,1]",
            }
        )

    if cashflows is not None and len(cashflows) > 0:
        cfs = cashflows.copy()
        if "loan_id" in cfs.columns and "cashflow_date" in cfs.columns:
            cfs["cashflow_date"] = pd.to_datetime(cfs["cashflow_date"], errors="coerce")
            merged = cfs.merge(
                work[["loan_id", "default_date"]].assign(
                    default_date=pd.to_datetime(work["default_date"], errors="coerce")
                ),
                on="loan_id",
                how="left",
            )
            invalid_dates = (
                merged["cashflow_date"].notna()
                & merged["default_date"].notna()
                & (merged["cashflow_date"] < merged["default_date"])
            )
            rows.append(
                {
                    "check_name": "recoveries_not_before_default_date",
                    "passed": (~invalid_dates).all(),
                    "failed_rows": int(invalid_dates.sum()),
                    "detail": "cashflow_date >= default_date",
                }
            )

    return pd.DataFrame(rows)
