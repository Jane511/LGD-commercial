from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.commercial_data_controls import (  # noqa: E402
    assign_framework_segment,
    run_commercial_data_controls,
)
from src.data_generation import generate_commercial_data  # noqa: E402


def _check_row(df: pd.DataFrame, check_name: str) -> pd.Series:
    row = df.loc[df["check_name"] == check_name]
    assert len(row) == 1, f"missing check {check_name}"
    return row.iloc[0]


def test_commercial_data_controls_pass_on_generated_data():
    loans, cashflows = generate_commercial_data(n_loans=120, seed=43)
    loans = loans.copy()
    loans["framework_segment"] = assign_framework_segment(loans)

    checks = run_commercial_data_controls(loans, cashflows, segment_col="framework_segment")
    required = {
        "required_fields_present",
        "no_impossible_balances",
        "ead_not_below_drawn_balance",
        "drawn_not_above_facility_limit",
        "segment_assignment_consistency",
        "recoveries_not_before_default_date",
    }
    assert required.issubset(set(checks["check_name"].astype(str)))
    assert checks["passed"].all()


def test_commercial_data_controls_detect_balance_and_segment_issues():
    loans, cashflows = generate_commercial_data(n_loans=50, seed=43)
    loans = loans.copy()
    loans["framework_segment"] = assign_framework_segment(loans)
    loans.loc[loans.index[0], "ead"] = loans.loc[loans.index[0], "drawn_balance"] - 1.0
    loans.loc[loans.index[1], "framework_segment"] = "Invalid Segment"

    checks = run_commercial_data_controls(loans, cashflows, segment_col="framework_segment")
    ead_check = _check_row(checks, "ead_not_below_drawn_balance")
    seg_check = _check_row(checks, "segment_assignment_consistency")

    assert not bool(ead_check["passed"])
    assert int(ead_check["failed_rows"]) >= 1
    assert not bool(seg_check["passed"])
    assert int(seg_check["failed_rows"]) >= 1


def test_commercial_data_controls_detect_probability_and_haircut_range_issues():
    loans, _ = generate_commercial_data(n_loans=40, seed=43)
    loans = loans.copy()
    loans["framework_segment"] = assign_framework_segment(loans)
    loans["claim_probability_base"] = 0.4
    loans["remarketing_discount_pct"] = 0.2
    loans.loc[loans.index[0], "claim_probability_base"] = 1.4
    loans.loc[loans.index[1], "remarketing_discount_pct"] = -0.1

    checks = run_commercial_data_controls(
        loans,
        None,
        segment_col="framework_segment",
        extra_haircut_cols=["remarketing_discount_pct"],
    )
    prob_check = _check_row(checks, "range_check_claim_probability_base")
    haircut_check = _check_row(checks, "range_check_remarketing_discount_pct")

    assert not bool(prob_check["passed"])
    assert int(prob_check["failed_rows"]) >= 1
    assert not bool(haircut_check["passed"])
    assert int(haircut_check["failed_rows"]) >= 1


def test_commercial_data_controls_missing_required_fields():
    df = pd.DataFrame({"loan_id": [1], "ead": [100.0]})
    checks = run_commercial_data_controls(df, None)
    req = _check_row(checks, "required_fields_present")
    assert not bool(req["passed"])
