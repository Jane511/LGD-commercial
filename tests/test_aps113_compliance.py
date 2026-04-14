"""
Tests for src/aps113_compliance.py — APS 113 compliance map.

Covers:
- generate_compliance_map: correct shape, valid status values, all products
- validate_observation_periods: 10-year window passes, short window fails
- export_compliance_map: file written
- Status 'met' for s.32, s.37, s.44 (always met in this framework)
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest


def test_generate_compliance_map_returns_dataframe():
    from src.aps113_compliance import generate_compliance_map
    result = generate_compliance_map()
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_compliance_map_has_required_columns():
    from src.aps113_compliance import generate_compliance_map
    result = generate_compliance_map()
    for col in ("product", "section_ref", "requirement", "status"):
        assert col in result.columns, f"Missing column: {col}"


def test_compliance_map_valid_status_values():
    from src.aps113_compliance import generate_compliance_map
    result = generate_compliance_map()
    valid = {"met", "partial", "not_met", "not_applicable"}
    assert set(result["status"].unique()).issubset(valid), \
        f"Unexpected status values: {set(result['status'].unique()) - valid}"


def test_compliance_map_covers_all_products():
    from src.aps113_compliance import generate_compliance_map, ALL_PRODUCTS
    result = generate_compliance_map()
    covered = set(result["product"].unique())
    assert set(ALL_PRODUCTS).issubset(covered), \
        f"Missing products: {set(ALL_PRODUCTS) - covered}"


def test_s32_s37_s44_always_met():
    """These sections are structurally met by this implementation."""
    from src.aps113_compliance import generate_compliance_map
    result = generate_compliance_map()
    for section in ("s.32", "s.37", "s.44 / Att A"):
        sec_rows = result[result["section_ref"] == section]
        if not sec_rows.empty:
            assert (sec_rows["status"] == "met").all(), \
                f"{section} should be 'met' but got: {sec_rows['status'].unique()}"


def test_compliance_map_with_calibration_results():
    """Providing calibration results should flip partial → met for s.43, s.46, s.58."""
    from src.aps113_compliance import generate_compliance_map
    mock_cal = {"mortgage": {"long_run_lgd_by_segment": True, "calibration_steps": True}}
    result = generate_compliance_map(
        calibration_results=mock_cal,
        products=["mortgage"],
    )
    s43_rows = result[result["section_ref"] == "s.43"]
    if not s43_rows.empty:
        assert (s43_rows["status"] == "met").all()


def test_validate_observation_periods_passes_10yr():
    from src.aps113_compliance import validate_observation_periods
    dates = pd.date_range("2014-01-01", "2024-01-01", periods=200)
    loans = pd.DataFrame({"default_date": dates})
    result = validate_observation_periods(loans, product="mortgage")
    assert result["compliant"] is True
    assert result["n_years"] >= 9.9


def test_validate_observation_periods_fails_short():
    from src.aps113_compliance import validate_observation_periods
    dates = pd.date_range("2022-01-01", "2023-12-31", periods=50)
    loans = pd.DataFrame({"default_date": dates})
    result = validate_observation_periods(loans, product="mortgage")
    assert result["compliant"] is False


def test_export_compliance_map_writes_file():
    from src.aps113_compliance import generate_compliance_map, export_compliance_map
    df = generate_compliance_map(products=["mortgage"])
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "compliance_map.csv"
        export_compliance_map(df, out)
        assert out.exists()
        loaded = pd.read_csv(out)
        assert len(loaded) == len(df)
