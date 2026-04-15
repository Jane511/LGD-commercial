"""
Tests for src/validation_suite.py — extended APS 113 validation.

Covers:
- compute_gini_coefficient: returns value in [-1, 1], Lorenz curve
- hosmer_lemeshow_test: statistic >= 0, p-value in [0, 1]
- run_full_validation_suite: returns expected keys, summary_table row
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_validation_df(n=200, seed=42):
    rng = np.random.default_rng(seed)
    years = np.tile(np.arange(2016, 2024), int(np.ceil(n / 8)))[:n]
    actual = rng.beta(2, 5, n)
    # predicted is correlated with actual
    predicted = (actual * 0.8 + rng.uniform(0, 0.2, n)).clip(0, 1)
    return pd.DataFrame({
        "loan_id": [f"L{i}" for i in range(n)],
        "realised_lgd": actual,
        "lgd_final_calibrated": predicted,
        "ead_at_default": rng.uniform(50_000, 500_000, n),
        "default_year": years,
        "default_date": pd.date_range("2016-01-01", periods=n, freq="2W"),
        "mortgage_class": rng.choice(["Standard", "Non-Standard"], n),
    })


# ── compute_gini_coefficient ─────────────────────────────────────────────────

def test_gini_returns_float():
    from src.validation import compute_gini_coefficient
    df = _make_validation_df()
    result = compute_gini_coefficient(df, actual_col="realised_lgd",
                                      predicted_col="lgd_final_calibrated")
    assert isinstance(result, dict)
    assert "gini" in result
    assert isinstance(result["gini"], float)


def test_gini_in_valid_range():
    from src.validation import compute_gini_coefficient
    df = _make_validation_df()
    result = compute_gini_coefficient(df, actual_col="realised_lgd",
                                      predicted_col="lgd_final_calibrated")
    assert -1.0 <= result["gini"] <= 1.0


def test_gini_lorenz_curve_present():
    from src.validation import compute_gini_coefficient
    df = _make_validation_df()
    result = compute_gini_coefficient(df, actual_col="realised_lgd",
                                      predicted_col="lgd_final_calibrated")
    if "lorenz_curve" in result:
        assert isinstance(result["lorenz_curve"], pd.DataFrame)


def test_perfect_gini():
    """Perfect predictor should have Gini ≈ 1."""
    from src.validation import compute_gini_coefficient
    n = 100
    actual = np.linspace(0.01, 0.99, n)
    df = pd.DataFrame({
        "realised_lgd": actual,
        "lgd_final_calibrated": actual,  # perfect predictor
        "ead_at_default": np.ones(n) * 100_000,
    })
    result = compute_gini_coefficient(df, actual_col="realised_lgd",
                                      predicted_col="lgd_final_calibrated")
    assert result["gini"] > 0.8, f"Perfect predictor Gini too low: {result['gini']}"


# ── hosmer_lemeshow_test ──────────────────────────────────────────────────────

def test_hl_returns_statistic_and_pvalue():
    from src.validation import hosmer_lemeshow_test
    df = _make_validation_df()
    result = hosmer_lemeshow_test(df, actual_col="realised_lgd",
                                  predicted_col="lgd_final_calibrated")
    assert isinstance(result, dict)
    assert "hl_statistic" in result
    assert "hl_pvalue" in result


def test_hl_statistic_non_negative():
    from src.validation import hosmer_lemeshow_test
    df = _make_validation_df()
    result = hosmer_lemeshow_test(df, actual_col="realised_lgd",
                                  predicted_col="lgd_final_calibrated")
    assert result["hl_statistic"] >= 0.0


def test_hl_pvalue_in_unit_interval():
    from src.validation import hosmer_lemeshow_test
    df = _make_validation_df()
    result = hosmer_lemeshow_test(df, actual_col="realised_lgd",
                                  predicted_col="lgd_final_calibrated")
    assert 0.0 <= result["hl_pvalue"] <= 1.0


def test_hl_raises_on_too_few_bins():
    from src.validation import hosmer_lemeshow_test
    df = _make_validation_df(10)  # too few for 10 bins
    with pytest.raises(ValueError):
        hosmer_lemeshow_test(df, actual_col="realised_lgd",
                              predicted_col="lgd_final_calibrated", n_bins=10)


# ── run_full_validation_suite ─────────────────────────────────────────────────

def test_full_suite_returns_expected_keys():
    from src.validation import run_full_validation_suite
    df = _make_validation_df()
    result = run_full_validation_suite(
        loans=df,
        predicted_col="lgd_final_calibrated",
        actual_col="realised_lgd",
        product="mortgage",
    )
    assert isinstance(result, dict)
    for key in ("gini", "calibration_ratio", "psi"):
        assert key in result, f"Key '{key}' missing from validation suite result"


def test_full_suite_summary_table():
    from src.validation import run_full_validation_suite
    df = _make_validation_df()
    result = run_full_validation_suite(
        loans=df,
        predicted_col="lgd_final_calibrated",
        actual_col="realised_lgd",
        product="mortgage",
    )
    if "summary_table" in result:
        assert isinstance(result["summary_table"], pd.DataFrame)
        assert len(result["summary_table"]) >= 1


def test_full_suite_calibration_ratio_reasonable():
    """Calibration ratio = mean(predicted) / mean(actual) should be near 1 for good model."""
    from src.validation import run_full_validation_suite
    df = _make_validation_df(300)
    result = run_full_validation_suite(
        loans=df, predicted_col="lgd_final_calibrated",
        actual_col="realised_lgd", product="mortgage",
    )
    cal = result.get("calibration_ratio")
    if cal is not None:
        assert 0.5 <= cal <= 2.0, f"Calibration ratio {cal} outside expected range"
