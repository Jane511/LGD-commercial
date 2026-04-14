"""
Tests for src/regime_classifier.py — economic regime classification.

Covers:
- classify_economic_regime: synthetic fallback, regime labels, data_source column
- assign_regime_to_workouts: join, is_downturn_period column
- COVID years (2020-2021) classified as severe_stress
- upstream_first mode gracefully falls back when file absent
"""
from __future__ import annotations

import pandas as pd
import pytest


def test_classify_returns_dataframe():
    from src.regime_classifier import classify_economic_regime
    result = classify_economic_regime()
    assert isinstance(result, pd.DataFrame)
    assert "year" in result.columns
    assert "regime" in result.columns


def test_classify_covers_2014_to_2024():
    from src.regime_classifier import classify_economic_regime
    result = classify_economic_regime()
    assert result["year"].min() <= 2014
    assert result["year"].max() >= 2024


def test_classify_valid_regime_labels():
    from src.regime_classifier import classify_economic_regime
    result = classify_economic_regime()
    valid = {"expansion", "mild_stress", "moderate_stress", "severe_stress"}
    assert set(result["regime"].unique()).issubset(valid), \
        f"Unexpected regime labels: {set(result['regime'].unique()) - valid}"


def test_covid_years_classified_as_severe_stress():
    """2020 and 2021 must be severe_stress — APS 113 requires downturn years in sample."""
    from src.regime_classifier import classify_economic_regime
    result = classify_economic_regime()
    covid = result[result["year"].isin([2020, 2021])]
    assert not covid.empty, "2020-2021 must appear in regime classification"
    assert (covid["regime"] == "severe_stress").all(), \
        f"COVID years not all severe_stress: {covid[['year', 'regime']]}"


def test_data_source_column_present():
    from src.regime_classifier import classify_economic_regime
    result = classify_economic_regime()
    assert "data_source" in result.columns
    assert result["data_source"].iloc[0] in ("rba_abs_real", "synthetic")


def test_is_downturn_period_column():
    from src.regime_classifier import classify_economic_regime
    result = classify_economic_regime()
    assert "is_downturn_period" in result.columns
    # 2020-2021 should be marked as downturn
    downturn_years = result[result["is_downturn_period"] == True]["year"].tolist()
    assert 2020 in downturn_years or 2021 in downturn_years, \
        "No downturn years found — APS 113 s.43 requires at least one stress period"


def test_upstream_first_falls_back_to_synthetic():
    """When upstream parquet doesn't exist, must gracefully use synthetic."""
    from src.regime_classifier import classify_economic_regime
    result = classify_economic_regime(
        upstream_parquet_path="/nonexistent/macro_regime_flags.parquet",
        method="upstream_first",
    )
    assert isinstance(result, pd.DataFrame)
    assert result["data_source"].iloc[0] == "synthetic"


def test_assign_regime_to_workouts():
    from src.regime_classifier import classify_economic_regime, assign_regime_to_workouts
    import numpy as np

    regimes = classify_economic_regime()
    rng = np.random.default_rng(0)
    years = rng.choice(range(2015, 2024), 50)
    loans = pd.DataFrame({
        "loan_id": range(50),
        "default_year": years,
        "realised_lgd": rng.uniform(0.1, 0.6, 50),
    })
    result = assign_regime_to_workouts(loans, regimes)
    assert "regime" in result.columns
    assert "is_downturn_period" in result.columns
    assert len(result) == len(loans)


def test_at_least_one_downturn_year_in_synthetic():
    """Synthetic data must include at least one downturn year to pass APS 113 s.43."""
    from src.regime_classifier import classify_economic_regime
    result = classify_economic_regime()
    n_downturn = result["is_downturn_period"].sum()
    assert n_downturn >= 1, \
        "No downturn years in synthetic regime data — APS 113 s.43 requires severe stress period"
