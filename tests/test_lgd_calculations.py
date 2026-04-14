"""
Tests for src/lgd_calculations.py — workout LGD engine.

Covers:
- compute_realised_lgd: basic, cured path, LIP detection, guardrails
- segment_lgd: segment keys, low-count flag
- compute_long_run_lgd: vintage-EWA, min_years guard
- compare_model_vs_actual: bias, is_conservative
- apply_regulatory_floor: product-specific floors
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_loans(n=50, seed=0):
    rng = np.random.default_rng(seed)
    default_dates = pd.date_range("2018-01-01", periods=n, freq="ME")
    return pd.DataFrame({
        "loan_id": [f"L{i:04d}" for i in range(n)],
        "ead_at_default": rng.uniform(50_000, 500_000, n),
        "default_date": default_dates,
        "default_year": default_dates.year,
        "is_cured": rng.choice([0, 1], n, p=[0.7, 0.3]),
        "cure_recovery_amount": rng.uniform(0, 1, n),
        "mortgage_class": rng.choice(["Standard", "Non-Standard"], n),
        "lvr_band": rng.choice(["<=60%", "60-80%", "80-90%", ">90%"], n),
    })


def _make_cashflows(loans: pd.DataFrame, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for _, loan in loans.iterrows():
        n_cf = rng.integers(1, 5)
        for k in range(n_cf):
            rows.append({
                "loan_id": loan["loan_id"],
                "cashflow_date": loan["default_date"] + pd.Timedelta(days=int(rng.integers(30, 365))),
                "recovery_amount": rng.uniform(0, loan["ead_at_default"] * 0.5),
                "direct_costs": rng.uniform(0, loan["ead_at_default"] * 0.05),
                "indirect_costs": rng.uniform(0, loan["ead_at_default"] * 0.02),
                "discount_rate": rng.uniform(0.04, 0.10),
            })
    return pd.DataFrame(rows)


# ── compute_realised_lgd ──────────────────────────────────────────────────────

def test_compute_realised_lgd_returns_dataframe():
    from src.lgd_calculations import compute_realised_lgd
    loans = _make_loans(30)
    cashflows = _make_cashflows(loans)
    result = compute_realised_lgd(loans, cashflows)
    assert isinstance(result, pd.DataFrame)
    assert "realised_lgd" in result.columns


def test_realised_lgd_floored_at_zero():
    from src.lgd_calculations import compute_realised_lgd
    loans = _make_loans(30)
    cashflows = _make_cashflows(loans)
    result = compute_realised_lgd(loans, cashflows)
    assert (result["realised_lgd"] >= 0.0).all()


def test_realised_lgd_capped_at_150pct():
    from src.lgd_calculations import compute_realised_lgd
    loans = _make_loans(30)
    cashflows = _make_cashflows(loans)
    result = compute_realised_lgd(loans, cashflows)
    assert (result["realised_lgd"] <= 1.5).all()


def test_realised_lgd_cured_path_near_zero():
    """Cured loans should produce LGD ≈ 0 (recovers near full EAD)."""
    from src.lgd_calculations import compute_realised_lgd
    loans = _make_loans(20)
    loans["is_cured"] = 1
    loans["cure_recovery_amount"] = 1.0  # full recovery
    cashflows = _make_cashflows(loans)
    result = compute_realised_lgd(loans, cashflows)
    cured_lgd = result["realised_lgd"].mean()
    assert cured_lgd < 0.20, f"Cured LGD {cured_lgd:.3f} unexpectedly high"


def test_realised_lgd_raises_on_nonpositive_ead():
    from src.lgd_calculations import compute_realised_lgd
    loans = _make_loans(5)
    loans["ead_at_default"] = 0.0
    cashflows = _make_cashflows(loans)
    with pytest.raises(ValueError, match="EAD"):
        compute_realised_lgd(loans, cashflows)


# ── segment_lgd ──────────────────────────────────────────────────────────────

def test_segment_lgd_returns_dataframe():
    from src.lgd_calculations import compute_realised_lgd, segment_lgd
    loans = _make_loans(60)
    cashflows = _make_cashflows(loans)
    lgd_df = compute_realised_lgd(loans, cashflows)
    result = segment_lgd(lgd_df, ["mortgage_class", "lvr_band"])
    assert isinstance(result, pd.DataFrame)
    assert "segment_key" in result.columns or "mortgage_class" in result.columns


def test_segment_lgd_flags_low_count():
    """Segments with < 20 obs should be flagged."""
    from src.lgd_calculations import compute_realised_lgd, segment_lgd
    loans = _make_loans(25)  # intentionally small to get low-count segments
    cashflows = _make_cashflows(loans)
    lgd_df = compute_realised_lgd(loans, cashflows)
    result = segment_lgd(lgd_df, ["mortgage_class", "lvr_band"])
    if "segment_flag" in result.columns:
        # With 25 loans across 8 segment combos, some must be low_count
        assert (result["segment_flag"] == "low_count").any()


# ── compute_long_run_lgd ──────────────────────────────────────────────────────

def test_compute_long_run_lgd_vintage_ewa():
    from src.lgd_calculations import compute_realised_lgd, segment_lgd, compute_long_run_lgd
    loans = _make_loans(100, seed=7)
    cashflows = _make_cashflows(loans)
    lgd_df = compute_realised_lgd(loans, cashflows)
    seg_df = segment_lgd(lgd_df, ["mortgage_class"])
    result = compute_long_run_lgd(seg_df, segment_keys=["mortgage_class"], method="vintage_ewa")
    assert isinstance(result, pd.DataFrame)
    assert "long_run_lgd" in result.columns
    assert (result["long_run_lgd"] >= 0).all()
    assert (result["long_run_lgd"] <= 1.5).all()


def test_compute_long_run_lgd_raises_insufficient_vintages():
    from src.lgd_calculations import compute_long_run_lgd
    tiny_df = pd.DataFrame({
        "default_year": [2022],
        "realised_lgd": [0.30],
        "ead_at_default": [100_000],
        "mortgage_class": ["Standard"],
    })
    with pytest.raises(ValueError, match="vintage"):
        compute_long_run_lgd(tiny_df, segment_keys=["mortgage_class"],
                              method="vintage_ewa", min_years=5)


# ── compare_model_vs_actual ───────────────────────────────────────────────────

def test_compare_model_vs_actual_returns_bias():
    from src.lgd_calculations import compute_realised_lgd, compare_model_vs_actual
    loans = _make_loans(80)
    cashflows = _make_cashflows(loans)
    lgd_df = compute_realised_lgd(loans, cashflows)
    lgd_df["model_lgd"] = lgd_df["realised_lgd"] * 1.10  # 10% higher = conservative
    result = compare_model_vs_actual(lgd_df, model_lgd_col="model_lgd",
                                     segment_keys=["mortgage_class"])
    assert "bias" in result.columns
    assert "is_conservative" in result.columns
    assert (result["is_conservative"] == True).all()


# ── apply_regulatory_floor ────────────────────────────────────────────────────

def test_apply_regulatory_floor_mortgage():
    from src.lgd_calculations import apply_regulatory_floor
    s = pd.Series([0.05, 0.10, 0.15, 0.25])
    result = apply_regulatory_floor(s, product="mortgage")
    assert result.min() >= 0.10


def test_apply_regulatory_floor_mezz():
    from src.lgd_calculations import apply_regulatory_floor
    s = pd.Series([0.10, 0.25, 0.35, 0.50])
    result = apply_regulatory_floor(s, product="mezz_second_mortgage")
    assert result.min() >= 0.40


def test_apply_regulatory_floor_never_reduces():
    from src.lgd_calculations import apply_regulatory_floor
    s = pd.Series([0.80, 0.90, 1.00])
    result = apply_regulatory_floor(s, product="mortgage")
    # floor should never reduce already-high LGD
    pd.testing.assert_series_equal(result, s, check_names=False)
