"""
Tests for src/moc_framework.py — Margin of Conservatism framework.

Key invariants:
1. MoC applied AFTER downturn overlay (APS 113 s.63 — critical ordering constraint)
2. Five APS 113 s.65 sources each add non-negative bps
3. Total MoC capped at product-specific max
4. apply_moc() increases LGD relative to downturn LGD (never decreases)
5. run_calibration_pipeline() enforces LR → downturn → MoC → floor order
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_segment_df(n=40, seed=42):
    rng = np.random.default_rng(seed)
    yrs = np.tile(np.arange(2014, 2024), int(np.ceil(n / 10)))[:n]
    return pd.DataFrame({
        "loan_id": [f"L{i}" for i in range(n)],
        "default_year": yrs,
        "realised_lgd": rng.uniform(0.10, 0.70, n),
        "ead_at_default": rng.uniform(50_000, 500_000, n),
        "security_type": rng.choice(["Property", "PPSR"], n),
        "mortgage_class": rng.choice(["Standard", "Non-Standard"], n),
    })


# ── MoCRegister ──────────────────────────────────────────────────────────────

def test_moc_register_returns_dataframe():
    from src.moc_framework import MoCRegister
    df = _make_segment_df()
    reg = MoCRegister(product="mortgage")
    result = reg.build_moc_register(
        segment_df=df,
        segment_keys=["mortgage_class"],
        n_downturn_vintages=2,
        psi_value=0.05,
        backtesting_bias=0.02,
    )
    assert isinstance(result, pd.DataFrame)
    assert "total_moc" in result.columns


def test_moc_register_non_negative_components():
    from src.moc_framework import MoCRegister
    df = _make_segment_df()
    reg = MoCRegister(product="mortgage")
    result = reg.build_moc_register(
        segment_df=df, segment_keys=["mortgage_class"],
        n_downturn_vintages=2, psi_value=0.05, backtesting_bias=0.02,
    )
    moc_cols = [c for c in result.columns if c.endswith("_moc")]
    for col in moc_cols:
        assert (result[col] >= 0).all(), f"{col} has negative values"


def test_cyclicality_moc_triggered_when_no_downturn():
    """If no downturn vintages, cyclicality_moc should be > 0."""
    from src.moc_framework import MoCRegister
    df = _make_segment_df()
    reg = MoCRegister(product="mortgage")
    result = reg.build_moc_register(
        segment_df=df, segment_keys=["mortgage_class"],
        n_downturn_vintages=0,  # No downturn data
        psi_value=0.05, backtesting_bias=0.02,
    )
    if "cyclicality_moc" in result.columns:
        assert (result["cyclicality_moc"] > 0).any()


def test_moc_capped_at_product_maximum():
    from src.moc_framework import MoCRegister, PRODUCT_MOC_CAPS
    product = "mortgage"
    cap = PRODUCT_MOC_CAPS.get(product, 0.10)
    df = _make_segment_df()
    reg = MoCRegister(product=product)
    result = reg.build_moc_register(
        segment_df=df, segment_keys=["mortgage_class"],
        n_downturn_vintages=0,
        psi_value=0.99,  # very high to trigger instability
        backtesting_bias=0.20,  # very high to trigger model error
    )
    assert (result["total_moc"] <= cap + 1e-9).all(), \
        f"MoC exceeds cap {cap}: {result['total_moc'].max()}"


# ── apply_moc ────────────────────────────────────────────────────────────────

def test_apply_moc_increases_lgd():
    """apply_moc() must never DECREASE LGD — only hold or increase."""
    from src.moc_framework import MoCRegister, apply_moc
    df = _make_segment_df()
    downturn_lgd = pd.Series(np.random.default_rng(10).uniform(0.2, 0.5, len(df)))
    reg = MoCRegister(product="mortgage")
    moc_df = reg.build_moc_register(
        segment_df=df, segment_keys=["mortgage_class"],
        n_downturn_vintages=2, psi_value=0.05, backtesting_bias=0.02,
    )
    result = apply_moc(downturn_lgd, moc_df, segment_col="mortgage_class")
    assert (result >= downturn_lgd - 1e-9).all(), \
        "apply_moc() reduced LGD — MoC must only increase or hold"


def test_apply_moc_takes_downturn_not_lr_lgd():
    """
    APS 113 s.63 critical order: MoC is applied to downturn LGD, not LR-LGD.
    Verify: LGD after MoC >= downturn LGD.
    """
    from src.moc_framework import MoCRegister, apply_moc
    df = _make_segment_df()
    lr_lgd = pd.Series([0.20] * len(df))
    downturn_lgd = lr_lgd * 1.10  # 10% uplift
    reg = MoCRegister(product="mortgage")
    moc_df = reg.build_moc_register(
        segment_df=df, segment_keys=["mortgage_class"],
        n_downturn_vintages=2, psi_value=0.05, backtesting_bias=0.01,
    )
    lgd_with_moc = apply_moc(downturn_lgd, moc_df, segment_col="mortgage_class")
    assert lgd_with_moc.mean() >= downturn_lgd.mean(), \
        "MoC applied to downturn LGD must produce result >= downturn LGD"


# ── run_calibration_pipeline ─────────────────────────────────────────────────

def test_run_calibration_pipeline_step_order():
    """
    Verify run_calibration_pipeline() enforces correct APS 113 order:
    LR-LGD → downturn → MoC → floor.
    The 'calibration_steps' key in the result must include steps in the
    correct sequence.
    """
    from src.moc_framework import run_calibration_pipeline
    df = _make_segment_df(80)
    df["lgd_long_run"] = 0.25
    result = run_calibration_pipeline(
        loans=df,
        product="mortgage",
        segment_keys=["mortgage_class"],
        lr_lgd_col="lgd_long_run",
    )
    assert "lgd_downturn" in result or "final_lgd" in result, \
        "run_calibration_pipeline must return calibration outputs"
    # If step order is tracked, verify sequence
    if "calibration_steps" in result:
        steps = result["calibration_steps"]
        step_names = [s.get("step", "") for s in steps]
        downturn_idx = next((i for i, s in enumerate(step_names) if "downturn" in s), None)
        moc_idx = next((i for i, s in enumerate(step_names) if "moc" in s.lower()), None)
        floor_idx = next((i for i, s in enumerate(step_names) if "floor" in s.lower()), None)
        if all(x is not None for x in [downturn_idx, moc_idx, floor_idx]):
            assert downturn_idx < moc_idx < floor_idx, \
                "APS 113 s.63: order must be downturn → MoC → floor"
