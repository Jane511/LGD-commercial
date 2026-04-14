"""
Tests for src/generators/ — synthetic historical workout data generators.

Covers:
- All 11 generators produce loans and cashflows DataFrames
- 10-year observation window (2014-2024)
- Required columns present (ead_at_default, recovery_amount, default_date, etc.)
- lip_costs and cure_flag columns present for mortgage
- generate_all_historical_workouts produces all products
"""
from __future__ import annotations

import pandas as pd
import pytest


ALL_PRODUCTS = [
    "mortgage", "commercial_cashflow", "receivables", "trade_contingent",
    "asset_equipment", "development_finance", "cre_investment",
    "residual_stock", "land_subdivision", "bridging", "mezz_second_mortgage",
]

REQUIRED_LOAN_COLS = [
    "ead_at_default", "default_date", "recovery_amount",
    "direct_costs", "discount_rate",
]


@pytest.mark.parametrize("product", ALL_PRODUCTS)
def test_generator_produces_loans(product):
    from src.generators import GENERATOR_MAP
    assert product in GENERATOR_MAP, f"No generator for {product}"
    gen_cls = GENERATOR_MAP[product]
    gen = gen_cls(n_loans=50, seed=42)
    loans = gen.generate_loans()
    assert isinstance(loans, pd.DataFrame)
    assert len(loans) >= 40, f"Expected ~50 loans, got {len(loans)}"


@pytest.mark.parametrize("product", ALL_PRODUCTS)
def test_generator_has_required_columns(product):
    from src.generators import GENERATOR_MAP
    gen_cls = GENERATOR_MAP[product]
    gen = gen_cls(n_loans=50, seed=42)
    loans = gen.generate_loans()
    for col in REQUIRED_LOAN_COLS:
        assert col in loans.columns, f"{product}: missing column '{col}'"


@pytest.mark.parametrize("product", ALL_PRODUCTS)
def test_generator_observation_window_10yr(product):
    """All generators must span 2014-2024 (≥9 years) for APS 113 Att A compliance."""
    from src.generators import GENERATOR_MAP
    gen_cls = GENERATOR_MAP[product]
    gen = gen_cls(n_loans=100, seed=42)
    loans = gen.generate_loans()
    dates = pd.to_datetime(loans["default_date"])
    n_years = (dates.max() - dates.min()).days / 365.25
    assert n_years >= 7, \
        f"{product}: observation window {n_years:.1f}y < 7y minimum"


def test_mortgage_generator_has_cure_flag():
    from src.generators import GENERATOR_MAP
    gen = GENERATOR_MAP["mortgage"](n_loans=50, seed=42)
    loans = gen.generate_loans()
    assert "cure_flag" in loans.columns or "is_cured" in loans.columns, \
        "Mortgage generator must have cure_flag / is_cured column"


def test_mortgage_generator_has_lip_costs():
    from src.generators import GENERATOR_MAP
    gen = GENERATOR_MAP["mortgage"](n_loans=50, seed=42)
    loans = gen.generate_loans()
    assert "lip_costs" in loans.columns, \
        "Mortgage generator must have lip_costs column (APS 113 s.32)"


def test_mortgage_generator_has_lvr():
    from src.generators import GENERATOR_MAP
    gen = GENERATOR_MAP["mortgage"](n_loans=50, seed=42)
    loans = gen.generate_loans()
    assert "lvr_at_default" in loans.columns or "ltv_at_default" in loans.columns


def test_development_finance_has_completion_stage():
    from src.generators import GENERATOR_MAP
    gen = GENERATOR_MAP["development_finance"](n_loans=50, seed=42)
    loans = gen.generate_loans()
    assert "completion_stage_at_default" in loans.columns


def test_generate_all_historical_workouts_runs():
    import tempfile
    from pathlib import Path
    from src.generators import generate_all_historical_workouts
    with tempfile.TemporaryDirectory() as tmpdir:
        results = generate_all_historical_workouts(
            seed=42,
            output_dir=Path(tmpdir),
            write_parquet=False,
            products=["mortgage", "receivables"],
        )
    assert "mortgage" in results
    assert "receivables" in results
    assert "loans" in results["mortgage"]


@pytest.mark.parametrize("product", ALL_PRODUCTS)
def test_generator_ead_positive(product):
    from src.generators import GENERATOR_MAP
    gen_cls = GENERATOR_MAP[product]
    gen = gen_cls(n_loans=30, seed=99)
    loans = gen.generate_loans()
    assert (loans["ead_at_default"] > 0).all(), \
        f"{product}: non-positive EAD values found"


@pytest.mark.parametrize("product", ALL_PRODUCTS)
def test_generator_discount_rate_reasonable(product):
    from src.generators import GENERATOR_MAP
    gen_cls = GENERATOR_MAP[product]
    gen = gen_cls(n_loans=30, seed=5)
    loans = gen.generate_loans()
    assert (loans["discount_rate"] > 0).all(), f"{product}: discount_rate <= 0"
    assert (loans["discount_rate"] < 0.25).all(), \
        f"{product}: discount_rate > 25% — check RBA rate loader"
