"""
Tests for src/rba_rates_loader.py — RBA B6 indicator lending rates.

Covers:
- load_rba_lending_rates: returns DataFrame, required columns
- get_discount_rate_for_loan: returns (rate, source) tuple, fallback logic
- build_discount_rate_register: one row per product/year combination
- source label is 'rba_b6' when real data available, 'rba_cash_plus_300bps' otherwise
"""
from __future__ import annotations

import pandas as pd
import pytest


def test_load_rba_lending_rates_returns_dataframe():
    from src.rba_rates_loader import load_rba_lending_rates
    result = load_rba_lending_rates()
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_load_rba_rates_required_columns():
    from src.rba_rates_loader import load_rba_lending_rates
    result = load_rba_lending_rates()
    for col in ("year", "product_type", "rate_pct", "rate_source"):
        assert col in result.columns, f"Missing column: {col}"


def test_load_rba_rates_covers_2014_2024():
    from src.rba_rates_loader import load_rba_lending_rates
    result = load_rba_lending_rates()
    assert result["year"].min() <= 2014
    assert result["year"].max() >= 2024


def test_load_rba_rates_positive():
    from src.rba_rates_loader import load_rba_lending_rates
    result = load_rba_lending_rates()
    assert (result["rate_pct"] > 0).all(), "All rates should be positive"
    assert (result["rate_pct"] < 0.25).all(), "Rates > 25% are implausible"


def test_get_discount_rate_returns_tuple():
    from src.rba_rates_loader import load_rba_lending_rates, get_discount_rate_for_loan
    rates_df = load_rba_lending_rates()
    rate, source = get_discount_rate_for_loan("mortgage_owner_occupier", 2020, rates_df)
    assert isinstance(rate, float)
    assert isinstance(source, str)
    assert rate > 0


def test_get_discount_rate_source_label():
    from src.rba_rates_loader import load_rba_lending_rates, get_discount_rate_for_loan
    rates_df = load_rba_lending_rates()
    _, source = get_discount_rate_for_loan("mortgage_owner_occupier", 2020, rates_df)
    assert source in ("rba_b6", "rba_cash_plus_300bps", "hardcoded_fallback")


def test_get_discount_rate_fallback_for_unknown_product():
    """Unknown product type should fall back gracefully."""
    from src.rba_rates_loader import load_rba_lending_rates, get_discount_rate_for_loan
    rates_df = load_rba_lending_rates()
    rate, source = get_discount_rate_for_loan("unknown_product_xyz", 2020, rates_df)
    assert rate > 0
    assert source != "rba_b6"  # must use fallback


def test_build_discount_rate_register_shape():
    from src.rba_rates_loader import load_rba_lending_rates, build_discount_rate_register
    rates_df = load_rba_lending_rates()
    products = ["mortgage_owner_occupier", "small_business_variable"]
    years = list(range(2018, 2025))
    register = build_discount_rate_register(products, years, rates_df)
    assert isinstance(register, pd.DataFrame)
    # Should have one row per product/year combination
    assert len(register) == len(products) * len(years)


def test_build_discount_rate_register_columns():
    from src.rba_rates_loader import load_rba_lending_rates, build_discount_rate_register
    rates_df = load_rba_lending_rates()
    register = build_discount_rate_register(
        ["mortgage_owner_occupier"], [2020, 2021, 2022], rates_df
    )
    for col in ("product_type", "year", "discount_rate", "rate_source"):
        assert col in register.columns, f"Missing column: {col}"
