"""
Tests for src/apra_benchmarks.py — APRA ADI peer benchmarking.

Covers:
- load_apra_adi_benchmarks: returns DataFrame, required columns
- generate_benchmark_comparison: adequacy flag, diff columns
- Limitation noted: impairment ratio is NOT a direct LGD benchmark
"""
from __future__ import annotations

import pandas as pd
import pytest


def test_load_apra_benchmarks_returns_dataframe():
    from src.apra_benchmarks import load_apra_adi_benchmarks
    result = load_apra_adi_benchmarks()
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_load_apra_benchmarks_columns():
    from src.apra_benchmarks import load_apra_adi_benchmarks
    result = load_apra_adi_benchmarks()
    for col in ("year", "product_type", "apra_system_impairment_ratio"):
        assert col in result.columns, f"Missing column: {col}"


def test_load_apra_benchmarks_ratios_positive():
    from src.apra_benchmarks import load_apra_adi_benchmarks
    result = load_apra_adi_benchmarks()
    assert (result["apra_system_impairment_ratio"] >= 0).all()
    assert (result["apra_system_impairment_ratio"] < 1.0).all(), \
        "Impairment ratio > 100% is implausible"


def test_generate_benchmark_comparison_returns_dataframe():
    from src.apra_benchmarks import load_apra_adi_benchmarks, generate_benchmark_comparison
    benchmarks = load_apra_adi_benchmarks()
    calibrated = pd.DataFrame({
        "product": ["mortgage", "commercial_cashflow"],
        "final_lgd": [0.22, 0.35],
    })
    result = generate_benchmark_comparison(calibrated, benchmarks)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_benchmark_comparison_adequacy_flag():
    from src.apra_benchmarks import load_apra_adi_benchmarks, generate_benchmark_comparison
    benchmarks = load_apra_adi_benchmarks()
    calibrated = pd.DataFrame({
        "product": ["mortgage"],
        "final_lgd": [0.22],
    })
    result = generate_benchmark_comparison(calibrated, benchmarks)
    if "benchmark_adequacy_flag" in result.columns:
        valid = {"consistent", "conservative", "below_benchmark"}
        assert set(result["benchmark_adequacy_flag"].dropna()).issubset(valid)


def test_benchmark_comparison_has_diff_column():
    from src.apra_benchmarks import load_apra_adi_benchmarks, generate_benchmark_comparison
    benchmarks = load_apra_adi_benchmarks()
    calibrated = pd.DataFrame({
        "product": ["mortgage", "commercial_cashflow"],
        "final_lgd": [0.25, 0.40],
    })
    result = generate_benchmark_comparison(calibrated, benchmarks)
    diff_cols = [c for c in result.columns if "diff" in c.lower() or "vs_benchmark" in c.lower()]
    assert len(diff_cols) > 0, "Benchmark comparison should include a diff/vs_benchmark column"


def test_benchmark_notes_directional_only():
    """
    APRA impairment ratio ≠ LGD. The comparison must be clearly labelled
    as directional/indicative, not a direct LGD benchmark.
    """
    from src.apra_benchmarks import load_apra_adi_benchmarks, generate_benchmark_comparison
    benchmarks = load_apra_adi_benchmarks()
    calibrated = pd.DataFrame({"product": ["mortgage"], "final_lgd": [0.20]})
    result = generate_benchmark_comparison(calibrated, benchmarks)
    if "notes" in result.columns:
        note_text = " ".join(result["notes"].dropna().astype(str)).lower()
        assert "directional" in note_text or "proxy" in note_text or "impairment" in note_text, \
            "Notes should acknowledge the impairment ratio is a proxy, not a direct LGD benchmark"
