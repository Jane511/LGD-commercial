from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.lgd_calculation import (  # noqa: E402
    CommercialLGDEngine,
    DevelopmentLGDEngine,
    MortgageLGDEngine,
    resolve_overlay_contract,
)
from src.overlay_parameters import OverlayParameterManager  # noqa: E402


def test_overlay_parameter_validation_fails_on_missing_columns(tmp_path: Path):
    bad = pd.DataFrame(
        {
            "parameter_group": ["overlay"],
            "product_scope": ["mortgage"],
            "value": [1.08],
        }
    )
    csv_path = tmp_path / "bad.csv"
    bad.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        OverlayParameterManager(csv_path=csv_path, manifest_path=tmp_path / "missing.json")


def test_overlay_parameter_validation_fails_on_duplicate_keys(tmp_path: Path):
    good = pd.read_csv(ROOT / "data" / "config" / "overlay_parameters.csv")
    dup = pd.concat([good, good.iloc[[0]]], ignore_index=True)
    csv_path = tmp_path / "dup.csv"
    dup.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="duplicate keys"):
        OverlayParameterManager(csv_path=csv_path, manifest_path=tmp_path / "missing.json")


def test_overlay_resolver_is_deterministic_and_precedence_ordered():
    manager = OverlayParameterManager()
    df = pd.DataFrame(
        {
            "security_type": ["Property", "PPSR - Mixed"],
            "industry_risk_score": [2.6, 4.2],
            "icr": [1.6, 1.1],
            "workout_months": [18, 28],
            "security_coverage_ratio": [1.1, 0.7],
        }
    )

    first = resolve_overlay_contract(df, "commercial", manager, scenario_id="baseline")
    second = resolve_overlay_contract(df, "commercial", manager, scenario_id="baseline")

    pd.testing.assert_frame_equal(first, second)
    assert (first["macro_downturn_scalar"] >= 1.0).all()
    assert (first["industry_downturn_adjustment"] > 0).all()
    assert (first["combined_downturn_scalar"] >= first["macro_downturn_scalar"]).all()


def test_standard_segments_present_for_core_engines():
    mortgage = pd.DataFrame(
        {
            "mortgage_class": ["Standard"],
            "ltv_at_default": [0.78],
            "credit_score": [700],
            "industry_risk_score": [2.8],
        }
    )
    commercial = pd.DataFrame(
        {
            "security_type": ["Property"],
            "security_coverage_ratio": [1.1],
            "annual_revenue": [5_000_000],
            "industry_risk_score": [3.1],
        }
    )
    development = pd.DataFrame(
        {
            "completion_stage": ["Mid-Construction"],
            "development_type": ["Residential Apartments"],
            "presale_coverage": [0.65],
            "lvr_as_if_complete": [0.72],
            "industry_risk_score": [3.4],
        }
    )

    m = MortgageLGDEngine.segment_loans(mortgage)
    c = CommercialLGDEngine.segment_loans(commercial)
    d = DevelopmentLGDEngine.segment_loans(development)

    for frame in [m, c, d]:
        assert {"std_module", "std_product_segment", "std_security_or_stage_band", "std_industry_risk_band"}.issubset(
            frame.columns
        )
