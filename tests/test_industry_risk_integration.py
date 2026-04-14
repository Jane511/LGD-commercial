from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.industry_risk_integration import (  # noqa: E402
    IndustryRiskLoader,
    UNMAPPED_DEFAULTS,
    enrich_loans_with_industry_risk,
)


def _write_compact_upstream(base: Path) -> Path:
    exports = base / "data" / "exports"
    exports.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "industry": ["Construction", "Manufacturing"],
            "industry_risk_score": [3.3, 3.9],
            "working_capital_lgd_overlay_score": [3.4, 4.1],
            "debt_to_ebitda_benchmark": [3.1, 3.8],
            "icr_benchmark": [2.9, 2.4],
            "esg_sensitive_sector": [False, True],
        }
    ).to_parquet(exports / "industry_risk_scores.parquet", index=False)

    pd.DataFrame(
        {
            "macro_regime": ["Base", "Downturn"],
            "is_downturn": [False, True],
        }
    ).to_parquet(exports / "macro_regime_flags.parquet", index=False)

    pd.DataFrame(
        {
            "industry": ["Construction", "Manufacturing"],
            "lgd_overlay_addon": [0.08, 0.17],
        }
    ).to_parquet(exports / "downturn_overlay_table.parquet", index=False)

    return exports


def test_compact_upstream_contract_validation_and_loaders(tmp_path: Path):
    exports = _write_compact_upstream(tmp_path)
    loader = IndustryRiskLoader(exports_path=exports, validate_contract=True)

    scorecard = loader.load_base_risk_scorecard()
    assert {"industry", "industry_base_risk_score", "industry_base_risk_level"}.issubset(
        set(scorecard.columns)
    )

    stress = loader.load_stress_matrix()
    assert {"industry", "scenario_name", "stress_delta"}.issubset(set(stress.columns))
    assert (stress["stress_delta"] > 0).all()


def test_contract_validation_fails_when_required_columns_missing(tmp_path: Path):
    exports = _write_compact_upstream(tmp_path)
    bad = pd.read_parquet(exports / "downturn_overlay_table.parquet")
    bad = bad.drop(columns=["industry"])
    bad.to_parquet(exports / "downturn_overlay_table.parquet", index=False)

    with pytest.raises(ValueError, match="downturn_overlay_table.parquet missing required column: industry"):
        IndustryRiskLoader(exports_path=exports, validate_contract=True)


def test_enrich_loans_with_compact_contract_and_unmapped_fallback(tmp_path: Path):
    exports = _write_compact_upstream(tmp_path)
    loader = IndustryRiskLoader(exports_path=exports, validate_contract=True)

    loans = pd.DataFrame(
        {
            "loan_id": ["L1", "L2"],
            "industry": ["Construction", "Mining"],
            "ead": [1_000_000.0, 750_000.0],
        }
    )

    out = enrich_loans_with_industry_risk(loans, loader, product_type="commercial")

    required_cols = {
        "industry_risk_score",
        "industry_risk_level",
        "wc_lgd_overlay_score",
        "industry_debt_to_ebitda_benchmark",
        "industry_icr_benchmark",
        "industry_esg_sensitive",
        "industry_recovery_haircut",
        "wc_lgd_adjustment",
    }
    assert required_cols.issubset(set(out.columns))

    mining_row = out.loc[out["industry"] == "Mining"].iloc[0]
    assert float(mining_row["industry_risk_score"]) == UNMAPPED_DEFAULTS["industry_base_risk_score"]
    assert float(mining_row["wc_lgd_overlay_score"]) == UNMAPPED_DEFAULTS[
        "working_capital_lgd_overlay_score"
    ]
