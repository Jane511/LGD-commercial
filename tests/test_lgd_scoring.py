from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.data_generation import generate_all_datasets  # noqa: E402
from src.lgd_scoring import (  # noqa: E402
    NORMALIZED_OUTPUT_COLUMNS,
    score_batch_from_source,
    score_batch_loans,
    score_single_loan,
    score_single_loan_from_source_template,
)


def _sample_row(product: str) -> dict:
    datasets = generate_all_datasets()
    return datasets[product]["loans"].head(1).iloc[0].to_dict()


def test_schema_validation_fails_on_missing_required_column():
    row = _sample_row("mortgage")
    row.pop("lmi_eligible", None)
    with pytest.raises(ValueError, match="missing required columns"):
        score_single_loan(row, product_type="mortgage")


def test_determinism_same_input_same_seed():
    row = _sample_row("commercial")
    first = score_single_loan(row, product_type="commercial_cashflow", seed=42)
    second = score_single_loan(row, product_type="commercial_cashflow", seed=42)

    for col in ["lgd_base", "lgd_downturn", "lgd_final", "combined_downturn_scalar", "parameter_version"]:
        assert first[col] == second[col]


def test_formula_invariants_hold_for_all_products():
    datasets = generate_all_datasets()
    # (dataset_key, product_type) — dataset keys match generate_all_datasets() keys;
    # product_type uses canonical sub-types (legacy "commercial" / "development" raise ValueError)
    product_pairs = [
        ("mortgage", "mortgage"),
        ("commercial", "commercial_cashflow"),
        ("development", "development_finance"),
        ("cashflow_lending", "cashflow_lending"),
    ]
    for dataset_key, product_type in product_pairs:
        loans = datasets[dataset_key]["loans"].head(25)
        out = score_batch_loans(loans, product_type=product_type)
        assert set(NORMALIZED_OUTPUT_COLUMNS).issubset(set(out.columns))
        assert (out["lgd_base"].between(0, 1)).all()
        assert (out["lgd_downturn"].between(0, 1)).all()
        assert (out["lgd_final"].between(0, 1)).all()
        assert (out["lgd_downturn"] >= out["lgd_base"]).all()


def test_adapter_integration_generated_and_controlled(tmp_path: Path):
    generated = score_single_loan_from_source_template(
        {"loan_id": "g_1", "ead": 120_000.0, "realised_lgd": 0.25, "lmi_eligible": 1, "mortgage_class": "Standard"},
        product_type="mortgage",
        source_mode="generated",
    )
    assert 0 <= float(generated["lgd_final"]) <= 1

    controlled_root = tmp_path / "controlled"
    controlled_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([_sample_row("mortgage")]).to_csv(controlled_root / "mortgage_loans.csv", index=False)
    pd.DataFrame([{"loan_id": _sample_row("mortgage")["loan_id"], "cashflow_amount": 1000.0}]).to_csv(
        controlled_root / "mortgage_cashflows.csv", index=False
    )
    controlled = score_single_loan_from_source_template(
        {"loan_id": "c_1", "ead": 110_000.0, "realised_lgd": 0.30, "lmi_eligible": 0, "mortgage_class": "Non-Standard"},
        product_type="mortgage",
        source_mode="controlled",
        controlled_root=controlled_root,
    )
    assert 0 <= float(controlled["lgd_final"]) <= 1


def test_batch_from_source_generated_all_products():
    for product in ["mortgage", "commercial_cashflow", "development_finance", "cashflow_lending"]:
        out = score_batch_from_source(product_type=product, source_mode="generated")
        assert not out.empty
        assert set(NORMALIZED_OUTPUT_COLUMNS).issubset(set(out.columns))


def test_batch_from_source_controlled_all_products(tmp_path: Path):
    controlled_root = tmp_path / "controlled_all"
    controlled_root.mkdir(parents=True, exist_ok=True)
    datasets = generate_all_datasets()
    # Write files using generate_all_datasets() keys (legacy dataset names)
    for dataset_key in ["mortgage", "commercial", "development", "cashflow_lending"]:
        datasets[dataset_key]["loans"].head(5).to_csv(
            controlled_root / f"{dataset_key}_loans.csv", index=False
        )
        datasets[dataset_key]["cashflows"].head(10).to_csv(
            controlled_root / f"{dataset_key}_cashflows.csv", index=False
        )

    # Score using canonical sub-type product_type values; _dataset_key() bridges
    # back to legacy file names (commercial_cashflow → commercial, etc.)
    product_types = ["mortgage", "commercial_cashflow", "development_finance", "cashflow_lending"]
    for product_type in product_types:
        out = score_batch_from_source(
            product_type=product_type,
            source_mode="controlled",
            controlled_root=controlled_root,
        )
        assert not out.empty
        assert (out["lgd_final"].between(0, 1)).all()


def test_cli_and_api_parity_single_payload(tmp_path: Path):
    payload = {
        "loan_id": "api_cli_1",
        "ead": 150000.0,
        "realised_lgd": 0.28,
        "lmi_eligible": 1,
        "mortgage_class": "Standard",
        "ltv_at_default": 0.82,
        "dti": 0.36,
    }
    api = score_single_loan(payload, product_type="mortgage", scenario_id="baseline", seed=42)

    in_json = tmp_path / "loan.json"
    out_json = tmp_path / "scored.json"
    in_json.write_text(json.dumps(payload), encoding="utf-8")

    cmd = [
        sys.executable,
        "-m", "src.scoring.scoring",
        "--product-type",
        "mortgage",
        "--single-json",
        str(in_json),
        "--output",
        str(out_json),
        "--scenario-id",
        "baseline",
        "--seed",
        "42",
    ]
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    cli = json.loads(out_json.read_text(encoding="utf-8"))

    for col in ["loan_id", "product_type", "lgd_base", "lgd_downturn", "lgd_final", "parameter_version", "scenario_id"]:
        assert api[col] == cli[col]


# ── Integration seam: proxy engine ↔ calibration layer ───────────────────────

def test_calibration_imports_do_not_conflict_with_scoring():
    """
    APS 113 integration seam test.

    The new calibration modules (src.lgd_calculations, src.moc_framework, etc.)
    must coexist with the existing proxy scoring engine (src.lgd_scoring,
    src.lgd_calculation) in the same Python process without import conflicts.

    Specifically: src.lgd_calculations (with 's') must not shadow or overwrite
    anything from src.lgd_calculation (without 's').
    """
    # Import both modules in the same process
    import src.lgd_calculation as proxy_engine
    import src.lgd_calculations as calib_engine  # noqa: F401 — import test

    # Proxy engine functions must still be accessible
    assert hasattr(proxy_engine, "apply_downturn_overlay"), \
        "apply_downturn_overlay missing from proxy engine after calibration import"
    assert hasattr(proxy_engine, "exposure_weighted_average"), \
        "exposure_weighted_average missing from proxy engine after calibration import"
    assert hasattr(proxy_engine, "build_weighted_lgd_output"), \
        "build_weighted_lgd_output missing from proxy engine after calibration import"

    # Calibration engine functions must be present
    assert hasattr(calib_engine, "compute_realised_lgd"), \
        "compute_realised_lgd missing from calibration engine"
    assert hasattr(calib_engine, "compute_long_run_lgd"), \
        "compute_long_run_lgd missing from calibration engine"


def test_scoring_output_stable_after_calibration_import():
    """
    Proxy scoring results must be unchanged after calibration modules are imported.
    This guards against any accidental monkey-patching or import side effects.
    """
    import src.lgd_calculations  # noqa: F401 — trigger import side effects (if any)
    import src.moc_framework  # noqa: F401

    row = _sample_row("mortgage")
    result = score_single_loan(row, product_type="mortgage", seed=42)
    # Proxy engine should still produce non-zero outputs
    assert result["lgd_base"] > 0
    assert result["lgd_final"] > 0
    assert result["lgd_downturn"] >= result["lgd_base"]


def test_ambiguous_product_types_raise_value_error():
    """Legacy ambiguous labels must raise ValueError with a helpful message."""
    from src.product_routing import LEGACY_AMBIGUOUS
    row = _sample_row("mortgage")
    for ambiguous_key in LEGACY_AMBIGUOUS:
        with pytest.raises(ValueError):
            score_single_loan(row, product_type=ambiguous_key)
