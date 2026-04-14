from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_generation import generate_all_datasets  # noqa: E402
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
    first = score_single_loan(row, product_type="commercial", seed=42)
    second = score_single_loan(row, product_type="commercial", seed=42)

    for col in ["lgd_base", "lgd_downturn", "lgd_final", "combined_downturn_scalar", "parameter_version"]:
        assert first[col] == second[col]


def test_formula_invariants_hold_for_all_products():
    datasets = generate_all_datasets()
    for product in ["mortgage", "commercial", "development", "cashflow_lending"]:
        loans = datasets[product]["loans"].head(25)
        out = score_batch_loans(loans, product_type=product)
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
    for product in ["mortgage", "commercial", "development", "cashflow_lending"]:
        out = score_batch_from_source(product_type=product, source_mode="generated")
        assert not out.empty
        assert set(NORMALIZED_OUTPUT_COLUMNS).issubset(set(out.columns))


def test_batch_from_source_controlled_all_products(tmp_path: Path):
    controlled_root = tmp_path / "controlled_all"
    controlled_root.mkdir(parents=True, exist_ok=True)
    datasets = generate_all_datasets()
    for product in ["mortgage", "commercial", "development", "cashflow_lending"]:
        datasets[product]["loans"].head(5).to_csv(controlled_root / f"{product}_loans.csv", index=False)
        datasets[product]["cashflows"].head(10).to_csv(controlled_root / f"{product}_cashflows.csv", index=False)

    for product in ["mortgage", "commercial", "development", "cashflow_lending"]:
        out = score_batch_from_source(
            product_type=product,
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
        str(ROOT / "scripts" / "score_new_loan.py"),
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
