from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_source_adapter import (
    PRODUCTS,
    SUBSETS,
    load_controlled_datasets,
    load_datasets,
    validate_dataset_contract,
)


def _write_minimal_controlled_files(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for product in PRODUCTS:
        pd.DataFrame({"loan_id": [f"{product}_1"], "ead_at_default": [100.0]}).to_csv(
            root / f"{product}_loans.csv", index=False
        )
        pd.DataFrame({"loan_id": [f"{product}_1"], "cashflow_amount": [10.0]}).to_csv(
            root / f"{product}_cashflows.csv", index=False
        )


def test_validate_dataset_contract_requires_loan_id():
    bad = {
        "mortgage": {"loans": pd.DataFrame({"x": [1]}), "cashflows": pd.DataFrame()},
        "commercial": {"loans": pd.DataFrame({"loan_id": [1]}), "cashflows": pd.DataFrame()},
        "development": {"loans": pd.DataFrame({"loan_id": [1]}), "cashflows": pd.DataFrame()},
        "cashflow_lending": {"loans": pd.DataFrame({"loan_id": [1]}), "cashflows": pd.DataFrame()},
    }
    with pytest.raises(ValueError, match="missing required column: loan_id"):
        validate_dataset_contract(bad, require_all_products=True)


def test_load_controlled_datasets_from_csv(tmp_path: Path):
    _write_minimal_controlled_files(tmp_path)
    datasets = load_controlled_datasets(tmp_path, require_all_products=True)

    for product in PRODUCTS:
        assert product in datasets
        for subset in SUBSETS:
            assert subset in datasets[product]
            assert not datasets[product][subset].empty


def test_load_datasets_controlled_fails_when_files_missing(tmp_path: Path):
    # Only one file is written; contract requires all products/subsets.
    pd.DataFrame({"loan_id": ["m1"], "ead_at_default": [100]}).to_csv(
        tmp_path / "mortgage_loans.csv", index=False
    )
    with pytest.raises(ValueError, match="missing required files"):
        load_datasets(source="controlled", controlled_root=tmp_path, require_all_products=True)
