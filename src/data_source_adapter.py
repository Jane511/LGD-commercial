"""
Canonical data-source adapter for LGD pipeline inputs.

Purpose
-------
Keep one stable in-memory contract for model execution while allowing
source swapping between:
1. Generated demo datasets
2. Controlled-system extracts (CSV/Parquet files)

Canonical contract
------------------
{
  "mortgage": {"loans": DataFrame, "cashflows": DataFrame},
  "commercial": {"loans": DataFrame, "cashflows": DataFrame},
  "development": {"loans": DataFrame, "cashflows": DataFrame},
  "cashflow_lending": {"loans": DataFrame, "cashflows": DataFrame},
}
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .data_generation import generate_all_datasets


PRODUCTS = ("mortgage", "commercial", "development", "cashflow_lending")
SUBSETS = ("loans", "cashflows")


@dataclass(frozen=True)
class DataSourceConfig:
    source: str = "generated"
    controlled_root: str | Path = "data/controlled"
    require_all_products: bool = True


def _normalise_source(source: str) -> str:
    value = str(source).strip().lower()
    if value not in {"generated", "controlled"}:
        raise ValueError("source must be one of: generated, controlled")
    return value


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type for controlled input: {path.name}")


def _find_table_path(root: Path, product: str, subset: str) -> Path | None:
    candidates = [
        root / f"{product}_{subset}.parquet",
        root / f"{product}_{subset}.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def validate_dataset_contract(
    datasets: dict[str, dict[str, pd.DataFrame]],
    require_all_products: bool = True,
) -> None:
    if not isinstance(datasets, dict):
        raise ValueError("datasets must be a dict keyed by product")

    missing_products = [p for p in PRODUCTS if p not in datasets]
    if require_all_products and missing_products:
        raise ValueError(f"Missing products in datasets contract: {missing_products}")

    products_to_check = PRODUCTS if require_all_products else [p for p in PRODUCTS if p in datasets]
    for product in products_to_check:
        block = datasets.get(product)
        if not isinstance(block, dict):
            raise ValueError(f"datasets['{product}'] must be a dict with loans/cashflows")

        for subset in SUBSETS:
            if subset not in block:
                raise ValueError(f"datasets['{product}'] missing subset '{subset}'")
            if not isinstance(block[subset], pd.DataFrame):
                raise ValueError(f"datasets['{product}']['{subset}'] must be a pandas DataFrame")

        loans = block["loans"]
        if "loan_id" not in loans.columns:
            raise ValueError(f"datasets['{product}']['loans'] missing required column: loan_id")


def load_controlled_datasets(
    controlled_root: str | Path,
    require_all_products: bool = True,
) -> dict[str, dict[str, pd.DataFrame]]:
    root = Path(controlled_root)
    if not root.exists():
        raise ValueError(f"Controlled data root does not exist: {root}")

    datasets: dict[str, dict[str, pd.DataFrame]] = {}
    missing_files: list[str] = []

    for product in PRODUCTS:
        product_block: dict[str, pd.DataFrame] = {}
        for subset in SUBSETS:
            path = _find_table_path(root, product, subset)
            if path is None:
                if require_all_products:
                    missing_files.append(f"{product}_{subset}.csv/.parquet")
                continue
            product_block[subset] = _read_table(path)

        if product_block:
            if "loans" not in product_block:
                raise ValueError(f"Controlled dataset for '{product}' must include a loans table")
            product_block.setdefault("cashflows", pd.DataFrame())
            datasets[product] = product_block

    if missing_files:
        raise ValueError(
            "Controlled datasets missing required files: " + ", ".join(sorted(missing_files))
        )

    validate_dataset_contract(datasets, require_all_products=require_all_products)
    return datasets


def load_datasets(
    source: str = "generated",
    controlled_root: str | Path = "data/controlled",
    require_all_products: bool = True,
) -> dict[str, dict[str, pd.DataFrame]]:
    source_norm = _normalise_source(source)

    if source_norm == "generated":
        datasets = generate_all_datasets()
        validate_dataset_contract(datasets, require_all_products=require_all_products)
        return datasets

    return load_controlled_datasets(
        controlled_root=controlled_root,
        require_all_products=require_all_products,
    )


def export_controlled_input_templates(
    output_root: str | Path = "data/controlled/templates",
    sample_datasets: dict[str, dict[str, pd.DataFrame]] | None = None,
) -> dict[str, Any]:
    """
    Export empty CSV templates for controlled-system loaders using canonical
    generated dataset column layouts.
    """
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)

    datasets = sample_datasets or generate_all_datasets()
    validate_dataset_contract(datasets, require_all_products=True)

    written: list[str] = []
    for product in PRODUCTS:
        for subset in SUBSETS:
            cols = list(datasets[product][subset].columns)
            template = pd.DataFrame(columns=cols)
            path = output / f"{product}_{subset}.csv"
            template.to_csv(path, index=False)
            written.append(str(path))

    schema_rows: list[dict[str, Any]] = []
    for product in PRODUCTS:
        for subset in SUBSETS:
            for col in datasets[product][subset].columns:
                dtype = str(datasets[product][subset][col].dtype)
                schema_rows.append(
                    {"product": product, "subset": subset, "column": col, "dtype_hint": dtype}
                )
    schema_path = output / "controlled_input_schema.csv"
    pd.DataFrame(schema_rows).to_csv(schema_path, index=False)
    written.append(str(schema_path))

    return {
        "output_root": str(output.resolve()),
        "files_written": written,
        "products": list(PRODUCTS),
    }
