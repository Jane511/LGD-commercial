from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

from src.data.data_source_adapter import load_datasets
from .lgd_calculation import (
    CashFlowLendingLGDEngine,
    CommercialLGDEngine,
    DevelopmentLGDEngine,
    MortgageLGDEngine,
)
from .overlay_parameters import OverlayParameterManager
from .product_routing import _resolve_product

# Subtypes that route to DevelopmentLGDEngine.
_DEVELOPMENT_SUBTYPES = frozenset({"development_finance", "residual_stock", "land_subdivision"})

# The single cashflow_lending sub_type that routes to CashFlowLendingLGDEngine.
# All other cashflow_lending subtypes route to CommercialLGDEngine.
_CASHFLOW_LENDING_GENERIC = "cashflow_lending"


@dataclass(frozen=True)
class ProductSchema:
    required_columns: tuple[str, ...]
    explicit_defaults: dict[str, Any]
    numeric_bounds: dict[str, tuple[float | None, float | None]]
    categorical_allowed: dict[str, set[Any]]


SCHEMAS: dict[str, ProductSchema] = {
    "mortgage": ProductSchema(
        required_columns=("loan_id", "ead", "realised_lgd", "lmi_eligible", "mortgage_class"),
        explicit_defaults={
            "ltv_at_default": 0.85,
            "dti": 0.35,
            "occupancy": "owner_occupied",
            "resolution_type": "Property Sale",
        },
        numeric_bounds={
            "ead": (0, None),
            "realised_lgd": (0, 2),
            "ltv_at_default": (0, None),
            "dti": (0, None),
        },
        categorical_allowed={
            "lmi_eligible": {0, 1},
            "mortgage_class": {"Standard", "Non-Standard"},
        },
    ),
    "commercial": ProductSchema(
        required_columns=("loan_id", "ead", "realised_lgd", "security_type", "seniority"),
        explicit_defaults={
            "security_coverage_ratio": 1.0,
            "icr": 1.5,
            "workout_months": 18,
            "industry": "Unknown",
        },
        numeric_bounds={
            "ead": (0, None),
            "realised_lgd": (0, 2),
            "security_coverage_ratio": (0, 5),
            "icr": (0, 50),
            "workout_months": (0, 240),
        },
        categorical_allowed={},
    ),
    "development": ProductSchema(
        required_columns=("loan_id", "ead", "realised_lgd", "completion_stage"),
        explicit_defaults={
            "presale_coverage": 0.50,
            "lvr_as_if_complete": 0.70,
            "industry": "Construction",
        },
        numeric_bounds={
            "ead": (0, None),
            "realised_lgd": (0, 2),
            "presale_coverage": (0, 1),
            "lvr_as_if_complete": (0, 5),
        },
        categorical_allowed={},
    ),
    "cashflow_lending": ProductSchema(
        required_columns=("loan_id", "ead", "realised_lgd", "pd_score_band", "cashflow_product", "seniority"),
        explicit_defaults={
            "dscr": 1.3,
            "conduct_classification": "Amber",
            "industry": "Unknown",
        },
        numeric_bounds={
            "ead": (0, None),
            "realised_lgd": (0, 2),
            "dscr": (0, 50),
        },
        categorical_allowed={
            "pd_score_band": {"A", "B", "C", "D", "E"},
            "conduct_classification": {"Green", "Amber", "Red"},
        },
    ),
}


NORMALIZED_OUTPUT_COLUMNS = [
    "loan_id",
    "product_type",
    "lgd_base",
    "lgd_downturn",
    "lgd_final",
    "macro_downturn_scalar",
    "industry_downturn_adjustment",
    "combined_downturn_scalar",
    "overlay_source",
    "parameter_version",
    "parameter_hash",
    "scenario_id",
    "source_mode",
    "seed",
    "model_option_used",
    "scored_at_utc",
]


def _schema_key(family: str, sub_type: str) -> str:
    """Map (family, sub_type) to a SCHEMAS dict key."""
    if family == "mortgage":
        return "mortgage"
    if sub_type == _CASHFLOW_LENDING_GENERIC:
        return "cashflow_lending"
    if sub_type in _DEVELOPMENT_SUBTYPES:
        return "development"
    return "commercial"


def _dataset_key(family: str, sub_type: str) -> str:
    """Map (family, sub_type) to the legacy data_source_adapter product key."""
    if family == "mortgage":
        return "mortgage"
    if sub_type == _CASHFLOW_LENDING_GENERIC:
        return "cashflow_lending"
    if sub_type in _DEVELOPMENT_SUBTYPES:
        return "development"
    return "commercial"


def _validate_bounds(df: pd.DataFrame, product: str, schema: ProductSchema) -> None:
    for col, (lo, hi) in schema.numeric_bounds.items():
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        na_mask = series.isna()
        if na_mask.any():
            bad_idx = df.index[na_mask].tolist()[:10]
            raise ValueError(
                f"{product}: column '{col}' has non-numeric/null values. "
                f"Count: {na_mask.sum()}. First offending indices: {bad_idx}"
            )
        if lo is not None:
            lo_mask = series < lo
            if lo_mask.any():
                bad_idx = df.index[lo_mask].tolist()[:10]
                raise ValueError(
                    f"{product}: column '{col}' has {lo_mask.sum()} value(s) below {lo}. "
                    f"First offending indices: {bad_idx}"
                )
        if hi is not None:
            hi_mask = series > hi
            if hi_mask.any():
                bad_idx = df.index[hi_mask].tolist()[:10]
                raise ValueError(
                    f"{product}: column '{col}' has {hi_mask.sum()} value(s) above {hi}. "
                    f"First offending indices: {bad_idx}"
                )


def _validate_categories(df: pd.DataFrame, product: str, schema: ProductSchema) -> None:
    for col, allowed in schema.categorical_allowed.items():
        if col not in df.columns:
            continue
        values = set(df[col].dropna().unique().tolist())
        invalid = sorted(v for v in values if v not in allowed)
        if invalid:
            raise ValueError(f"{product}: column '{col}' has invalid values {invalid}; allowed={sorted(allowed)}")


def _apply_explicit_defaults(df: pd.DataFrame, schema: ProductSchema) -> pd.DataFrame:
    out = df.copy()
    for col, default in schema.explicit_defaults.items():
        if col not in out.columns:
            logger.warning(
                "Column '%s' is absent from input; filling all rows with default value %r",
                col,
                default,
            )
            out[col] = default
        else:
            na_mask = out[col].isna()
            if na_mask.any():
                logger.warning(
                    "Column '%s': %d null value(s) filled with default %r",
                    col,
                    na_mask.sum(),
                    default,
                )
                out[col] = out[col].where(~na_mask, default)
    return out


def validate_scoring_inputs(df: pd.DataFrame, product_type: str) -> pd.DataFrame:
    family, sub_type = _resolve_product(product_type)
    schema = SCHEMAS[_schema_key(family, sub_type)]
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    if len(df) == 0:
        raise ValueError("Input DataFrame must not be empty")

    missing = [c for c in schema.required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"{sub_type}: missing required columns {missing}")

    out = _apply_explicit_defaults(df, schema)
    if out["loan_id"].isna().any():
        raise ValueError(f"{sub_type}: 'loan_id' must not contain null values")

    _validate_bounds(out, sub_type, schema)
    _validate_categories(out, sub_type, schema)
    return out


def _build_engine(
    family: str,
    sub_type: str,
    parameter_manager: OverlayParameterManager,
    scenario_id: str,
):
    if family == "mortgage":
        return MortgageLGDEngine(parameter_manager=parameter_manager, scenario_id=scenario_id), "apply_apra_overlays"
    if sub_type == _CASHFLOW_LENDING_GENERIC:
        return CashFlowLendingLGDEngine(parameter_manager=parameter_manager, scenario_id=scenario_id), "apply_overlays"
    if sub_type in _DEVELOPMENT_SUBTYPES:
        return DevelopmentLGDEngine(parameter_manager=parameter_manager, scenario_id=scenario_id), "apply_overlays"
    # commercial_cashflow, receivables, trade_contingent, asset_equipment,
    # cre_investment, bridging, mezz_second_mortgage
    return CommercialLGDEngine(parameter_manager=parameter_manager, scenario_id=scenario_id), "apply_overlays"


def _normalize_scoring_output(
    scored: pd.DataFrame,
    product: str,
    parameter_manager: OverlayParameterManager,
    scenario_id: str,
    source_mode: str,
    seed: int,
) -> pd.DataFrame:
    out = pd.DataFrame(index=scored.index)
    base_series = (
        scored["lgd_base"]
        if "lgd_base" in scored.columns
        else scored["lgd_industry_adjusted"]
        if "lgd_industry_adjusted" in scored.columns
        else scored.get("realised_lgd")
    )
    out["loan_id"] = scored["loan_id"]
    out["product_type"] = product
    out["lgd_base"] = pd.to_numeric(base_series, errors="coerce").clip(0, 1)
    out["lgd_downturn"] = pd.to_numeric(scored.get("lgd_downturn"), errors="coerce").clip(0, 1)
    out["lgd_final"] = pd.to_numeric(scored.get("lgd_final"), errors="coerce").clip(0, 1)
    out["macro_downturn_scalar"] = pd.to_numeric(scored.get("macro_downturn_scalar"), errors="coerce")
    out["industry_downturn_adjustment"] = pd.to_numeric(
        scored.get("industry_downturn_adjustment"), errors="coerce"
    )
    out["combined_downturn_scalar"] = pd.to_numeric(scored.get("combined_downturn_scalar"), errors="coerce")
    out["overlay_source"] = scored.get("overlay_source", "unknown").astype(str)
    out["parameter_version"] = parameter_manager.meta.version
    out["parameter_hash"] = parameter_manager.meta.parameter_hash
    out["scenario_id"] = str(scenario_id)
    out["source_mode"] = str(source_mode)
    out["seed"] = int(seed)
    out["model_option_used"] = "proxy_component_chain_v1"
    out["scored_at_utc"] = pd.Timestamp.utcnow().isoformat()
    return out[NORMALIZED_OUTPUT_COLUMNS]


def score_batch_loans(
    df: pd.DataFrame,
    product_type: str,
    scenario_id: str = "baseline",
    seed: int = 42,
    source_mode: str = "generated",
    parameter_manager: OverlayParameterManager | None = None,
    return_full: bool = False,
) -> pd.DataFrame:
    family, sub_type = _resolve_product(product_type)
    clean = validate_scoring_inputs(df, sub_type)
    pm = parameter_manager or OverlayParameterManager()

    engine, method_name = _build_engine(family, sub_type, pm, scenario_id=scenario_id)
    method = getattr(engine, method_name)
    scored = method(clean)

    normalized = _normalize_scoring_output(
        scored=scored,
        product=sub_type,
        parameter_manager=pm,
        scenario_id=scenario_id,
        source_mode=source_mode,
        seed=seed,
    )
    if return_full:
        merged = scored.copy()
        for col in normalized.columns:
            merged[col] = normalized[col]
        return merged
    return normalized


def score_single_loan(
    payload: dict[str, Any],
    product_type: str,
    scenario_id: str = "baseline",
    seed: int = 42,
    source_mode: str = "generated",
    parameter_manager: OverlayParameterManager | None = None,
    return_full: bool = False,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict")
    df = pd.DataFrame([payload])
    scored = score_batch_loans(
        df=df,
        product_type=product_type,
        scenario_id=scenario_id,
        seed=seed,
        source_mode=source_mode,
        parameter_manager=parameter_manager,
        return_full=return_full,
    )
    return scored.iloc[0].to_dict()


def _product_template_row(
    product_type: str,
    source_mode: str = "generated",
    controlled_root: str | Path = "data/controlled",
) -> pd.DataFrame:
    family, sub_type = _resolve_product(product_type)
    dkey = _dataset_key(family, sub_type)
    datasets = load_datasets(
        source=source_mode,
        controlled_root=controlled_root,
        require_all_products=False,
    )
    if dkey not in datasets:
        raise ValueError(f"{source_mode}: no dataset available for product '{sub_type}' (dataset key '{dkey}')")
    loans = datasets[dkey]["loans"]
    if loans is None or len(loans) == 0:
        raise ValueError(f"{source_mode}: empty loans table for product '{sub_type}' (dataset key '{dkey}')")
    return loans.head(1).copy()


def score_single_loan_from_source_template(
    payload: dict[str, Any],
    product_type: str,
    source_mode: str = "generated",
    controlled_root: str | Path = "data/controlled",
    scenario_id: str = "baseline",
    seed: int = 42,
    parameter_manager: OverlayParameterManager | None = None,
    return_full: bool = False,
) -> dict[str, Any]:
    template = _product_template_row(
        product_type=product_type,
        source_mode=source_mode,
        controlled_root=controlled_root,
    )
    template = template.iloc[:1].copy()
    for k, v in payload.items():
        template[k] = [v]
    if "loan_id" not in template.columns:
        template["loan_id"] = payload.get("loan_id", "single_loan")
    return score_batch_loans(
        df=template,
        product_type=product_type,
        scenario_id=scenario_id,
        seed=seed,
        source_mode=source_mode,
        parameter_manager=parameter_manager,
        return_full=return_full,
    ).iloc[0].to_dict()


def score_batch_from_source(
    product_type: str,
    source_mode: str = "generated",
    controlled_root: str | Path = "data/controlled",
    scenario_id: str = "baseline",
    seed: int = 42,
    parameter_manager: OverlayParameterManager | None = None,
    return_full: bool = False,
) -> pd.DataFrame:
    family, sub_type = _resolve_product(product_type)
    dkey = _dataset_key(family, sub_type)
    datasets = load_datasets(
        source=source_mode,
        controlled_root=controlled_root,
        require_all_products=False,
    )
    if dkey not in datasets:
        raise ValueError(f"{source_mode}: no loans dataset for product '{sub_type}' (dataset key '{dkey}')")
    loans = datasets[dkey]["loans"]
    return score_batch_loans(
        df=loans,
        product_type=sub_type,
        scenario_id=scenario_id,
        seed=seed,
        source_mode=source_mode,
        parameter_manager=parameter_manager,
        return_full=return_full,
    )
