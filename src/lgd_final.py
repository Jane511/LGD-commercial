"""
Final LGD layer builder for EL-ready facility-level outputs.

This module adds a simplified final LGD layer on top of the detailed APRA LGD
framework. It standardises the three product datasets into one portfolio view,
applies a compact set of base LGD assumptions and risk-driver adjustments, and
produces a clean `lgd_final` field for downstream Expected Loss use.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data_generation import generate_all_datasets


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"
DEFAULT_DOWNTURN_SCALAR = 1.05

# Simple lookup requested in the final-layer instructions, extended to cover
# development finance so all repo products can feed one EL-ready dataset.
BASE_LGD_LOOKUP = {
    ("property", "residential"): 0.20,
    ("property", "commercial"): 0.35,
    ("sme_cashflow", "secured"): 0.45,
    ("sme_cashflow", "partially_secured"): 0.55,
    ("sme_cashflow", "unsecured"): 0.65,
    ("overdraft", "unsecured"): 0.75,
    ("development", "secured"): 0.40,
    # Cash flow lending products (unsecured / receivables-secured)
    ("cashflow_lending", "secured"): 0.40,
    ("cashflow_lending", "partially_secured"): 0.50,
    ("cashflow_lending", "unsecured"): 0.60,
}

# PD score band adjustment for cash flow lending (additive)
PD_BAND_LGD_ADDON = {
    "A": -0.05,
    "B": -0.02,
    "C":  0.00,
    "D":  0.03,
    "E":  0.06,
}

HIGH_RISK_INDUSTRY_TOKENS = (
    "construction",
    "accommodation",
    "food",
    "hospitality",
    "manufacturing",
)

REQUIRED_COLUMNS = [
    "loan_id",
    "product_type",
    "security_type",
    "property_type",
    "property_value",
    "current_lvr",
    "loan_stage",
    "industry",
    "ead",
]

OUTPUT_COLUMNS = [
    "loan_id",
    "source_product",
    "source_loan_id",
    "product_type",
    "security_type",
    "property_type",
    "property_value",
    "current_lvr",
    "loan_stage",
    "industry",
    "ead",
    "pd_score_band",
    "dscr",
    "conduct_classification",
    "house_price_decline",
    "unemployment_shock",
    "rate_shock",
    "value_decline",
    "cashflow_weakness",
    "recovery_delay",
    "grv_decline",
    "cost_overrun",
    "sell_through_delay",
    "lgd_base",
    "lgd_adj_lvr",
    "lgd_adj_stage",
    "lgd_adj_industry",
    "lgd_adj_pd_band",
    "lgd_adj_dscr",
    "lgd_adj_conduct",
    "lgd_adjusted",
    "downturn_scalar",
    "lgd_downturn",
    "lgd_final",
]


def _normalise_label(value) -> str:
    """Normalise text labels for lookup logic."""
    if pd.isna(value):
        return ""
    return str(value).strip().lower().replace("-", "_").replace("/", "_").replace("&", "and")


def _classify_commercial_security(row) -> str:
    """
    Collapse repo-specific commercial collateral detail into simple secured /
    partially secured / unsecured buckets for the final layer.
    """
    if _normalise_label(row.get("facility_type")) == "overdraft":
        return "unsecured"

    seniority = _normalise_label(row.get("seniority"))
    coverage = row.get("security_coverage_ratio")

    if seniority == "senior_unsecured":
        return "unsecured"
    if pd.isna(coverage):
        return "partially_secured"
    if coverage >= 1.0:
        return "secured"
    if coverage >= 0.5:
        return "partially_secured"
    return "unsecured"


def _normalise_development_stage(stage) -> str:
    """Map detailed development stages to the simple early / mid / late buckets."""
    label = _normalise_label(stage)
    if label in {"pre_construction", "early_construction", "early"}:
        return "early"
    if label in {"mid_construction", "mid"}:
        return "mid"
    if label:
        return "late"
    return ""


def _prepare_mortgage_inputs(df: pd.DataFrame) -> pd.DataFrame:
    if {"property_value_orig", "property_value_at_default"}.issubset(df.columns):
        house_price_decline = (
            (df["property_value_orig"] - df["property_value_at_default"])
            / df["property_value_orig"].replace(0, np.nan)
        ).clip(lower=0).fillna(0.0)
    else:
        lvr = pd.to_numeric(df.get("ltv_at_default", pd.Series(0.80, index=df.index)), errors="coerce").fillna(0.80)
        house_price_decline = ((lvr - 0.80) / 0.40).clip(lower=0, upper=0.30)
    out = pd.DataFrame(
        {
            "loan_id": df["loan_id"].map(lambda x: f"MTG-{int(x)}"),
            "source_product": "Mortgage",
            "source_loan_id": df["loan_id"],
            "product_type": "property",
            "security_type": "residential",
            "property_type": df["property_type"],
            "property_value": df["property_value_at_default"],
            "current_lvr": df["ltv_at_default"],
            "loan_stage": pd.NA,
            "industry": pd.NA,
            "ead": df["ead"],
            "house_price_decline": df.get("house_price_decline", house_price_decline),
            "unemployment_shock": df.get("unemployment_shock", 0.02),
            "rate_shock": df.get("rate_shock", 0.01),
        }
    )
    return out


def _prepare_commercial_inputs(df: pd.DataFrame) -> pd.DataFrame:
    security_bucket = df.apply(_classify_commercial_security, axis=1)
    product_type = df["facility_type"].map(
        lambda value: "overdraft" if _normalise_label(value) == "overdraft" else "sme_cashflow"
    )
    current_lvr = (df["ead"] / df["collateral_value"]).where(df["collateral_value"] > 0)
    coverage = pd.to_numeric(df.get("security_coverage_ratio", pd.Series(1.0, index=df.index)), errors="coerce").fillna(1.0)
    icr = pd.to_numeric(df.get("icr", pd.Series(1.5, index=df.index)), errors="coerce").fillna(1.5)
    workout_months = pd.to_numeric(df.get("workout_months", pd.Series(18, index=df.index)), errors="coerce").fillna(18)

    out = pd.DataFrame(
        {
            "loan_id": df["loan_id"].map(lambda x: f"COM-{int(x)}"),
            "source_product": "Commercial",
            "source_loan_id": df["loan_id"],
            "product_type": product_type,
            "security_type": security_bucket,
            "property_type": pd.NA,
            "property_value": df["collateral_value"],
            "current_lvr": current_lvr,
            "loan_stage": pd.NA,
            "industry": df["industry"],
            "ead": df["ead"],
            "value_decline": (1 - coverage).clip(lower=0),
            "cashflow_weakness": ((1.5 - icr) / 1.5).clip(lower=0, upper=1),
            "recovery_delay": ((workout_months - 18) / 24).clip(lower=0, upper=1),
        }
    )
    return out


def _prepare_development_inputs(df: pd.DataFrame) -> pd.DataFrame:
    grv = pd.to_numeric(df.get("grv", df.get("as_if_complete_value")), errors="coerce")
    as_is = pd.to_numeric(df.get("as_is_value", df.get("property_value_at_default")), errors="coerce")
    cost_to_complete = pd.to_numeric(df.get("cost_to_complete", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    workout_months = pd.to_numeric(df.get("workout_months", pd.Series(18, index=df.index)), errors="coerce").fillna(18)
    out = pd.DataFrame(
        {
            "loan_id": df["loan_id"].map(lambda x: f"DEV-{int(x)}"),
            "source_product": "Development",
            "source_loan_id": df["loan_id"],
            "product_type": "development",
            "security_type": "secured",
            "property_type": df["development_type"],
            "property_value": df["as_if_complete_value"],
            "current_lvr": df["lvr_as_if_complete"],
            "loan_stage": df["completion_stage"].map(_normalise_development_stage),
            "industry": df["industry"],
            "ead": df["ead"],
            "grv_decline": df.get(
                "grv_decline",
                ((grv - as_is) / grv.replace(0, np.nan)).clip(lower=0),
            ),
            "cost_overrun": (cost_to_complete / df["ead"].replace(0, np.nan)).clip(lower=0),
            "sell_through_delay": ((workout_months - 18) / 18).clip(lower=0, upper=1),
        }
    )
    return out


def _prepare_cashflow_lending_inputs(df: pd.DataFrame) -> pd.DataFrame:
    security_bucket = df["has_receivables_security"].map(
        {1: "secured", 0: "unsecured"}
    )
    # Partially secured: unsecured with some coverage
    security_bucket = security_bucket.where(
        ~((security_bucket == "unsecured") & (df["security_coverage_ratio"] > 0.20)),
        "partially_secured",
    )
    dscr = pd.to_numeric(df.get("dscr", pd.Series(1.3, index=df.index)), errors="coerce").fillna(1.3)
    workout_months = pd.to_numeric(df.get("workout_months", pd.Series(14, index=df.index)), errors="coerce").fillna(14)
    out = pd.DataFrame(
        {
            "loan_id": df["loan_id"].map(lambda x: f"CFL-{int(x)}"),
            "source_product": "Cashflow Lending",
            "source_loan_id": df["loan_id"],
            "product_type": "cashflow_lending",
            "security_type": security_bucket,
            "property_type": df["cashflow_product"],
            "property_value": df["ead"] * df["security_coverage_ratio"],
            "current_lvr": (1.0 / df["security_coverage_ratio"].replace(0, float("inf"))).clip(0, 5),
            "loan_stage": pd.NA,
            "industry": df["industry"],
            "ead": df["ead"],
            "pd_score_band": df["pd_score_band"],
            "dscr": dscr,
            "conduct_classification": df["conduct_classification"],
            "cashflow_weakness": ((1.3 - dscr) / 1.3).clip(lower=0, upper=1),
            "recovery_delay": ((workout_months - 14) / 20).clip(lower=0, upper=1),
        }
    )
    return out


def load_repo_portfolio_inputs(raw_dir: str | Path | None = None) -> pd.DataFrame:
    """
    Load raw repo loan files and standardise them into one final-layer input set.

    If the raw CSVs are not present, synthetic datasets are generated in-memory.
    """
    raw_path = Path(raw_dir) if raw_dir is not None else DEFAULT_RAW_DIR
    file_map = {
        "mortgage": raw_path / "mortgage_loans.csv",
        "commercial": raw_path / "commercial_loans.csv",
        "development": raw_path / "development_loans.csv",
        "cashflow_lending": raw_path / "cashflow_lending_loans.csv",
    }

    if all(path.exists() for path in file_map.values()):
        mortgage = pd.read_csv(file_map["mortgage"])
        commercial = pd.read_csv(file_map["commercial"])
        development = pd.read_csv(file_map["development"])
        cashflow = pd.read_csv(file_map["cashflow_lending"])
    else:
        datasets = generate_all_datasets()
        mortgage = datasets["mortgage"]["loans"]
        commercial = datasets["commercial"]["loans"]
        development = datasets["development"]["loans"]
        cashflow = datasets["cashflow_lending"]["loans"]

    portfolio_inputs = pd.concat(
        [
            _prepare_mortgage_inputs(mortgage),
            _prepare_commercial_inputs(commercial),
            _prepare_development_inputs(development),
            _prepare_cashflow_lending_inputs(cashflow),
        ],
        ignore_index=True,
    )
    return portfolio_inputs


def assign_base_lgd(row: pd.Series) -> float:
    """Assign a simple base LGD by product and security type."""
    product = _normalise_label(row["product_type"])
    security = _normalise_label(row["security_type"])

    if product in {"mortgage", "home_loan", "residential_mortgage"}:
        product, security = "property", "residential"
    elif product in {"commercial", "commercial_cashflow", "sme"}:
        product = "sme_cashflow"
    elif product in {"development_finance"}:
        product = "development"
    elif product in {"cashflow", "cashflow_lending", "cash_flow_lending"}:
        product = "cashflow_lending"

    if product == "overdraft":
        security = "unsecured"
    elif product == "property" and security not in {"residential", "commercial"}:
        security = "commercial"
    elif product == "development" and security != "secured":
        security = "secured"
    elif product == "cashflow_lending" and security not in {"secured", "partially_secured", "unsecured"}:
        security = "unsecured"

    key = (product, security)
    if key in BASE_LGD_LOOKUP:
        return BASE_LGD_LOOKUP[key]

    if product == "sme_cashflow":
        return BASE_LGD_LOOKUP[(product, "partially_secured")]
    if product == "cashflow_lending":
        return BASE_LGD_LOOKUP[(product, "unsecured")]

    raise ValueError(
        f"Unsupported product/security combination for loan {row.get('loan_id')}: "
        f"{row['product_type']} / {row['security_type']}"
    )


def lvr_adjustment(lvr) -> float:
    """Additive LGD adjustment for higher leverage."""
    if pd.isna(lvr):
        return 0.0
    if float(lvr) > 0.80:
        return 0.05
    if float(lvr) > 0.60:
        return 0.02
    return 0.0


def stage_adjustment(stage) -> float:
    """Additive LGD adjustment for development-stage risk."""
    normalised_stage = _normalise_development_stage(stage)
    if normalised_stage == "early":
        return 0.08
    if normalised_stage == "mid":
        return 0.04
    return 0.0


def industry_adjustment(industry) -> float:
    """Simple high-risk industry overlay."""
    label = _normalise_label(industry)
    if any(token in label for token in HIGH_RISK_INDUSTRY_TOKENS):
        return 0.03
    return 0.0


def _ead_weighted_average(values: pd.Series, ead: pd.Series) -> float:
    total = pd.to_numeric(ead, errors="coerce").fillna(0.0).sum()
    if total <= 0:
        return 0.0
    vals = pd.to_numeric(values, errors="coerce").fillna(0.0)
    return float((vals * pd.to_numeric(ead, errors="coerce").fillna(0.0)).sum() / total)


def _build_macro_downturn_scalar(out: pd.DataFrame, base_scalar: float) -> pd.Series:
    """
    Build product-specific macro-linked downturn scalar.

    Product rules:
      - Mortgage: house price decline, unemployment, rate shock
      - Commercial: value decline, weaker cashflow, slower recovery
      - Development: GRV decline, cost overrun, slower sell-through
      - Cashflow lending: weaker cashflow, slower recovery
    """
    scalar = pd.Series(base_scalar, index=out.index, dtype=float)
    source = out["source_product"].astype(str)

    m_mask = source.eq("Mortgage")
    if m_mask.any():
        hpd = pd.to_numeric(out.loc[m_mask, "house_price_decline"], errors="coerce").fillna(0.0).clip(0, 0.40)
        unemp = pd.to_numeric(out.loc[m_mask, "unemployment_shock"], errors="coerce").fillna(0.02).clip(0, 0.06)
        rate = pd.to_numeric(out.loc[m_mask, "rate_shock"], errors="coerce").fillna(0.01).clip(0, 0.05)
        scalar.loc[m_mask] = base_scalar * (1 + 0.40 * hpd + 1.20 * unemp + 1.60 * rate)

    c_mask = source.eq("Commercial")
    if c_mask.any():
        value_decline = pd.to_numeric(out.loc[c_mask, "value_decline"], errors="coerce").fillna(0.0).clip(0, 0.50)
        cashflow_weakness = pd.to_numeric(out.loc[c_mask, "cashflow_weakness"], errors="coerce").fillna(0.0).clip(0, 1.0)
        recovery_delay = pd.to_numeric(out.loc[c_mask, "recovery_delay"], errors="coerce").fillna(0.0).clip(0, 1.0)
        scalar.loc[c_mask] = base_scalar * (
            1 + 0.30 * value_decline + 0.10 * cashflow_weakness + 0.08 * recovery_delay
        )

    d_mask = source.eq("Development")
    if d_mask.any():
        grv_decline = pd.to_numeric(out.loc[d_mask, "grv_decline"], errors="coerce").fillna(0.0).clip(0, 0.60)
        cost_overrun = pd.to_numeric(out.loc[d_mask, "cost_overrun"], errors="coerce").fillna(0.0).clip(0, 0.40)
        sell_delay = pd.to_numeric(out.loc[d_mask, "sell_through_delay"], errors="coerce").fillna(0.0).clip(0, 1.0)
        scalar.loc[d_mask] = base_scalar * (
            1 + 0.35 * grv_decline + 0.25 * cost_overrun + 0.12 * sell_delay
        )

    cf_mask = source.eq("Cashflow Lending")
    if cf_mask.any():
        cashflow_weakness = pd.to_numeric(out.loc[cf_mask, "cashflow_weakness"], errors="coerce").fillna(0.0).clip(0, 1.0)
        recovery_delay = pd.to_numeric(out.loc[cf_mask, "recovery_delay"], errors="coerce").fillna(0.0).clip(0, 1.0)
        scalar.loc[cf_mask] = base_scalar * (1 + 0.10 * cashflow_weakness + 0.06 * recovery_delay)

    return scalar.clip(1.00, 1.90)


def build_final_lgd_layer(
    df: pd.DataFrame,
    downturn_scalar: float = DEFAULT_DOWNTURN_SCALAR,
) -> pd.DataFrame:
    """
    Build the simplified final LGD layer from an EL-ready input dataset.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for LGD final layer: {missing}")

    out = df.copy()
    macro_cols = [
        "house_price_decline",
        "unemployment_shock",
        "rate_shock",
        "value_decline",
        "cashflow_weakness",
        "recovery_delay",
        "grv_decline",
        "cost_overrun",
        "sell_through_delay",
    ]
    for col in macro_cols:
        if col not in out.columns:
            out[col] = pd.NA

    out["lgd_base"] = out.apply(assign_base_lgd, axis=1)
    out["lgd_adj_lvr"] = out["current_lvr"].apply(lvr_adjustment)
    out["lgd_adj_stage"] = out["loan_stage"].apply(stage_adjustment)
    out["lgd_adj_industry"] = out["industry"].apply(industry_adjustment)

    # PD score band adjustment (cash flow lending only)
    if "pd_score_band" not in out.columns:
        out["pd_score_band"] = pd.NA
    out["lgd_adj_pd_band"] = out["pd_score_band"].map(PD_BAND_LGD_ADDON).fillna(0.0)

    # DSCR stress adjustment (cash flow lending only)
    if "dscr" not in out.columns:
        out["dscr"] = pd.NA
    dscr_vals = pd.to_numeric(out["dscr"], errors="coerce")
    out["lgd_adj_dscr"] = (((1.3 - dscr_vals) * 0.02).clip(lower=0)).fillna(0.0)

    # Conduct overlay (cash flow lending only)
    if "conduct_classification" not in out.columns:
        out["conduct_classification"] = pd.NA
    conduct_map = {"Green": 0.000, "Amber": 0.005, "Red": 0.015}
    out["lgd_adj_conduct"] = out["conduct_classification"].map(conduct_map).fillna(0.0)

    out["lgd_adjusted"] = (
        out["lgd_base"]
        + out["lgd_adj_lvr"]
        + out["lgd_adj_stage"]
        + out["lgd_adj_industry"]
        + out["lgd_adj_pd_band"]
        + out["lgd_adj_dscr"]
        + out["lgd_adj_conduct"]
    )
    out["downturn_scalar"] = _build_macro_downturn_scalar(out, base_scalar=downturn_scalar)
    out["lgd_downturn"] = out["lgd_adjusted"] * out["downturn_scalar"]
    out["lgd_final"] = out["lgd_downturn"].clip(0, 1)

    extra_columns = [
        column for column in out.columns if column not in OUTPUT_COLUMNS
    ]
    return out[OUTPUT_COLUMNS + extra_columns]


def summarise_final_lgd_by_product(df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact product-level summary for validation and handoff."""
    summary = (
        df.groupby(["source_product", "product_type"], observed=True)
        .apply(
            lambda group: pd.Series(
                {
                    "loan_count": len(group),
                    "total_ead": group["ead"].sum(),
                    "ead_weighted_lgd_base": _ead_weighted_average(group["lgd_base"], group["ead"]),
                    "ead_weighted_lgd_final": _ead_weighted_average(group["lgd_final"], group["ead"]),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    return summary.sort_values(["source_product", "product_type"]).reset_index(drop=True)


def validate_final_lgd_layer(df: pd.DataFrame) -> pd.DataFrame:
    """Run the sanity checks requested in the final-layer instructions."""
    commercial_like = df[df["product_type"].isin(["sme_cashflow", "overdraft"])].copy()
    secured = commercial_like[commercial_like["security_type"] == "secured"]
    unsecured = commercial_like[commercial_like["security_type"] == "unsecured"]
    secured_weighted = _ead_weighted_average(secured["lgd_final"], secured["ead"]) if len(secured) else np.nan
    unsecured_weighted = _ead_weighted_average(unsecured["lgd_final"], unsecured["ead"]) if len(unsecured) else np.nan

    checks = [
        {
            "check_name": "no_negative_lgd",
            "passed": bool((df["lgd_final"] >= 0).all()),
            "detail": f"min_lgd_final={df['lgd_final'].min():.4f}",
        },
        {
            "check_name": "no_lgd_above_100pct",
            "passed": bool((df["lgd_final"] <= 1).all()),
            "detail": f"max_lgd_final={df['lgd_final'].max():.4f}",
        },
        {
            "check_name": "downturn_not_below_adjusted",
            "passed": bool((df["lgd_downturn"] >= df["lgd_adjusted"]).all()),
            "detail": (
                f"downturn_scalar_range={df['downturn_scalar'].min():.2f}"
                f" to {df['downturn_scalar'].max():.2f}"
            ),
        },
        {
            "check_name": "secured_below_unsecured",
            "passed": bool(
                pd.notna(secured_weighted)
                and pd.notna(unsecured_weighted)
                and secured_weighted < unsecured_weighted
            ),
            "detail": (
                f"secured_weighted={secured_weighted:.4f}; "
                f"unsecured_weighted={unsecured_weighted:.4f}"
                if pd.notna(secured_weighted) and pd.notna(unsecured_weighted)
                else "insufficient commercial observations"
            ),
        },
    ]
    return pd.DataFrame(checks)


def build_and_save_repo_final_lgd(
    raw_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    downturn_scalar: float = DEFAULT_DOWNTURN_SCALAR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build the repo's final LGD layer and save the CSV outputs.
    """
    portfolio_inputs = load_repo_portfolio_inputs(raw_dir=raw_dir)
    final_lgd = build_final_lgd_layer(portfolio_inputs, downturn_scalar=downturn_scalar)
    summary = summarise_final_lgd_by_product(final_lgd)
    checks = validate_final_lgd_layer(final_lgd)

    target_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    final_lgd.to_csv(target_dir / "lgd_final.csv", index=False)
    summary.to_csv(target_dir / "lgd_final_summary_by_product.csv", index=False)
    checks.to_csv(target_dir / "lgd_final_validation_checks.csv", index=False)

    return final_lgd, summary, checks


if __name__ == "__main__":  # pragma: no cover
    # Run via: python -m src.lgd_final
    import sys
    _final, _summary, _checks = build_and_save_repo_final_lgd()
    print(f"Final LGD built: {len(_final)} rows")
    print(_summary.to_string(index=False))
    sys.exit(0)


def main() -> None:
    """CLI entry point for building the final LGD layer outputs."""
    final_lgd, summary, checks = build_and_save_repo_final_lgd()

    print(
        f"Saved {len(final_lgd)} rows to "
        f"{DEFAULT_OUTPUT_DIR / 'lgd_final.csv'}"
    )
    print("\nAverage final LGD by product:")
    print(summary.to_string(index=False))
    print("\nValidation checks:")
    print(checks.to_string(index=False))


if __name__ == "__main__":
    main()
