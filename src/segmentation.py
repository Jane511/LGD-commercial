from __future__ import annotations

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _industry_risk_band(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    bins = [0, 2.5, 3.0, 5.0]
    labels = ["Low", "Medium", "Elevated"]
    return pd.cut(vals, bins=bins, labels=labels, right=True).astype("object")


def _series_or_default(df: pd.DataFrame, col: str, default: str, seg_col: str = "") -> pd.Series:
    if col in df.columns:
        result = df[col].astype(str)
        if seg_col and logger.isEnabledFor(logging.WARNING):
            unknown_count = (result == default).sum()
            if unknown_count > 0:
                logger.warning(
                    "Segment '%s': %d row(s) have unrecognised/null source values in '%s' — defaulting to 'Unknown'",
                    seg_col,
                    unknown_count,
                    col,
                )
        return result
    if seg_col:
        logger.warning(
            "Segment '%s': source column '%s' is absent — all %d row(s) default to %r",
            seg_col,
            col,
            len(df),
            default,
        )
    return pd.Series(default, index=df.index, dtype=object)


def apply_standard_segments(df: pd.DataFrame, product: str) -> pd.DataFrame:
    """Apply cross-module standardized segment tags while keeping existing columns."""
    out = df.copy()

    out["std_module"] = product
    out["std_security_or_stage_band"] = "Unknown"
    out["std_product_segment"] = "Unknown"
    out["std_cashflow_family"] = "NotApplicable"
    out["std_property_backed_subtype"] = "NotApplicable"

    if product == "mortgage":
        out["std_product_segment"] = _series_or_default(out, "mortgage_class", "Unknown", "std_product_segment")
        out["std_security_or_stage_band"] = _series_or_default(out, "ltv_band", "Unknown", "std_security_or_stage_band")
        out["std_property_backed_subtype"] = "Residential Mortgage"

    elif product == "commercial":
        out["std_product_segment"] = _series_or_default(out, "security_type", "Unknown", "std_product_segment")
        out["std_security_or_stage_band"] = _series_or_default(out, "coverage_band", "Unknown", "std_security_or_stage_band")
        out["std_cashflow_family"] = _series_or_default(out, "security_type", "Unknown", "std_cashflow_family")

    elif product == "development":
        out["std_product_segment"] = _series_or_default(out, "completion_stage", "Unknown", "std_product_segment")
        out["std_security_or_stage_band"] = _series_or_default(out, "presale_band", "Unknown", "std_security_or_stage_band")
        out["std_property_backed_subtype"] = _series_or_default(out, "development_type", "Unknown", "std_property_backed_subtype")

    elif product == "cashflow_lending":
        out["std_product_segment"] = _series_or_default(out, "cashflow_product", "Unknown", "std_product_segment")
        out["std_security_or_stage_band"] = _series_or_default(out, "pd_score_band", "Unknown", "std_security_or_stage_band")
        out["std_cashflow_family"] = _series_or_default(out, "cashflow_product", "Unknown", "std_cashflow_family")

    if "industry_risk_score" in out.columns:
        out["std_industry_risk_band"] = _industry_risk_band(out["industry_risk_score"]).fillna("Unknown")
    else:
        out["std_industry_risk_band"] = "Unknown"

    return out


def build_segmentation_consistency_report(results: dict) -> pd.DataFrame:
    rows = []
    required_cols = [
        "std_module",
        "std_product_segment",
        "std_security_or_stage_band",
        "std_industry_risk_band",
    ]

    for product, payload in results.items():
        if not isinstance(payload, dict) or "loans_with_overlays" not in payload:
            continue
        df = payload["loans_with_overlays"]
        if df is None or len(df) == 0:
            continue

        missing = [c for c in required_cols if c not in df.columns]
        rows.append(
            {
                "product": product,
                "loan_count": int(len(df)),
                "segment_cols_present": len(missing) == 0,
                "missing_segment_cols": ",".join(missing) if missing else "none",
                "unique_std_product_segment": int(df.get("std_product_segment", pd.Series(dtype=object)).nunique()),
                "unique_std_security_or_stage_band": int(df.get("std_security_or_stage_band", pd.Series(dtype=object)).nunique()),
                "unique_std_industry_risk_band": int(df.get("std_industry_risk_band", pd.Series(dtype=object)).nunique()),
            }
        )

    return pd.DataFrame(rows).sort_values("product").reset_index(drop=True) if rows else pd.DataFrame(
        columns=[
            "product",
            "loan_count",
            "segment_cols_present",
            "missing_segment_cols",
            "unique_std_product_segment",
            "unique_std_security_or_stage_band",
            "unique_std_industry_risk_band",
        ]
    )
