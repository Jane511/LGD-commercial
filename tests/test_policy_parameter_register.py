from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.demo_run_pipeline import run_pipeline  # noqa: E402


def _non_blank(series: pd.Series) -> pd.Series:
    txt = series.astype(str).str.strip()
    return ~(txt.eq("") | txt.eq("nan") | txt.eq("None"))


def test_policy_parameter_register_structural_validity():
    result = run_pipeline(project_root=ROOT, persist=False)
    assert "policy_parameter_register.csv" in result["outputs"]
    df = result["outputs"]["policy_parameter_register.csv"]

    required_cols = {
        "record_type",
        "group",
        "parameter",
        "value",
        "description",
        "fallback_hierarchy",
        "category",
        "status",
        "calibration_status",
    }
    assert required_cols.issubset(df.columns)
    assert not df.empty
    assert len(df) >= 12  # not a placeholder-only file

    for col in ["record_type", "group", "parameter", "value"]:
        assert _non_blank(df[col]).all(), f"{col} contains blank values"


def test_policy_parameter_register_stage_rows_and_core_fields():
    result = run_pipeline(project_root=ROOT, persist=False)
    df = result["outputs"]["policy_parameter_register.csv"]

    policy_rows = df[df["record_type"] == "policy_parameter"].copy()
    key_rows = {
        "mortgage_stage1_probability_of_cure",
        "mortgage_stage2_liquidation_loss_if_not_cured",
    }
    present = set(policy_rows["parameter"].astype(str))
    assert key_rows.issubset(present)

    stage_df = policy_rows[policy_rows["parameter"].isin(key_rows)]
    assert len(stage_df) == 2
    assert _non_blank(stage_df["category"]).all()
    assert _non_blank(stage_df["status"]).all()
    assert _non_blank(stage_df["calibration_status"]).all()


def test_policy_parameter_register_has_fallback_and_policy_records():
    result = run_pipeline(project_root=ROOT, persist=False)
    df = result["outputs"]["policy_parameter_register.csv"]

    record_types = set(df["record_type"].astype(str))
    assert "policy_parameter" in record_types
    assert "fallback_hierarchy" in record_types
