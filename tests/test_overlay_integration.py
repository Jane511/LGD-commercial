from pathlib import Path
import json
import shutil
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_generation import generate_all_datasets  # noqa: E402
from src.lgd_calculation import run_full_pipeline  # noqa: E402
from src.overlay_parameters import OverlayParameterManager  # noqa: E402
from src.reproducibility import set_global_seed  # noqa: E402


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_index(axis=1)
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], errors="coerce").round(10)
        else:
            out[col] = out[col].astype(str)
    return out


def test_run_full_pipeline_emits_new_reporting_tables():
    datasets = generate_all_datasets()
    results = run_full_pipeline(datasets, include_reporting=True, seed=42, scenario_id="baseline")

    reporting = results.get("reporting_tables", {})
    required = {
        "overlay_trace_report.csv",
        "parameter_version_report.csv",
        "segmentation_consistency_report.csv",
        "run_metadata_report.csv",
    }
    assert required.issubset(reporting.keys())
    for name in required:
        assert not reporting[name].empty

    # Invariant checks still hold.
    for product in ["mortgage", "commercial", "development"]:
        loans = results[product]["loans_with_overlays"]
        assert (loans["lgd_downturn"] >= loans["lgd_base"]).all()
        if product != "mortgage":
            assert (loans["lgd_final"] >= loans["lgd_downturn"].clip(upper=1.0)).all()


def test_same_seed_reproducibility_for_key_outputs():
    set_global_seed(42)
    d1 = generate_all_datasets()
    r1 = run_full_pipeline(d1, include_reporting=True, seed=42, scenario_id="baseline")

    set_global_seed(42)
    d2 = generate_all_datasets()
    r2 = run_full_pipeline(d2, include_reporting=True, seed=42, scenario_id="baseline")

    pd.testing.assert_frame_equal(
        _norm(r1["commercial"]["weighted_output"]),
        _norm(r2["commercial"]["weighted_output"]),
    )
    pd.testing.assert_frame_equal(
        _norm(r1["reporting_tables"]["overlay_trace_report.csv"]),
        _norm(r2["reporting_tables"]["overlay_trace_report.csv"]),
    )


def test_parameter_hash_mismatch_raises(tmp_path: Path):
    csv_src = ROOT / "data" / "config" / "overlay_parameters.csv"
    csv_dst = tmp_path / "overlay_parameters.csv"
    shutil.copyfile(csv_src, csv_dst)

    bad_manifest = tmp_path / "overlay_parameters_manifest.json"
    bad_manifest.write_text(
        json.dumps({"expected_version": "v1.0", "expected_sha256": "deadbeef"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="hash mismatch"):
        OverlayParameterManager(csv_path=csv_dst, manifest_path=bad_manifest)
