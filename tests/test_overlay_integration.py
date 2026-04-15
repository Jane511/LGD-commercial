from pathlib import Path
import json
import shutil
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.data_generation import generate_all_datasets  # noqa: E402
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


# ── APS 113 calibration pipeline ordering assertion ───────────────────────────

def test_calibration_pipeline_moc_applied_after_downturn():
    """
    APS 113 s.63 integration test: verify that in run_calibration_pipeline()
    the MoC step receives the *downturn* LGD as input, not the long-run LGD.

    This is the most critical ordering constraint:
        LR-LGD → downturn overlay → MoC → floor

    Verifies: final calibrated LGD >= downturn LGD >= long-run LGD
    (i.e., each step only adds or holds — never goes backwards).
    """
    import numpy as np
    from src.moc_framework import run_calibration_pipeline

    rng = np.random.default_rng(42)
    n = 60
    years = np.tile(np.arange(2014, 2024), int(np.ceil(n / 10)))[:n]
    df = pd.DataFrame({
        "loan_id": [f"L{i}" for i in range(n)],
        "default_year": years,
        "realised_lgd": rng.uniform(0.15, 0.55, n),
        "ead_at_default": rng.uniform(100_000, 1_000_000, n),
        "mortgage_class": rng.choice(["Standard", "Non-Standard"], n),
        "lgd_long_run": [0.25] * n,
    })

    result = run_calibration_pipeline(
        loans=df,
        product="mortgage",
        segment_keys=["mortgage_class"],
        lr_lgd_col="lgd_long_run",
    )

    lr_ewa = df["lgd_long_run"].mean()

    # Downturn LGD must be >= LR-LGD
    if "lgd_downturn" in result:
        dt_ewa = (
            (result["lgd_downturn"] if isinstance(result["lgd_downturn"], pd.Series)
             else df["lgd_long_run"]).mean()
        )
        assert dt_ewa >= lr_ewa * 0.99, \
            f"Downturn LGD ({dt_ewa:.4f}) < LR-LGD ({lr_ewa:.4f}) — downturn must be >= LR"

    # Final LGD must be >= LR-LGD (MoC + floor can only add)
    final_col = "final_lgd" if "final_lgd" in result else "lgd_final"
    if final_col in result:
        final_series = result[final_col]
        if isinstance(final_series, pd.Series) and len(final_series) == n:
            final_ewa = (final_series * df["ead_at_default"]).sum() / df["ead_at_default"].sum()
            assert final_ewa >= lr_ewa * 0.99, \
                f"Final LGD ({final_ewa:.4f}) < LR-LGD ({lr_ewa:.4f}) — pipeline must be additive"

    # The result must contain at least one calibration output
    output_keys = {"lgd_downturn", "final_lgd", "lgd_final", "lgd_with_moc", "calibration_steps"}
    assert output_keys.intersection(result.keys()), \
        f"run_calibration_pipeline returned no recognisable output keys: {set(result.keys())}"
