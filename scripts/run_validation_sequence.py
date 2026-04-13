from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import pandas as pd

sys.dont_write_bytecode = True


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"


def _format_detail(detail: str) -> str:
    return detail.replace("\n", " ").strip()


def _run_step(step_name: str, fn):
    try:
        payload = fn() or {}
        passed = bool(payload.get("passed", True))
        status = "PASS" if passed else "FAIL"
        detail = _format_detail(str(payload.get("detail", "")))
        return {"step": step_name, "status": status, "detail": detail}
    except Exception as exc:  # pragma: no cover - defensive reporting path
        return {
            "step": step_name,
            "status": "FAIL",
            "detail": _format_detail(f"{type(exc).__name__}: {exc}"),
            "traceback": traceback.format_exc(),
        }


def _step_demo_pipeline():
    from src.demo_run_pipeline import run_pipeline

    result = run_pipeline(project_root=PROJECT_ROOT, persist=True)
    outputs = result["outputs"]
    required = {
        "lgd_segment_summary.csv",
        "recovery_waterfall.csv",
        "downturn_lgd_output.csv",
        "lgd_validation_report.csv",
        "policy_parameter_register.csv",
        "pipeline_validation_report.csv",
    }
    missing = sorted(required - set(outputs.keys()))
    all_valid = bool(result["validation"]["status"].all())
    passed = (len(missing) == 0) and all_valid
    detail = (
        f"outputs={len(outputs)}; validation_pass={all_valid}; "
        f"missing={missing if missing else 'none'}"
    )
    return {"passed": passed, "detail": detail}


def _step_core_governance_reports():
    from src.data_generation import generate_all_datasets
    from src.lgd_calculation import run_full_pipeline

    datasets = generate_all_datasets()
    results = run_full_pipeline(datasets, include_reporting=True)
    reporting = results.get("reporting_tables", {})

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    counts = {}
    required_tables = {
        "fallback_usage_report.csv",
        "unemployment_year_bucket_report.csv",
        "proxy_flags_report.csv",
        "cure_overlay_report.csv",
    }
    for name, df in reporting.items():
        df.to_csv(TABLE_DIR / name, index=False)
        counts[name] = len(df)

    missing = sorted(required_tables - set(reporting.keys()))
    empties = sorted([name for name in required_tables if len(reporting.get(name, [])) == 0])
    passed = (len(missing) == 0) and (len(empties) == 0)
    detail = f"rows={json.dumps(counts, sort_keys=True)}; missing={missing or 'none'}; empty={empties or 'none'}"
    return {"passed": passed, "detail": detail}


def _step_validation_report_hooks():
    from src.data_generation import generate_all_datasets
    from src.lgd_calculation import run_full_pipeline
    from src.validation import generate_validation_report

    datasets = generate_all_datasets()
    results = run_full_pipeline(datasets, include_reporting=False)

    product_configs = {
        "mortgage": {"segment_col": "mortgage_class", "fallback_yob_years": 4.5},
        "commercial": {"segment_col": "security_type", "fallback_yob_years": 3.0},
        "development": {"segment_col": "completion_stage", "fallback_yob_years": 2.0},
    }

    governance_parts = []
    vintage_parts = []
    time_parts = []
    source_parts = []
    oot_rows = []
    ranking_rows = []
    weighted_flags = []

    for product, cfg in product_configs.items():
        payload = results.get(product, {})
        loans = payload.get("loans_with_overlays")
        if loans is None or len(loans) == 0:
            continue
        work = loans.copy()
        work["product"] = product

        report = generate_validation_report(
            work,
            actual_col="realised_lgd",
            predicted_col="lgd_final",
            segment_col=cfg["segment_col"],
            date_col="default_date",
            oot_holdout_start="2023-01-01",
            fallback_years_on_book=cfg["fallback_yob_years"],
            ranking_segment_col=cfg["segment_col"],
        )
        weighted_flags.append("weighted_accuracy" in report)

        governance = report.get("governance_flags", pd.DataFrame())
        if isinstance(governance, pd.DataFrame) and not governance.empty:
            g = governance.copy()
            g.insert(0, "product", product)
            governance_parts.append(g)

        vintage = report.get("vintage_summary", pd.DataFrame())
        if isinstance(vintage, pd.DataFrame) and not vintage.empty:
            v = vintage.copy()
            v.insert(0, "product", product)
            vintage_parts.append(v)

        weighted_time = report.get("weighted_lgd_over_time", pd.DataFrame())
        if isinstance(weighted_time, pd.DataFrame) and not weighted_time.empty:
            t = weighted_time.copy()
            if "product" not in t.columns:
                t.insert(0, "product", product)
            time_parts.append(t)

        source = report.get("origination_year_source_summary", pd.DataFrame())
        if isinstance(source, pd.DataFrame) and not source.empty:
            s = source.copy()
            s.insert(0, "product", product)
            source_parts.append(s)

        oot = report.get("out_of_time")
        if isinstance(oot, dict):
            stability = oot.get("stability_summary", {})
            ranking = oot.get("ranking_consistency", {})
            oot_rows.append(
                {
                    "product": product,
                    "holdout_start": oot.get("holdout_start"),
                    "train_period_start": oot.get("train_period_start"),
                    "train_period_end": oot.get("train_period_end"),
                    "test_period_start": oot.get("test_period_start"),
                    "test_period_end": oot.get("test_period_end"),
                    "train_size": oot.get("train_size"),
                    "test_size": oot.get("test_size"),
                    "train_weighted_actual_lgd": stability.get("train_weighted_actual_lgd"),
                    "test_weighted_actual_lgd": stability.get("test_weighted_actual_lgd"),
                    "weighted_actual_shift_test_minus_train": stability.get("weighted_actual_shift_test_minus_train"),
                    "weighted_actual_shift_pct": stability.get("weighted_actual_shift_pct"),
                }
            )
            ranking_rows.append(
                {
                    "product": product,
                    "segment_col": ranking.get("segment_col"),
                    "common_segments": ranking.get("common_segments"),
                    "spearman_rank_corr": ranking.get("spearman_rank_corr"),
                    "top_segment_train": ranking.get("top_segment_train"),
                    "top_segment_test": ranking.get("top_segment_test"),
                    "top_segment_match": ranking.get("top_segment_match"),
                }
            )

    governance_flags = (
        pd.concat(governance_parts, ignore_index=True)
        if governance_parts
        else pd.DataFrame(columns=["product", "column", "value", "loan_count", "total_ead", "ead_share"])
    )
    vintage_summary = (
        pd.concat(vintage_parts, ignore_index=True)
        if vintage_parts
        else pd.DataFrame(columns=[
            "product", "origination_year", "default_year", "loan_count", "total_ead",
            "ead_weighted_actual_lgd", "ead_weighted_predicted_lgd", "weighted_lgd_gap_pred_minus_actual",
        ])
    )
    weighted_over_time = (
        pd.concat(time_parts, ignore_index=True)
        if time_parts
        else pd.DataFrame(columns=[
            "product", "default_year", "loan_count", "total_ead",
            "ead_weighted_actual_lgd", "ead_weighted_predicted_lgd", "weighted_lgd_gap_pred_minus_actual",
        ])
    )
    source_summary = (
        pd.concat(source_parts, ignore_index=True)
        if source_parts
        else pd.DataFrame(columns=["product", "source", "loan_count", "total_ead"])
    )
    oot_summary = pd.DataFrame(oot_rows)
    ranking_summary = pd.DataFrame(ranking_rows)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    governance_flags.to_csv(TABLE_DIR / "lgd_validation_governance_flags.csv", index=False)
    vintage_summary.to_csv(TABLE_DIR / "lgd_vintage_summary.csv", index=False)
    weighted_over_time.to_csv(TABLE_DIR / "lgd_weighted_over_time.csv", index=False)
    oot_summary.to_csv(TABLE_DIR / "lgd_out_of_time_summary.csv", index=False)
    ranking_summary.to_csv(TABLE_DIR / "lgd_out_of_time_ranking_consistency.csv", index=False)
    source_summary.to_csv(TABLE_DIR / "origination_year_source_summary.csv", index=False)

    # Backward-compatible mortgage-only governance file
    governance_flags.loc[
        governance_flags["product"] == "mortgage"
    ].to_csv(TABLE_DIR / "mortgage_validation_governance_flags.csv", index=False)

    has_weighted = bool(weighted_flags) and bool(all(weighted_flags))
    has_governance = not governance_flags.empty
    has_vintage = not vintage_summary.empty
    has_time_series = not weighted_over_time.empty
    has_oot = not oot_summary.empty
    has_ranking = not ranking_summary.empty
    has_source = not source_summary.empty
    passed = all(
        [
            has_weighted,
            has_governance,
            has_vintage,
            has_time_series,
            has_oot,
            has_ranking,
            has_source,
        ]
    )
    detail = (
        f"weighted_accuracy_all_products={has_weighted}; "
        f"governance_rows={len(governance_flags)}; "
        f"vintage_rows={len(vintage_summary)}; "
        f"time_rows={len(weighted_over_time)}; "
        f"oot_rows={len(oot_summary)}; "
        f"ranking_rows={len(ranking_summary)}; "
        f"origination_source_rows={len(source_summary)}"
    )
    return {"passed": passed, "detail": detail}


def _step_final_lgd_layer():
    from src.lgd_final import build_and_save_repo_final_lgd

    _, summary, checks = build_and_save_repo_final_lgd(
        raw_dir=PROJECT_ROOT / "data" / "raw",
        output_dir=TABLE_DIR,
    )
    pass_rate = float(checks["passed"].mean()) if len(checks) else 0.0
    passed = bool(checks["passed"].all())
    detail = f"summary_rows={len(summary)}; check_pass_rate={pass_rate:.2%}"
    return {"passed": passed, "detail": detail}


def _step_notebook_reproducibility_scan():
    notebooks = sorted((PROJECT_ROOT / "notebooks").glob("*.ipynb"))
    records = []
    for nb_path in notebooks:
        text = nb_path.read_text(encoding="utf-8", errors="ignore")
        has_seed = (
            "np.random.seed(" in text
            or "random_state=" in text
            or "seed=" in text
            or "set_global_seed(" in text
            or "default_rng(" in text
            or "bootstrap_notebook(" in text
        )
        has_repo_import_hook = (
            "sys.path.insert(" in text
            or "from src." in text
            or "import src." in text
        )
        uses_deterministic_generators = (
            "generate_all_datasets(" in text
            or "generate_mortgage_data(" in text
            or "generate_commercial_data(" in text
            or "generate_development_data(" in text
        )
        compliant = has_repo_import_hook and (has_seed or uses_deterministic_generators)
        records.append(
            {
                "notebook": nb_path.name,
                "has_repo_import_hook": has_repo_import_hook,
                "has_seed_hint": has_seed,
                "uses_deterministic_generator": uses_deterministic_generators,
                "is_reproducible_proxy": compliant,
            }
        )

    df = pd.DataFrame(records)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE_DIR / "notebook_reproducibility_scan.csv", index=False)

    compliance_rate = float(df["is_reproducible_proxy"].mean()) if len(df) else 0.0
    passed = compliance_rate >= 0.75
    detail = f"notebooks={len(df)}; reproducibility_proxy_rate={compliance_rate:.2%}"
    return {"passed": passed, "detail": detail}


def main():
    from src.reproducibility import set_global_seed

    set_global_seed(42)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    steps = [
        ("demo_pipeline", _step_demo_pipeline),
        ("core_governance_reports", _step_core_governance_reports),
        ("validation_report_hooks", _step_validation_report_hooks),
        ("final_lgd_layer", _step_final_lgd_layer),
        ("notebook_reproducibility_scan", _step_notebook_reproducibility_scan),
    ]

    results = [_run_step(name, fn) for name, fn in steps]
    report_df = pd.DataFrame(results)
    report_df.to_csv(TABLE_DIR / "validation_sequence_report.csv", index=False)

    passed = int((report_df["status"] == "PASS").sum())
    total = len(report_df)
    lines = [
        "# Validation Sequence Report",
        "",
        f"- Steps passed: {passed}/{total}",
        "",
        "| Step | Status | Detail |",
        "|---|---|---|",
    ]
    for _, row in report_df.iterrows():
        lines.append(f"| {row['step']} | {row['status']} | {row['detail']} |")
    (REPORT_DIR / "validation_sequence_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(f"Validation sequence completed: {passed}/{total} steps passed.")
    print(f"CSV: {TABLE_DIR / 'validation_sequence_report.csv'}")
    print(f"MD:  {REPORT_DIR / 'validation_sequence_report.md'}")


if __name__ == "__main__":
    main()
