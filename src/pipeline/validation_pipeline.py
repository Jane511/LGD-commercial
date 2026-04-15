from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import pandas as pd

sys.dont_write_bytecode = True


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = PROJECT_ROOT / "outputs" / "portfolio"
REPORT_DIR = PROJECT_ROOT / "outputs" / "portfolio" / "reports"
RUNTIME_SOURCE = "generated"
RUNTIME_CONTROLLED_ROOT = PROJECT_ROOT / "data" / "controlled"
RUNTIME_REQUIRE_ALL_PRODUCTS = True


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


def _load_runtime_datasets():
    from src.data.data_source_adapter import load_datasets

    return load_datasets(
        source=RUNTIME_SOURCE,
        controlled_root=RUNTIME_CONTROLLED_ROOT,
        require_all_products=RUNTIME_REQUIRE_ALL_PRODUCTS,
    )


def _step_demo_pipeline():
    from src.lgd_calculation import run_full_pipeline

    datasets = _load_runtime_datasets()
    results = run_full_pipeline(datasets, include_reporting=False)
    required_products = {"mortgage", "commercial", "development", "cashflow_lending"}
    missing_products = sorted(required_products - set(results.keys()))

    has_core_payload = True
    for product in required_products:
        payload = results.get(product, {})
        if product == "cashflow_lending":
            ok = ("loans_with_overlays" in payload) and ("segment_summary" in payload)
        else:
            ok = (
                ("loans_with_overlays" in payload)
                and ("segment_summary" in payload)
                and ("weighted_output" in payload)
            )
        has_core_payload = has_core_payload and ok

    passed = (len(missing_products) == 0) and has_core_payload
    detail = (
        f"source={RUNTIME_SOURCE}; products={sorted(results.keys())}; "
        f"missing_products={missing_products or 'none'}; core_payload={has_core_payload}"
    )
    return {"passed": passed, "detail": detail}


def _step_core_governance_reports():
    from src.lgd_calculation import run_full_pipeline

    datasets = _load_runtime_datasets()
    results = run_full_pipeline(datasets, include_reporting=True)
    reporting = results.get("reporting_tables", {})

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    counts = {}
    required_tables = {
        "fallback_usage_report.csv",
        "unemployment_year_bucket_report.csv",
        "proxy_flags_report.csv",
        "cure_overlay_report.csv",
        "overlay_trace_report.csv",
        "parameter_version_report.csv",
        "segmentation_consistency_report.csv",
        "run_metadata_report.csv",
    }
    for name, df in reporting.items():
        df.to_csv(TABLE_DIR / name, index=False)
        counts[name] = len(df)

    missing = sorted(required_tables - set(reporting.keys()))
    empties = sorted([name for name in required_tables if len(reporting.get(name, [])) == 0])
    passed = (len(missing) == 0) and (len(empties) == 0)
    detail = f"rows={json.dumps(counts, sort_keys=True)}; missing={missing or 'none'}; empty={empties or 'none'}"
    return {"passed": passed, "detail": detail}


def _step_reproducibility_determinism():
    from src.lgd_calculation import run_full_pipeline
    from src.reproducibility import set_global_seed

    def _norm(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out = out.sort_index(axis=1)
        for col in out.columns:
            if pd.api.types.is_numeric_dtype(out[col]):
                out[col] = pd.to_numeric(out[col], errors="coerce").round(10)
            else:
                out[col] = out[col].astype(str)
        return out

    set_global_seed(42)
    d1 = _load_runtime_datasets()
    r1 = run_full_pipeline(d1, include_reporting=True, seed=42, scenario_id="baseline")
    set_global_seed(42)
    d2 = _load_runtime_datasets()
    r2 = run_full_pipeline(d2, include_reporting=True, seed=42, scenario_id="baseline")

    checks = []
    for product in ["mortgage", "commercial", "development", "cashflow_lending"]:
        if product in r1 and product in r2:
            if "weighted_output" in r1[product] and "weighted_output" in r2[product]:
                left = _norm(r1[product]["weighted_output"])
                right = _norm(r2[product]["weighted_output"])
                checks.append(
                    {
                        "target": f"{product}.weighted_output",
                        "is_equal": bool(left.equals(right)),
                    }
                )

    for table in ["overlay_trace_report.csv", "parameter_version_report.csv", "segmentation_consistency_report.csv"]:
        left = _norm(r1["reporting_tables"][table])
        right = _norm(r2["reporting_tables"][table])
        checks.append(
            {
                "target": f"reporting.{table}",
                "is_equal": bool(left.equals(right)),
            }
        )

    df = pd.DataFrame(checks)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE_DIR / "reproducibility_determinism_report.csv", index=False)
    passed = bool(df["is_equal"].all()) if len(df) else False
    detail = f"checks={len(df)}; all_equal={passed}"
    return {"passed": passed, "detail": detail}


def _step_validation_report_hooks():
    from src.lgd_calculation import run_full_pipeline
    from src.validation import generate_validation_report

    datasets = _load_runtime_datasets()
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

    raw_dir = PROJECT_ROOT / "data" / "raw"
    if RUNTIME_SOURCE == "controlled":
        raw_dir = Path(RUNTIME_CONTROLLED_ROOT)

    _, summary, checks = build_and_save_repo_final_lgd(
        raw_dir=raw_dir,
        output_dir=TABLE_DIR,
    )
    pass_rate = float(checks["passed"].mean()) if len(checks) else 0.0
    passed = bool(checks["passed"].all())
    detail = f"summary_rows={len(summary)}; check_pass_rate={pass_rate:.2%}"
    return {"passed": passed, "detail": detail}


def _step_aps113_calibration_validation():
    """
    Step 7: APS 113 extended validation suite (Gini, Hosmer-Lemeshow, PSI, OOT).
    """
    from src.validation import run_full_validation_suite
    from src.product_routing import PRODUCT_TO_FAMILY

    PRODUCTS = [
        "mortgage", "commercial_cashflow", "receivables", "trade_contingent",
        "asset_equipment", "development_finance", "cre_investment",
        "residual_stock", "land_subdivision", "bridging", "mezz_second_mortgage",
    ]

    rows = []
    for product in PRODUCTS:
        family = PRODUCT_TO_FAMILY.get(product, "portfolio")
        cal_path = PROJECT_ROOT / "outputs" / family / f"{product}_final_calibrated_lgd.csv"
        if not cal_path.exists():
            rows.append({
                "product": product, "status": "SKIPPED",
                "gini": None, "calibration_ratio": None, "psi": None,
                "hl_pvalue": None, "detail": "No calibrated LGD output found",
            })
            continue

        try:
            loans = pd.read_csv(cal_path)
            if "lgd_final_calibrated" not in loans.columns or "realised_lgd" not in loans.columns:
                rows.append({
                    "product": product, "status": "SKIPPED",
                    "gini": None, "calibration_ratio": None, "psi": None,
                    "hl_pvalue": None, "detail": "Missing required columns",
                })
                continue

            result = run_full_validation_suite(
                loans=loans,
                predicted_col="lgd_final_calibrated",
                actual_col="realised_lgd",
                product=product,
            )
            gini = result.get("gini")
            cal_ratio = result.get("calibration_ratio")
            psi = result.get("psi")
            hl_p = result.get("hl_pvalue")

            pass_gini = gini is None or gini > 0.50
            pass_cal = cal_ratio is None or (0.85 <= cal_ratio <= 1.15)
            pass_psi = psi is None or psi < 0.10
            step_passed = pass_gini and pass_cal and pass_psi

            if "summary_table" in result:
                result["summary_table"].to_csv(
                    TABLE_DIR / f"{product}_aps113_validation_summary.csv", index=False
                )

            rows.append({
                "product": product,
                "status": "PASS" if step_passed else "FAIL",
                "gini": round(gini, 4) if gini is not None else None,
                "calibration_ratio": round(cal_ratio, 4) if cal_ratio is not None else None,
                "psi": round(psi, 4) if psi is not None else None,
                "hl_pvalue": round(hl_p, 4) if hl_p is not None else None,
                "detail": (
                    f"gini_ok={pass_gini}; cal_ratio_ok={pass_cal}; psi_ok={pass_psi}"
                ),
            })
        except Exception as exc:
            rows.append({
                "product": product, "status": "FAIL",
                "gini": None, "calibration_ratio": None, "psi": None,
                "hl_pvalue": None, "detail": f"{type(exc).__name__}: {exc}",
            })

    summary = pd.DataFrame(rows)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLE_DIR / "aps113_extended_validation_summary.csv", index=False)

    products_run = sum(1 for r in rows if r["status"] != "SKIPPED")
    products_pass = sum(1 for r in rows if r["status"] == "PASS")
    products_skip = sum(1 for r in rows if r["status"] == "SKIPPED")
    passed = products_run == 0 or products_pass >= products_run * 0.8
    detail = (
        f"products_run={products_run}; passed={products_pass}; "
        f"skipped={products_skip}; "
        f"pass_rate={products_pass / max(products_run, 1):.0%}"
    )
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
    passed = compliance_rate >= 0.90
    detail = f"notebooks={len(df)}; reproducibility_proxy_rate={compliance_rate:.2%}"
    return {"passed": passed, "detail": detail}


def main():
    from src.reproducibility import set_global_seed

    global RUNTIME_SOURCE
    global RUNTIME_CONTROLLED_ROOT
    global RUNTIME_REQUIRE_ALL_PRODUCTS

    parser = argparse.ArgumentParser(
        description="Run repo-safe validation sequence against generated or controlled input source."
    )
    parser.add_argument("--source", choices=["generated", "controlled"], default="generated")
    parser.add_argument("--controlled-root", default=str(PROJECT_ROOT / "data" / "controlled"))
    parser.add_argument(
        "--allow-missing-products",
        action="store_true",
        help="Allow adapter loading when some product files are intentionally missing.",
    )
    args = parser.parse_args()

    RUNTIME_SOURCE = args.source
    RUNTIME_CONTROLLED_ROOT = Path(args.controlled_root)
    RUNTIME_REQUIRE_ALL_PRODUCTS = not bool(args.allow_missing_products)

    set_global_seed(42)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    steps = [
        ("demo_pipeline", _step_demo_pipeline),
        ("core_governance_reports", _step_core_governance_reports),
        ("validation_report_hooks", _step_validation_report_hooks),
        ("final_lgd_layer", _step_final_lgd_layer),
        ("reproducibility_determinism", _step_reproducibility_determinism),
        ("notebook_reproducibility_scan", _step_notebook_reproducibility_scan),
        ("aps113_calibration_validation", _step_aps113_calibration_validation),
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
    print(f"Source: {RUNTIME_SOURCE}")
    if RUNTIME_SOURCE == "controlled":
        print(f"Controlled root: {RUNTIME_CONTROLLED_ROOT}")
    print(f"CSV: {TABLE_DIR / 'validation_sequence_report.csv'}")
    print(f"MD:  {REPORT_DIR / 'validation_sequence_report.md'}")


if __name__ == "__main__":
    main()
