"""
Full APS 113 Calibration Pipeline — all 11 product modules.

Runs the complete calibration sequence for each product:
    1. Load or generate synthetic historical workout data
    2. Compute realised LGD (workout method, LIP costs, cure leg)
    3. Classify economic regimes (real RBA/ABS upstream if available)
    4. Segment by product-specific keys
    5. Compute long-run LGD (vintage-EWA method)
    6. Compare model vs actual LGD
    7. Apply downturn overlay
    8. Apply LGD-PD correlation adjustment (Frye-Jacobs)
    9. Compute and apply MoC (AFTER downturn — per APS 113 s.63)
    10. Apply regulatory/policy floors
    11. Export all outputs (9 CSV files per module)
    12. Run full validation suite

Outputs to outputs/tables/. Parquet source data in data/generated/historical/.

Usage:
    python scripts/run_calibration_pipeline.py
    python scripts/run_calibration_pipeline.py --module mortgage
    python scripts/run_calibration_pipeline.py --products mortgage development_finance
    python scripts/run_calibration_pipeline.py --force-regen  (regenerate data)
    python scripts/run_calibration_pipeline.py --skip-validation

APS 113: All methodology steps cite relevant sections in module docstrings.
SYNTHETIC DATA: All workout data is synthetically generated for demonstration only.
"""
import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.calibration_utils import (
    compute_realised_lgd,
    segment_lgd,
    compute_long_run_lgd,
    compare_model_vs_actual,
    run_calibration_pipeline,
    classify_economic_regime,
    assign_regime_to_workouts,
    run_full_validation_suite,
    generate_compliance_map,
    export_compliance_map,
    load_apra_adi_benchmarks,
    generate_benchmark_comparison,
    export_benchmark_comparison,
    build_lgd_pd_annual_series,
    estimate_lgd_pd_correlation,
    export_correlation_report,
    load_rba_lending_rates,
    export_discount_rate_register,
)
from src.generators import GENERATOR_MAP, generate_all_historical_workouts

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_calibration_pipeline")

# Product-specific segment keys
PRODUCT_SEGMENT_KEYS: dict[str, list[str]] = {
    "mortgage":              ["mortgage_class", "lvr_band"],
    "commercial_cashflow":   ["security_type", "facility_type"],
    "receivables":           ["recourse_flag", "collections_control_flag"],
    "trade_contingent":      ["facility_type", "cash_collateral_band"],
    "asset_equipment":       ["asset_class", "secondary_market_liquidity"],
    "development_finance":   ["completion_stage_at_default", "presale_cover_band"],
    "cre_investment":        ["asset_class_cre", "lvr_band"],
    "residual_stock":        ["market_depth_proxy", "stock_age_band"],
    "land_subdivision":      ["zoning_stage", "market_depth_proxy"],
    "bridging":              ["exit_certainty_band", "exit_type"],
    "mezz_second_mortgage":  ["seniority", "attachment_point_band"],
}

HISTORY_DIR = REPO_ROOT / "data" / "generated" / "historical"
OUTPUTS_DIR = REPO_ROOT / "outputs" / "tables"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run APS 113 calibration pipeline for LGD models."
    )
    parser.add_argument(
        "--module", "--product", dest="module", default=None,
        help="Single product to run (e.g., --module mortgage).",
    )
    parser.add_argument(
        "--products", nargs="+", default=None,
        help="List of products to run.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--force-regen", action="store_true",
        help="Force regeneration of workout data even if Parquet files exist.",
    )
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="Skip validation suite (faster for development).",
    )
    parser.add_argument(
        "--upstream-path", default=None,
        help="Path to macro_regime_flags.parquet from industry-analysis repo.",
    )
    return parser.parse_args()


def _get_products(args) -> list[str]:
    if args.module:
        return [args.module]
    if args.products:
        return args.products
    return list(GENERATOR_MAP.keys())


def _load_or_generate(product: str, seed: int, force_regen: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Parquet workout data, generating if not present or force_regen."""
    loans_path = HISTORY_DIR / f"{product}_workouts.parquet"
    cashflows_path = HISTORY_DIR / f"{product}_cashflows.parquet"

    if not force_regen and loans_path.exists() and cashflows_path.exists():
        loans = pd.read_parquet(loans_path)
        cashflows = pd.read_parquet(cashflows_path)
        logger.info("Loaded %s: %d loans from %s", product, len(loans), loans_path.name)
        return loans, cashflows

    logger.info("Generating %s workout data (seed=%d)...", product, seed)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    GeneratorClass = GENERATOR_MAP[product]
    gen = GeneratorClass(seed=seed)
    loans, cashflows = gen.generate()
    loans.to_parquet(loans_path, index=False)
    cashflows.to_parquet(cashflows_path, index=False)
    return loans, cashflows


def _add_lvr_band(df: pd.DataFrame) -> pd.DataFrame:
    """Add lvr_band column for mortgage segmentation."""
    if "lvr_at_default" in df.columns and "lvr_band" not in df.columns:
        lvr = pd.to_numeric(df["lvr_at_default"], errors="coerce")
        df["lvr_band"] = pd.cut(
            lvr, bins=[0, 0.60, 0.70, 0.80, 0.90, 1.5],
            labels=["<=60%", "60-70%", "70-80%", "80-90%", ">90%"],
            right=True,
        ).astype(str)
    return df


def _add_standard_bands(df: pd.DataFrame, product: str) -> pd.DataFrame:
    """Add segmentation band columns that may not be in base generator."""
    df = _add_lvr_band(df)

    if product == "development_finance" and "presale_cover_pct" in df.columns:
        if "presale_cover_band" not in df.columns:
            pc = pd.to_numeric(df["presale_cover_pct"], errors="coerce")
            df["presale_cover_band"] = pd.cut(
                pc, bins=[-0.01, 0.50, 0.80, 10],
                labels=["<50%", "50-80%", ">80%"],
            ).astype(str)

    if product == "trade_contingent" and "cash_collateral_pct" in df.columns:
        if "cash_collateral_band" not in df.columns:
            cc = pd.to_numeric(df["cash_collateral_pct"], errors="coerce")
            df["cash_collateral_band"] = pd.cut(
                cc, bins=[-0.01, 0.25, 0.60, 1.01],
                labels=["<25%", "25-60%", ">60%"],
            ).astype(str)

    if product == "residual_stock" and "stock_age_months" in df.columns:
        if "stock_age_band" not in df.columns:
            sa = pd.to_numeric(df["stock_age_months"], errors="coerce")
            df["stock_age_band"] = pd.cut(
                sa, bins=[-1, 6, 12, 200],
                labels=["<6m", "6-12m", ">12m"],
            ).astype(str)

    if product == "mezz_second_mortgage" and "mezz_attachment_point_pct" in df.columns:
        if "attachment_point_band" not in df.columns:
            ap = pd.to_numeric(df["mezz_attachment_point_pct"], errors="coerce")
            df["attachment_point_band"] = pd.cut(
                ap, bins=[-0.01, 0.60, 0.75, 0.85, 1.01],
                labels=["<60%", "60-75%", "75-85%", ">85%"],
            ).astype(str)

    return df


def run_product_calibration(
    product: str,
    loans: pd.DataFrame,
    cashflows: pd.DataFrame,
    regimes: pd.DataFrame,
    rates_df: pd.DataFrame,
    skip_validation: bool = False,
) -> dict:
    """Run the 11-step calibration pipeline for a single product."""
    t0 = time.time()
    logger.info("--- Starting calibration: %s ---", product)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Validate observation period
    from src.aps113_compliance import validate_observation_periods
    obs_check = validate_observation_periods(loans, product)
    if not obs_check["compliant"]:
        logger.error("Observation period non-compliant for %s: %s", product, obs_check)

    # Add segmentation bands
    loans = _add_standard_bands(loans, product)

    # Step 2: Compute realised LGD
    loans = compute_realised_lgd(loans, cashflows)

    # Step 3: Assign regime
    loans = assign_regime_to_workouts(loans, regimes)

    # Export discount rate register
    regime_source = regimes["data_source"].iloc[0] if len(regimes) > 0 else "synthetic"
    dr_path = OUTPUTS_DIR / "rba_discount_rate_register.csv"
    if not dr_path.exists():
        export_discount_rate_register(
            loans, rates_df, product, dr_path
        )

    # Step 4-10: Run calibration pipeline (segments, LR-LGD, downturn, MoC, floor)
    segment_keys = PRODUCT_SEGMENT_KEYS.get(product, ["macro_regime"])
    # Filter to valid segment keys
    segment_keys = [k for k in segment_keys if k in loans.columns]
    if not segment_keys:
        segment_keys = ["macro_regime"]

    # Model LGD column: use existing proxy engine output if available
    model_col = "lgd_final" if "lgd_final" in loans.columns else "realised_lgd"

    cal_results = run_calibration_pipeline(
        loans=loans,
        product=product,
        segment_keys=segment_keys,
        regime_data_source=regime_source,
    )

    # Step 6: Model vs actual
    backtest_df = compare_model_vs_actual(
        loans, model_lgd_col=model_col, segment_keys=segment_keys
    )

    # Step 11: Validation suite
    val_results = {}
    if not skip_validation and "realised_lgd" in loans.columns and model_col in loans.columns:
        val_results = run_full_validation_suite(
            df=loans,
            actual_col="realised_lgd",
            predicted_col=model_col,
            segment_col=None,
        )

    # ---- Export all 9 per-module CSV files ----
    prefix = product
    exports = {}

    # 1. Historical workouts
    hw_path = OUTPUTS_DIR / f"{prefix}_historical_workouts.csv"
    loans.to_csv(hw_path, index=False)
    exports["historical_workouts"] = hw_path

    # 2. Long-run LGD by segment
    lr_df = cal_results["long_run_lgd_by_segment"]
    lr_path = OUTPUTS_DIR / f"{prefix}_long_run_lgd_by_segment.csv"
    lr_df.to_csv(lr_path, index=False)
    exports["long_run_lgd_by_segment"] = lr_path

    # 3. Model vs actual
    mv_path = OUTPUTS_DIR / f"{prefix}_model_vs_actual_comparison.csv"
    backtest_df.to_csv(mv_path, index=False)
    exports["model_vs_actual_comparison"] = mv_path

    # 4. Calibration adjustments
    cal_steps = cal_results["calibration_steps"]
    ca_path = OUTPUTS_DIR / f"{prefix}_calibration_adjustments.csv"
    cal_steps.to_csv(ca_path, index=False)
    exports["calibration_adjustments"] = ca_path

    # 5. MoC register
    moc_df = cal_results["moc_register"]
    moc_path = OUTPUTS_DIR / f"{prefix}_moc_register.csv"
    moc_df.to_csv(moc_path, index=False)
    exports["moc_register"] = moc_path

    # 6. Downturn LGD
    dt_path = OUTPUTS_DIR / f"{prefix}_downturn_lgd_by_segment.csv"
    lr_df[["segment_key_concat", "long_run_lgd", "downturn_lgd"]].to_csv(dt_path, index=False)
    exports["downturn_lgd_by_segment"] = dt_path

    # 7. Final calibrated LGD
    fc_path = OUTPUTS_DIR / f"{prefix}_final_calibrated_lgd.csv"
    lr_df[["segment_key_concat", "long_run_lgd", "downturn_lgd",
           "total_moc", "lgd_with_moc", "policy_floor", "final_lgd",
           "aps113_pipeline"]].to_csv(fc_path, index=False)
    exports["final_calibrated_lgd"] = fc_path

    # 8. Backtest results
    bt_path = OUTPUTS_DIR / f"{prefix}_backtest_results.csv"
    if val_results and "summary_table" in val_results:
        val_results["summary_table"].to_csv(bt_path, index=False)
    else:
        pd.DataFrame([{"note": "Validation skipped or insufficient data"}]).to_csv(bt_path, index=False)
    exports["backtest_results"] = bt_path

    # 9. Validation report
    vr_path = OUTPUTS_DIR / f"{prefix}_validation_report.csv"
    if val_results and "validation_report" in val_results:
        val_results["validation_report"].to_csv(vr_path, index=False)
    else:
        pd.DataFrame([{"note": "Validation skipped"}]).to_csv(vr_path, index=False)
    exports["validation_report"] = vr_path

    elapsed = time.time() - t0
    logger.info("Completed %s in %.1fs | %d output files", product, elapsed, len(exports))

    return {
        "loans": loans,
        "calibration": cal_results,
        "backtest": backtest_df,
        "validation": val_results,
        "exports": exports,
        "regime_source": regime_source,
    }


def main() -> None:
    args = parse_args()
    products = _get_products(args)

    logger.info("=" * 70)
    logger.info("LGD Calibration Pipeline — APS 113 Compliant")
    logger.info("SYNTHETIC HISTORICAL CALIBRATION DATA — FOR DEMONSTRATION ONLY")
    logger.info("=" * 70)
    logger.info("Products: %s | Seed: %d", products, args.seed)

    # Validate product names
    unknown = [p for p in products if p not in GENERATOR_MAP]
    if unknown:
        logger.error("Unknown product(s): %s", unknown)
        sys.exit(1)

    # Load RBA rates once
    rates_df = load_rba_lending_rates()

    # Classify economic regimes (use upstream if available)
    regimes = classify_economic_regime(
        upstream_parquet_path=args.upstream_path,
        method="upstream_first",
    )
    logger.info(
        "Economic regimes: %d years | data_source=%s | downturn_years=%d",
        len(regimes),
        regimes["data_source"].iloc[0] if len(regimes) > 0 else "unknown",
        regimes["is_downturn_period"].sum(),
    )

    # Run pipeline per product
    all_results = {}
    all_final_lgd = []
    all_moc_registers = {}

    for product in products:
        try:
            loans, cashflows = _load_or_generate(product, args.seed, args.force_regen)
            result = run_product_calibration(
                product=product,
                loans=loans,
                cashflows=cashflows,
                regimes=regimes,
                rates_df=rates_df,
                skip_validation=args.skip_validation,
            )
            all_results[product] = result

            # Collect final LGD for consolidated output
            fc_df = result["calibration"]["long_run_lgd_by_segment"][
                ["segment_key_concat", "final_lgd"]
            ].copy()
            fc_df["product"] = product
            all_final_lgd.append(fc_df)

            moc_reg = result["calibration"].get("moc_register", pd.DataFrame())
            if not moc_reg.empty:
                all_moc_registers[product] = moc_reg

        except Exception as exc:
            logger.error("Pipeline failed for %s: %s", product, exc, exc_info=True)
            sys.exit(1)

    # Consolidated outputs
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    if all_final_lgd:
        consolidated = pd.concat(all_final_lgd, ignore_index=True)
        consolidated.to_csv(OUTPUTS_DIR / "lgd_final_calibrated.csv", index=False)
        logger.info("Written: lgd_final_calibrated.csv (%d rows)", len(consolidated))

    if all_moc_registers:
        moc_all = pd.concat(all_moc_registers.values(), ignore_index=True)
        moc_all.to_csv(OUTPUTS_DIR / "moc_summary_all_products.csv", index=False)
        logger.info("Written: moc_summary_all_products.csv (%d rows)", len(moc_all))

    # APRA benchmark comparison
    try:
        apra_benchmarks = load_apra_adi_benchmarks()
        all_cal = pd.concat(
            [r["calibration"]["long_run_lgd_by_segment"].assign(product=p)
             for p, r in all_results.items()],
            ignore_index=True,
        )
        if "final_lgd" in all_cal.columns:
            bench_df = generate_benchmark_comparison(
                all_cal, apra_benchmarks, product_col="product", lgd_col="final_lgd"
            )
            export_benchmark_comparison(bench_df, OUTPUTS_DIR / "apra_benchmark_comparison.csv")
    except Exception as e:
        logger.warning("APRA benchmark comparison skipped: %s", e)

    # APS 113 compliance map
    regime_source = regimes["data_source"].iloc[0] if len(regimes) > 0 else "synthetic"
    compliance_df = generate_compliance_map(
        calibration_results={p: r["calibration"] for p, r in all_results.items()},
        moc_registers=all_moc_registers,
        regime_data_source=regime_source,
        products=products,
    )
    export_compliance_map(compliance_df, OUTPUTS_DIR / "aps113_compliance_map.csv")

    # Calibration summary dashboard
    summary_rows = []
    for product, result in all_results.items():
        lr_df = result["calibration"].get("long_run_lgd_by_segment", pd.DataFrame())
        val = result.get("validation", {})
        summary_rows.append({
            "product": product,
            "n_segments": len(lr_df),
            "mean_long_run_lgd": round(float(lr_df["long_run_lgd"].mean()), 4) if "long_run_lgd" in lr_df.columns else None,
            "mean_downturn_lgd": round(float(lr_df["downturn_lgd"].mean()), 4) if "downturn_lgd" in lr_df.columns else None,
            "mean_final_lgd": round(float(lr_df["final_lgd"].mean()), 4) if "final_lgd" in lr_df.columns else None,
            "regime_data_source": result.get("regime_source", "unknown"),
            "gini": val.get("gini", {}).get("gini") if val else None,
            "calibration_ratio": val.get("calibration_ratio", {}).get("calibration_ratio") if val else None,
            "status": "final_calibrated_output",
            "calibration_date": datetime.now().strftime("%Y-%m-%d"),
        })

    dashboard = pd.DataFrame(summary_rows)
    dashboard.to_csv(OUTPUTS_DIR / "calibration_summary_dashboard.csv", index=False)
    logger.info("Written: calibration_summary_dashboard.csv")

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("Calibration complete. Key outputs:")
    logger.info("  outputs/tables/lgd_final_calibrated.csv")
    logger.info("  outputs/tables/aps113_compliance_map.csv")
    logger.info("  outputs/tables/calibration_summary_dashboard.csv")
    logger.info("  outputs/tables/apra_benchmark_comparison.csv")
    n_csv = sum(
        len(r["exports"]) for r in all_results.values()
    )
    logger.info("  + %d per-module CSV files (9 per product)", n_csv)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
