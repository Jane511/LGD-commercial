"""
Extended LGD Validation Suite.

Adds IRB-grade validation metrics on top of the existing src/validation.py
module. Does NOT re-implement functions already present in validation.py —
it imports and wraps them.

New functions added here:
    - compute_gini_coefficient()    — rank-ordering power for LGD
    - hosmer_lemeshow_test()        — calibration goodness-of-fit
    - run_full_validation_suite()   — unified entry point for notebooks

Existing functions reused from src/validation.py:
    - weighted_accuracy_metrics()   — MAE, RMSE, bias
    - compute_psi()                 — Population Stability Index
    - out_of_time_backtest()        — OOT holdout performance
    - conservatism_test()           — model >= actual conservatism check
    - discriminatory_power()        — Spearman rank correlation

APS 113 References:
    s.66-68: Validation requirements for IRB models
    s.66: Model must be validated against independent datasets
    s.67: Backtesting (OOT) required
    s.68: Rank-ordering (discriminatory power) required
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

# Import existing validation functions — do not re-implement
from src.validation import (
    weighted_accuracy_metrics,
    compute_psi,
    out_of_time_backtest,
    conservatism_test,
    discriminatory_power,
    calibration_by_segment,
)

logger = logging.getLogger(__name__)

# Acceptable ranges per Section E of the implementation spec
GINI_THRESHOLD = 0.50           # Gini must exceed 0.50
CALIBRATION_RATIO_LOWER = 0.85  # mean model / mean actual in [0.85, 1.15]
CALIBRATION_RATIO_UPPER = 1.15
PSI_STABLE = 0.10               # PSI < 0.10 = stable
PSI_MONITOR = 0.25              # PSI in [0.10, 0.25] = monitor
HL_P_VALUE_THRESHOLD = 0.05     # HL p-value > 0.05 = adequate calibration


# ---------------------------------------------------------------------------
# Gini / AUROC
# ---------------------------------------------------------------------------

def compute_gini_coefficient(
    actual_lgd: pd.Series,
    predicted_lgd: pd.Series,
    n_bins: int = 10,
) -> dict:
    """
    Compute Gini coefficient and AUROC for LGD rank-ordering power.

    For continuous LGD, AUROC is computed by binarising actual_lgd:
        high_loss = 1 if actual_lgd > median(actual_lgd) else 0
    The Gini coefficient = 2 * AUROC - 1.

    Parameters
    ----------
    actual_lgd : Series of observed LGD values
    predicted_lgd : Series of model LGD values
    n_bins : number of decile bins for Lorenz curve

    Returns
    -------
    dict with:
        gini (float) : Gini coefficient [-1, 1]; higher is better
        auroc (float) : Area Under ROC curve [0.5, 1]; higher is better
        spearman_rho (float) : Spearman rank correlation (from validation.py)
        rank_ordering_adequate (bool) : Gini > GINI_THRESHOLD
        lorenz_curve (pd.DataFrame) : for plotting
        aps113_section (str)

    APS 113 s.68: Models must demonstrate discriminatory power.
    """
    a = pd.to_numeric(actual_lgd, errors="coerce").dropna()
    p = pd.to_numeric(predicted_lgd, errors="coerce")
    p = p.loc[a.index].dropna()
    a = a.loc[p.index]

    if len(a) < 10:
        return {
            "gini": np.nan,
            "auroc": np.nan,
            "spearman_rho": np.nan,
            "rank_ordering_adequate": False,
            "lorenz_curve": pd.DataFrame(),
            "aps113_section": "s.68",
            "note": "Insufficient data for Gini computation.",
        }

    # AUROC via binarisation at median
    median_lgd = float(a.median())
    binary_actual = (a > median_lgd).astype(int)

    # Compute AUROC manually (trapezoidal rule)
    sorted_idx = p.argsort()[::-1]
    y_sorted = binary_actual.iloc[sorted_idx].values
    n_pos = y_sorted.sum()
    n_neg = len(y_sorted) - n_pos

    if n_pos == 0 or n_neg == 0:
        auroc = 0.5
    else:
        tp = np.cumsum(y_sorted) / n_pos
        fp = np.cumsum(1 - y_sorted) / n_neg
        auroc = float(np.trapz(tp, fp))

    gini = 2 * auroc - 1

    # Spearman rho from existing function
    disc = discriminatory_power(a, p)
    spearman_rho = disc.get("Spearman_Rho", np.nan)

    # Lorenz curve
    n = len(a)
    bin_size = max(1, n // n_bins)
    sorted_actual = a.sort_values().reset_index(drop=True)
    lorenz_rows = []
    for i in range(0, n + 1, bin_size):
        pct_pop = i / n
        pct_losses = sorted_actual.iloc[:i].sum() / sorted_actual.sum() if sorted_actual.sum() > 0 else 0
        lorenz_rows.append({"pct_population": round(pct_pop, 4), "pct_losses": round(pct_losses, 4)})
    lorenz_df = pd.DataFrame(lorenz_rows)

    return {
        "gini": round(float(gini), 6),
        "auroc": round(float(auroc), 6),
        "spearman_rho": round(float(spearman_rho), 6) if np.isfinite(spearman_rho) else np.nan,
        "rank_ordering_adequate": gini > GINI_THRESHOLD,
        "lorenz_curve": lorenz_df,
        "aps113_section": "s.68",
    }


# ---------------------------------------------------------------------------
# Hosmer-Lemeshow test
# ---------------------------------------------------------------------------

def hosmer_lemeshow_test(
    actual: pd.Series,
    predicted: pd.Series,
    n_groups: int = 10,
) -> dict:
    """
    Hosmer-Lemeshow goodness-of-fit test adapted for continuous LGD.

    For LGD (continuous), the test bins by predicted LGD decile and
    compares mean actual vs mean predicted within each bin.

    A high p-value (> 0.05) indicates adequate calibration — the model's
    predictions are consistent with actual outcomes.

    Parameters
    ----------
    actual : Series of observed LGD values
    predicted : Series of model LGD values
    n_groups : number of decile groups (default: 10)

    Returns
    -------
    dict with:
        hl_statistic (float) : chi-squared statistic
        p_value (float) : p-value; > 0.05 = adequate calibration
        degrees_of_freedom (int) : n_groups - 2
        calibration_adequate (bool) : p_value > HL_P_VALUE_THRESHOLD
        group_detail (pd.DataFrame) : per-decile comparison
        aps113_section (str)

    APS 113 s.66-67: Backtesting and calibration validation.
    """
    a = pd.to_numeric(actual, errors="coerce")
    p = pd.to_numeric(predicted, errors="coerce")
    mask = np.isfinite(a) & np.isfinite(p)
    a, p = a[mask], p[mask]

    if len(a) < n_groups * 2:
        return {
            "hl_statistic": np.nan,
            "p_value": np.nan,
            "degrees_of_freedom": n_groups - 2,
            "calibration_adequate": False,
            "group_detail": pd.DataFrame(),
            "aps113_section": "s.66-67",
            "note": f"Insufficient data ({len(a)} obs) for {n_groups}-group HL test.",
        }

    # Bin by predicted LGD decile
    bin_labels = pd.qcut(p, q=n_groups, labels=False, duplicates="drop")
    n_actual_groups = bin_labels.nunique()

    group_rows = []
    hl_stat = 0.0
    for g in range(n_actual_groups):
        mask_g = bin_labels == g
        n_g = mask_g.sum()
        actual_mean = float(a[mask_g].mean())
        predicted_mean = float(p[mask_g].mean())
        # Chi-squared contribution using MSE-based approximation
        if n_g > 0 and predicted_mean > 0:
            chi_contrib = n_g * ((actual_mean - predicted_mean) ** 2) / (predicted_mean * (1 - predicted_mean + 1e-6))
            hl_stat += chi_contrib
        group_rows.append({
            "decile_group": g + 1,
            "n_obs": n_g,
            "actual_mean_lgd": round(actual_mean, 6),
            "predicted_mean_lgd": round(predicted_mean, 6),
            "bias": round(actual_mean - predicted_mean, 6),
        })

    dof = n_actual_groups - 2
    p_value = float(1 - stats.chi2.cdf(hl_stat, dof)) if dof > 0 else np.nan

    return {
        "hl_statistic": round(float(hl_stat), 4),
        "p_value": round(float(p_value), 6) if np.isfinite(p_value) else np.nan,
        "degrees_of_freedom": dof,
        "calibration_adequate": p_value > HL_P_VALUE_THRESHOLD if np.isfinite(p_value) else False,
        "group_detail": pd.DataFrame(group_rows),
        "aps113_section": "s.66-67",
    }


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def run_full_validation_suite(
    df: pd.DataFrame,
    actual_col: str,
    predicted_col: str,
    segment_col: str | None = None,
    date_col: str = "default_year",
    ead_col: str = "ead_at_default",
    holdout_start: str | None = None,
) -> dict[str, pd.DataFrame | dict]:
    """
    Run all IRB validation metrics for a single product module.

    Called by every product notebook (steps 11 in the 11-step calibration
    pattern). Never re-implemented locally — always imported from here.

    Parameters
    ----------
    df : loan-level DataFrame with actual, predicted, segment, date, ead columns
    actual_col : column with realised LGD
    predicted_col : column with model (proxy) LGD
    segment_col : column for segmentation (optional)
    date_col : column with default year (for OOT split)
    ead_col : column with EAD (for exposure-weighted metrics)
    holdout_start : year to use as OOT holdout start (e.g., '2023')
        If None, uses last 2 vintage years as holdout.

    Returns
    -------
    dict with keys:
        'accuracy'          : weighted MAE, RMSE, bias
        'gini'              : Gini coefficient, AUROC, rank_ordering_adequate
        'hosmer_lemeshow'   : HL statistic, p-value, calibration_adequate
        'psi'               : Population Stability Index, stability_flag
        'out_of_time'       : OOT calibration ratio
        'conservatism'      : conservatism_test results
        'summary_table'     : pd.DataFrame, one row, all key metrics
        'validation_report' : pd.DataFrame, all metrics for CSV export
    """
    actual = pd.to_numeric(df[actual_col], errors="coerce")
    predicted = pd.to_numeric(df[predicted_col], errors="coerce")
    exposure = pd.to_numeric(df[ead_col], errors="coerce") if ead_col in df.columns else None

    results = {}

    # 1. Accuracy (from existing validation.py)
    results["accuracy"] = weighted_accuracy_metrics(actual, predicted, exposure)

    # 2. Calibration ratio
    mean_actual = float(actual.mean()) if not actual.empty else np.nan
    mean_predicted = float(predicted.mean()) if not predicted.empty else np.nan
    cal_ratio = mean_predicted / mean_actual if mean_actual > 0 else np.nan
    results["calibration_ratio"] = {
        "calibration_ratio": round(cal_ratio, 6) if np.isfinite(cal_ratio) else np.nan,
        "calibration_adequate": (
            CALIBRATION_RATIO_LOWER <= cal_ratio <= CALIBRATION_RATIO_UPPER
        ) if np.isfinite(cal_ratio) else False,
    }

    # 3. Gini / AUROC (new)
    results["gini"] = compute_gini_coefficient(actual, predicted)

    # 4. Hosmer-Lemeshow (new)
    results["hosmer_lemeshow"] = hosmer_lemeshow_test(actual, predicted)

    # 5. PSI (from existing validation.py)
    psi_val = np.nan
    psi_flag = "unknown"
    try:
        psi_val = float(compute_psi(actual, predicted))
        psi_flag = "stable" if psi_val < PSI_STABLE else "monitor" if psi_val < PSI_MONITOR else "unstable"
    except Exception:
        pass
    results["psi"] = {"psi": round(psi_val, 6) if np.isfinite(psi_val) else np.nan, "psi_flag": psi_flag}

    # 6. Out-of-time backtest (from existing validation.py, extended with calendar cutoff)
    results["out_of_time"] = {}
    try:
        if holdout_start and date_col in df.columns:
            train_df = df[pd.to_numeric(df[date_col], errors="coerce") < int(holdout_start)]
            test_df = df[pd.to_numeric(df[date_col], errors="coerce") >= int(holdout_start)]
        elif date_col in df.columns:
            # Default: last 2 years as holdout
            years_sorted = sorted(df[date_col].dropna().unique())
            holdout_years = years_sorted[-2:] if len(years_sorted) >= 2 else years_sorted
            train_df = df[~df[date_col].isin(holdout_years)]
            test_df = df[df[date_col].isin(holdout_years)]
        else:
            train_df = df.iloc[:int(len(df) * 0.8)]
            test_df = df.iloc[int(len(df) * 0.8):]

        if len(test_df) >= 10:
            oot_result = out_of_time_backtest(
                train_df, test_df,
                lgd_col=actual_col,
            )
            results["out_of_time"] = oot_result
    except Exception as e:
        logger.debug("OOT backtest skipped: %s", e)

    # 7. Conservatism (from existing validation.py)
    try:
        results["conservatism"] = conservatism_test(actual, predicted, exposure)
    except Exception:
        results["conservatism"] = {}

    # Build summary table
    summary = {
        "gini": results["gini"].get("gini"),
        "auroc": results["gini"].get("auroc"),
        "rank_ordering_adequate": results["gini"].get("rank_ordering_adequate"),
        "spearman_rho": results["gini"].get("spearman_rho"),
        "hl_statistic": results["hosmer_lemeshow"].get("hl_statistic"),
        "hl_p_value": results["hosmer_lemeshow"].get("p_value"),
        "calibration_adequate": results["hosmer_lemeshow"].get("calibration_adequate"),
        "calibration_ratio": results["calibration_ratio"].get("calibration_ratio"),
        "calibration_ratio_adequate": results["calibration_ratio"].get("calibration_adequate"),
        "psi": results["psi"].get("psi"),
        "psi_flag": results["psi"].get("psi_flag"),
        "weighted_mae": results["accuracy"].get("Weighted_MAE"),
        "weighted_rmse": results["accuracy"].get("Weighted_RMSE"),
        "weighted_bias": results["accuracy"].get("Weighted_Bias"),
        "aps113_section": "s.66-68",
    }
    results["summary_table"] = pd.DataFrame([summary])

    # Validation report (all metrics as exportable CSV rows)
    report_rows = []
    for metric_name, value in summary.items():
        if metric_name in ("aps113_section",):
            continue
        report_rows.append({
            "metric": metric_name,
            "value": value,
            "acceptable_range": _acceptable_range(metric_name),
            "status": _metric_status(metric_name, value),
        })
    results["validation_report"] = pd.DataFrame(report_rows)

    _log_validation_summary(summary)
    return results


def _acceptable_range(metric_name: str) -> str:
    ranges = {
        "gini": "> 0.50",
        "calibration_ratio": "0.85 – 1.15",
        "hl_p_value": "> 0.05",
        "psi": "< 0.10 stable",
        "weighted_bias": "~0.00",
    }
    return ranges.get(metric_name, "—")


def _metric_status(metric_name: str, value) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if np.isnan(v):
        return "insufficient_data"
    if metric_name == "gini":
        return "PASS" if v > GINI_THRESHOLD else "FAIL"
    if metric_name == "calibration_ratio":
        return "PASS" if CALIBRATION_RATIO_LOWER <= v <= CALIBRATION_RATIO_UPPER else "FAIL"
    if metric_name == "hl_p_value":
        return "PASS" if v > HL_P_VALUE_THRESHOLD else "FAIL"
    if metric_name == "psi":
        return "stable" if v < PSI_STABLE else "monitor" if v < PSI_MONITOR else "unstable"
    return "N/A"


def _log_validation_summary(summary: dict) -> None:
    gini = summary.get("gini", np.nan)
    cal = summary.get("calibration_ratio", np.nan)
    psi = summary.get("psi", np.nan)
    hl = summary.get("hl_p_value", np.nan)
    logger.info(
        "Validation summary: Gini=%.3f [%s] | CalRatio=%.3f [%s] | "
        "PSI=%.3f [%s] | HL_p=%.3f [%s]",
        gini or 0, "OK" if (gini or 0) > GINI_THRESHOLD else "FAIL",
        cal or 0, "OK" if CALIBRATION_RATIO_LOWER <= (cal or 0) <= CALIBRATION_RATIO_UPPER else "FAIL",
        psi or 0, "stable" if (psi or 1) < PSI_STABLE else "monitor",
        hl or 0, "OK" if (hl or 0) > HL_P_VALUE_THRESHOLD else "FAIL",
    )
