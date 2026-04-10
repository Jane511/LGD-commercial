"""
LGD model validation and backtesting framework.

Provides metrics and diagnostics aligned with APRA model risk management
expectations for IRB LGD models:

  1. Accuracy metrics       (MAE, RMSE, R-squared)
  2. Discriminatory power   (Spearman rank correlation, AUC on loss bands)
  3. Calibration            (predicted vs actual by segment)
  4. Conservatism testing   (predicted >= actual at portfolio and segment level)
  5. Stability monitoring   (PSI on predicted LGD distributions)
  6. Out-of-time testing    (vintage-based holdout performance)
  7. Sensitivity analysis   (parameter perturbation)
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================================================
# 1. ACCURACY METRICS
# ==========================================================================

def accuracy_metrics(actual, predicted):
    """
    Compute standard accuracy metrics.

    Returns dict with MAE, RMSE, R-squared, mean bias.
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    mask = np.isfinite(actual) & np.isfinite(predicted)
    a, p = actual[mask], predicted[mask]

    mae = mean_absolute_error(a, p)
    rmse = np.sqrt(mean_squared_error(a, p))
    r2 = r2_score(a, p) if len(a) > 1 else np.nan
    mean_bias = (p - a).mean()

    return {
        "MAE": round(mae, 6),
        "RMSE": round(rmse, 6),
        "R_squared": round(r2, 6),
        "Mean_Bias": round(mean_bias, 6),
        "N": int(mask.sum()),
    }


# ==========================================================================
# 2. DISCRIMINATORY POWER
# ==========================================================================

def discriminatory_power(actual, predicted):
    """
    Assess rank-ordering ability.

    Returns Spearman correlation and p-value.
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    mask = np.isfinite(actual) & np.isfinite(predicted)
    a, p = actual[mask], predicted[mask]

    if len(a) < 3:
        return {"Spearman_Corr": np.nan, "Spearman_PValue": np.nan, "N": len(a)}

    corr, pval = stats.spearmanr(a, p)
    return {
        "Spearman_Corr": round(corr, 6),
        "Spearman_PValue": round(pval, 6),
        "N": int(len(a)),
    }


# ==========================================================================
# 3. CALIBRATION BY SEGMENT
# ==========================================================================

def calibration_by_segment(df, actual_col, predicted_col, segment_col):
    """
    Compare mean predicted vs mean actual LGD by segment.

    Returns DataFrame with:
      - segment, count, mean_actual, mean_predicted, ratio, difference
    """
    grouped = df.groupby(segment_col).agg(
        count=(actual_col, "size"),
        mean_actual=(actual_col, "mean"),
        mean_predicted=(predicted_col, "mean"),
    ).reset_index()

    grouped["ratio"] = grouped["mean_predicted"] / grouped["mean_actual"].replace(0, np.nan)
    grouped["difference"] = grouped["mean_predicted"] - grouped["mean_actual"]
    grouped["is_conservative"] = grouped["mean_predicted"] >= grouped["mean_actual"]

    return grouped


# ==========================================================================
# 4. CONSERVATISM TESTING
# ==========================================================================

def conservatism_test(actual, predicted, confidence=0.95):
    """
    Test whether predicted LGD is conservative (>= actual).

    Returns:
      - Portfolio-level: mean predicted >= mean actual?
      - Statistical test: one-sided t-test that predicted >= actual
      - Conservatism ratio: mean_predicted / mean_actual
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    mask = np.isfinite(actual) & np.isfinite(predicted)
    a, p = actual[mask], predicted[mask]

    mean_a = a.mean()
    mean_p = p.mean()

    # One-sided paired t-test: H0: predicted <= actual, H1: predicted > actual
    if len(a) > 1:
        t_stat, two_sided_p = stats.ttest_rel(p, a)
        one_sided_p = two_sided_p / 2 if t_stat > 0 else 1 - two_sided_p / 2
    else:
        t_stat, one_sided_p = np.nan, np.nan

    return {
        "Mean_Actual": round(mean_a, 6),
        "Mean_Predicted": round(mean_p, 6),
        "Conservatism_Ratio": round(mean_p / mean_a, 4) if mean_a > 0 else np.nan,
        "Is_Conservative": bool(mean_p >= mean_a),
        "T_Statistic": round(t_stat, 4) if np.isfinite(t_stat) else np.nan,
        "P_Value_OneSided": round(one_sided_p, 6) if np.isfinite(one_sided_p) else np.nan,
        "N": int(len(a)),
    }


# ==========================================================================
# 5. POPULATION STABILITY INDEX (PSI)
# ==========================================================================

def compute_psi(baseline, current, n_bins=10):
    """
    Compute Population Stability Index between two distributions.

    PSI = Sum( (Actual_% - Expected_%) * ln(Actual_% / Expected_%) )

    Interpretation:
      PSI < 0.10  -> No significant shift
      PSI 0.10-0.25 -> Moderate shift, investigate
      PSI > 0.25 -> Significant shift, action required
    """
    baseline = np.asarray(baseline)
    current = np.asarray(current)

    # Create bins from baseline
    _, bin_edges = np.histogram(baseline, bins=n_bins)

    # Compute proportions
    baseline_counts = np.histogram(baseline, bins=bin_edges)[0]
    current_counts = np.histogram(current, bins=bin_edges)[0]

    # Avoid zeros
    baseline_pct = (baseline_counts + 1) / (baseline_counts.sum() + n_bins)
    current_pct = (current_counts + 1) / (current_counts.sum() + n_bins)

    psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))

    # Per-bin detail
    detail = pd.DataFrame({
        "bin_lower": bin_edges[:-1],
        "bin_upper": bin_edges[1:],
        "baseline_pct": np.round(baseline_pct, 6),
        "current_pct": np.round(current_pct, 6),
        "psi_contribution": np.round(
            (current_pct - baseline_pct) * np.log(current_pct / baseline_pct), 6
        ),
    })

    interpretation = (
        "No significant shift" if psi < 0.10
        else "Moderate shift - investigate" if psi < 0.25
        else "Significant shift - action required"
    )

    return {
        "PSI": round(psi, 6),
        "Interpretation": interpretation,
        "Detail": detail,
    }


# ==========================================================================
# 6. OUT-OF-TIME TESTING
# ==========================================================================

def out_of_time_split(df, date_col="default_date", holdout_start="2023-01-01"):
    """
    Split dataset into training (before holdout_start) and test (after).

    Returns (train_df, test_df).
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    cutoff = pd.Timestamp(holdout_start)
    train = df[df[date_col] < cutoff].copy()
    test = df[df[date_col] >= cutoff].copy()
    return train, test


def out_of_time_backtest(train_df, test_df, lgd_col="realised_lgd",
                         segment_col=None, ead_col="ead"):
    """
    Backtest: use training-period segment averages to predict test-period LGD.

    Returns dict with accuracy metrics and calibration.
    """
    if segment_col is None:
        # Portfolio-level: use training mean as prediction
        train_mean = train_df[lgd_col].mean()
        test_df = test_df.copy()
        test_df["predicted_lgd"] = train_mean
    else:
        # Segment-level
        if isinstance(segment_col, str):
            segment_col = [segment_col]
        seg_means = train_df.groupby(segment_col)[lgd_col].mean().reset_index()
        seg_means.columns = list(segment_col) + ["predicted_lgd"]
        test_df = test_df.merge(seg_means, on=segment_col, how="left")
        # Fill missing segments with portfolio average
        test_df["predicted_lgd"] = test_df["predicted_lgd"].fillna(train_df[lgd_col].mean())

    metrics = accuracy_metrics(test_df[lgd_col], test_df["predicted_lgd"])
    cons = conservatism_test(test_df[lgd_col], test_df["predicted_lgd"])
    disc = discriminatory_power(test_df[lgd_col], test_df["predicted_lgd"])

    return {
        "accuracy": metrics,
        "conservatism": cons,
        "discriminatory_power": disc,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "test_predictions": test_df,
    }


# ==========================================================================
# 7. SENSITIVITY ANALYSIS
# ==========================================================================

def sensitivity_analysis(base_lgd, parameter_name, parameter_values, compute_fn):
    """
    Run sensitivity analysis by varying a single parameter.

    Parameters
    ----------
    base_lgd : array-like, baseline LGD values
    parameter_name : str, name of parameter being varied
    parameter_values : list of values to test
    compute_fn : callable(param_value) -> array of LGD values

    Returns DataFrame with parameter_value, mean_lgd, std_lgd, p5, p95.
    """
    results = []
    base_mean = np.mean(base_lgd)

    for pv in parameter_values:
        lgd_vals = compute_fn(pv)
        results.append({
            "parameter": parameter_name,
            "value": pv,
            "mean_lgd": round(np.mean(lgd_vals), 6),
            "std_lgd": round(np.std(lgd_vals), 6),
            "p5_lgd": round(np.percentile(lgd_vals, 5), 6),
            "p95_lgd": round(np.percentile(lgd_vals, 95), 6),
            "change_from_base": round(np.mean(lgd_vals) - base_mean, 6),
        })

    return pd.DataFrame(results)


# ==========================================================================
# FULL VALIDATION REPORT
# ==========================================================================

def generate_validation_report(df, actual_col="realised_lgd",
                               predicted_col="lgd_final",
                               segment_col=None, date_col="default_date"):
    """
    Generate a comprehensive validation report for an LGD model.

    Returns dict with all validation components.
    """
    report = {}

    # Accuracy
    report["accuracy"] = accuracy_metrics(df[actual_col], df[predicted_col])

    # Discriminatory power
    report["discriminatory_power"] = discriminatory_power(
        df[actual_col], df[predicted_col]
    )

    # Conservatism
    report["conservatism"] = conservatism_test(df[actual_col], df[predicted_col])

    # Calibration by segment
    if segment_col:
        report["calibration"] = calibration_by_segment(
            df, actual_col, predicted_col, segment_col
        )

    # PSI (split at median date for illustration)
    if date_col in df.columns:
        dates = pd.to_datetime(df[date_col])
        median_date = dates.median()
        early = df[dates < median_date][predicted_col].values
        late = df[dates >= median_date][predicted_col].values
        if len(early) > 10 and len(late) > 10:
            report["stability_psi"] = compute_psi(early, late)

    # Out-of-time
    if date_col in df.columns:
        train, test = out_of_time_split(df, date_col)
        if len(test) > 5:
            report["out_of_time"] = out_of_time_backtest(
                train, test, lgd_col=actual_col, segment_col=segment_col
            )

    # Calibration by industry (if available)
    if "industry_risk_band" in df.columns:
        report["calibration_by_industry"] = calibration_by_segment(
            df, actual_col, predicted_col, "industry_risk_band"
        )
    elif "industry" in df.columns:
        report["calibration_by_industry"] = calibration_by_segment(
            df, actual_col, predicted_col, "industry"
        )

    # PD score band calibration (for cash flow lending)
    if "pd_score_band" in df.columns:
        report["calibration_by_pd_band"] = calibration_by_score_band(
            df, actual_col, predicted_col, band_col="pd_score_band"
        )
    if "pd_estimate" in df.columns and "pd_score_band" in df.columns:
        report["pd_lgd_consistency"] = pd_lgd_consistency_check(
            df, pd_col="pd_estimate", lgd_col=predicted_col,
            band_col="pd_score_band"
        )

    return report


# ==========================================================================
# 8. INDUSTRY RISK ATTRIBUTION ANALYSIS
# ==========================================================================

def industry_attribution_analysis(df, actual_col="realised_lgd",
                                  predicted_col="lgd_final",
                                  industry_col="industry_risk_band",
                                  security_col="security_type"):
    """
    Quantify how much LGD variation is explained by industry risk vs other factors.

    Returns dict with:
      - R-squared from industry alone
      - Incremental R-squared from adding industry to a security-type model
      - Calibration by industry risk band
    """
    from sklearn.preprocessing import LabelEncoder

    result = {}
    df_clean = df.dropna(subset=[actual_col, industry_col])

    if len(df_clean) < 10:
        return {"error": "Insufficient data for attribution analysis"}

    le_ind = LabelEncoder()
    ind_encoded = le_ind.fit_transform(df_clean[industry_col])

    # Industry alone: group-mean model
    ind_means = df_clean.groupby(industry_col)[actual_col].transform("mean")
    ss_total = ((df_clean[actual_col] - df_clean[actual_col].mean()) ** 2).sum()
    ss_resid_ind = ((df_clean[actual_col] - ind_means) ** 2).sum()
    r2_industry = 1 - ss_resid_ind / ss_total if ss_total > 0 else 0.0
    result["r2_industry_alone"] = round(r2_industry, 6)

    # Security type alone (if available)
    if security_col in df_clean.columns:
        sec_means = df_clean.groupby(security_col)[actual_col].transform("mean")
        ss_resid_sec = ((df_clean[actual_col] - sec_means) ** 2).sum()
        r2_security = 1 - ss_resid_sec / ss_total if ss_total > 0 else 0.0
        result["r2_security_alone"] = round(r2_security, 6)

        # Combined: security + industry group means
        combined_means = df_clean.groupby(
            [security_col, industry_col]
        )[actual_col].transform("mean")
        ss_resid_combined = ((df_clean[actual_col] - combined_means) ** 2).sum()
        r2_combined = 1 - ss_resid_combined / ss_total if ss_total > 0 else 0.0
        result["r2_combined"] = round(r2_combined, 6)
        result["r2_incremental_industry"] = round(r2_combined - r2_security, 6)

    # Calibration by industry
    result["calibration_by_industry"] = calibration_by_segment(
        df_clean, actual_col,
        predicted_col if predicted_col in df_clean.columns else actual_col,
        industry_col
    )

    return result


# ==========================================================================
# 9. PD-LGD CONSISTENCY VALIDATION
# ==========================================================================

def calibration_by_score_band(df, actual_col="realised_lgd",
                              predicted_col="lgd_final",
                              pd_col="pd_estimate",
                              band_col="pd_score_band"):
    """
    Validate LGD calibration across PD score bands.

    Checks that:
    - LGD increases monotonically with PD band (A < B < ... < E)
    - Each band is conservatively calibrated
    - PD-LGD correlation is positive (internal consistency)

    Returns dict with calibration table, monotonicity check, and
    PD-LGD rank correlation.
    """
    result = {}

    # Calibration table by PD band
    cal = calibration_by_segment(df, actual_col, predicted_col, band_col)
    band_order = ["A", "B", "C", "D", "E"]
    cal[band_col] = pd.Categorical(cal[band_col], categories=band_order, ordered=True)
    cal = cal.sort_values(band_col)
    result["calibration_table"] = cal

    # Monotonicity: predicted LGD should increase across bands A -> E
    predicted_means = cal["mean_predicted"].values
    is_monotonic = all(
        predicted_means[i] <= predicted_means[i + 1]
        for i in range(len(predicted_means) - 1)
        if np.isfinite(predicted_means[i]) and np.isfinite(predicted_means[i + 1])
    )
    result["is_monotonic"] = is_monotonic

    # PD-LGD rank correlation
    if pd_col in df.columns:
        mask = np.isfinite(df[pd_col]) & np.isfinite(df[predicted_col])
        if mask.sum() > 5:
            corr, pval = stats.spearmanr(
                df.loc[mask, pd_col], df.loc[mask, predicted_col]
            )
            result["pd_lgd_correlation"] = round(corr, 6)
            result["pd_lgd_pvalue"] = round(pval, 6)

    # Mean PD and LGD by band for EL computation
    if pd_col in df.columns:
        el_table = df.groupby(band_col).agg(
            mean_pd=(pd_col, "mean"),
            mean_lgd=(predicted_col, "mean"),
            count=(actual_col, "size"),
        ).reset_index()
        el_table["implied_el"] = el_table["mean_pd"] * el_table["mean_lgd"]
        el_table[band_col] = pd.Categorical(
            el_table[band_col], categories=band_order, ordered=True
        )
        result["el_table"] = el_table.sort_values(band_col)

    return result


def pd_lgd_consistency_check(df, pd_col="pd_estimate",
                             lgd_col="lgd_final",
                             band_col="pd_score_band"):
    """
    Check PD-LGD internal consistency for APRA compliance.

    APRA APS 113 requires that PD and LGD estimates are internally
    consistent -- i.e., they should not be independently optimistic.

    Returns dict with:
    - Correlation test (should be positive)
    - Implied EL by band
    - Joint conservatism test
    - Downturn dependency check
    """
    result = {}

    # 1. PD-LGD correlation (should be positive for consistency)
    mask = np.isfinite(df[pd_col]) & np.isfinite(df[lgd_col])
    df_clean = df[mask]
    if len(df_clean) > 5:
        corr, pval = stats.spearmanr(df_clean[pd_col], df_clean[lgd_col])
        result["correlation"] = {
            "spearman_rho": round(corr, 6),
            "p_value": round(pval, 6),
            "is_positive": bool(corr > 0),
            "interpretation": (
                "Consistent: higher PD associated with higher LGD"
                if corr > 0
                else "WARNING: negative PD-LGD correlation suggests inconsistency"
            ),
        }

    # 2. Implied EL by score band
    band_order = ["A", "B", "C", "D", "E"]
    band_stats = df_clean.groupby(band_col).agg(
        n=(pd_col, "size"),
        mean_pd=(pd_col, "mean"),
        mean_lgd=(lgd_col, "mean"),
        total_ead=("ead", "sum") if "ead" in df_clean.columns else (pd_col, "size"),
    ).reset_index()
    band_stats["implied_el"] = band_stats["mean_pd"] * band_stats["mean_lgd"]
    band_stats[band_col] = pd.Categorical(
        band_stats[band_col], categories=band_order, ordered=True
    )
    result["el_by_band"] = band_stats.sort_values(band_col)

    # 3. Portfolio-level implied EL
    portfolio_el = (df_clean[pd_col] * df_clean[lgd_col]).mean()
    result["portfolio_implied_el"] = round(portfolio_el, 6)

    # 4. Joint conservatism: is EL increasing across bands?
    el_vals = result["el_by_band"]["implied_el"].values
    el_monotonic = all(
        el_vals[i] <= el_vals[i + 1]
        for i in range(len(el_vals) - 1)
        if np.isfinite(el_vals[i]) and np.isfinite(el_vals[i + 1])
    )
    result["el_monotonic"] = el_monotonic

    return result


def compare_models(df, actual_col="realised_lgd",
                   baseline_col="lgd_final_baseline",
                   enhanced_col="lgd_final"):
    """
    Side-by-side validation of baseline vs industry-enhanced LGD model.

    Returns DataFrame comparing accuracy, discriminatory power, conservatism,
    and stability metrics for both models.
    """
    comparison = {}

    # Accuracy
    acc_base = accuracy_metrics(df[actual_col], df[baseline_col])
    acc_enh = accuracy_metrics(df[actual_col], df[enhanced_col])
    comparison["accuracy"] = pd.DataFrame({
        "Metric": list(acc_base.keys()),
        "Baseline": list(acc_base.values()),
        "Enhanced": list(acc_enh.values()),
    })

    # Discriminatory power
    disc_base = discriminatory_power(df[actual_col], df[baseline_col])
    disc_enh = discriminatory_power(df[actual_col], df[enhanced_col])
    comparison["discriminatory_power"] = pd.DataFrame({
        "Metric": list(disc_base.keys()),
        "Baseline": list(disc_base.values()),
        "Enhanced": list(disc_enh.values()),
    })

    # Conservatism
    cons_base = conservatism_test(df[actual_col], df[baseline_col])
    cons_enh = conservatism_test(df[actual_col], df[enhanced_col])
    comparison["conservatism"] = pd.DataFrame({
        "Metric": list(cons_base.keys()),
        "Baseline": list(cons_base.values()),
        "Enhanced": list(cons_enh.values()),
    })

    # PSI between models
    if len(df) > 20:
        psi = compute_psi(
            df[baseline_col].dropna().values,
            df[enhanced_col].dropna().values,
        )
        comparison["model_shift_psi"] = psi

    return comparison
