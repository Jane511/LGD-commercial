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
import logging
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


def _weighted_mean(values, weights=None):
    """Return weighted mean with safe fallbacks."""
    v = pd.to_numeric(pd.Series(values), errors="coerce")
    mask = np.isfinite(v)
    if weights is None:
        return float(v[mask].mean()) if mask.any() else np.nan
    w = pd.to_numeric(pd.Series(weights), errors="coerce").fillna(0.0)
    mask = mask & np.isfinite(w)
    w_sum = w[mask].sum()
    if w_sum <= 0:
        return float(v[mask].mean()) if mask.any() else np.nan
    return float((v[mask] * w[mask]).sum() / w_sum)


def _weighted_rmse(actual, predicted, weights=None):
    """Return weighted RMSE with safe fallback to unweighted RMSE."""
    a = pd.to_numeric(pd.Series(actual), errors="coerce")
    p = pd.to_numeric(pd.Series(predicted), errors="coerce")
    mask = np.isfinite(a) & np.isfinite(p)
    if not mask.any():
        return np.nan
    if weights is None:
        return float(np.sqrt(np.mean((a[mask] - p[mask]) ** 2)))
    w = pd.to_numeric(pd.Series(weights), errors="coerce").fillna(0.0)
    mask = mask & np.isfinite(w)
    if not mask.any():
        return np.nan
    w_sum = w[mask].sum()
    if w_sum <= 0:
        return float(np.sqrt(np.mean((a[mask] - p[mask]) ** 2)))
    return float(np.sqrt(np.sum(w[mask] * (a[mask] - p[mask]) ** 2) / w_sum))


def weighted_accuracy_metrics(actual, predicted, exposure=None):
    """
    Exposure-weighted accuracy metrics for LGD validation reporting.
    """
    mae = _weighted_mean(np.abs(pd.Series(predicted) - pd.Series(actual)), exposure)
    rmse = _weighted_rmse(actual, predicted, exposure)
    mean_actual = _weighted_mean(actual, exposure)
    mean_predicted = _weighted_mean(predicted, exposure)
    return {
        "Weighted_MAE": round(mae, 6) if np.isfinite(mae) else np.nan,
        "Weighted_RMSE": round(rmse, 6) if np.isfinite(rmse) else np.nan,
        "Weighted_Mean_Actual": round(mean_actual, 6) if np.isfinite(mean_actual) else np.nan,
        "Weighted_Mean_Predicted": round(mean_predicted, 6) if np.isfinite(mean_predicted) else np.nan,
        "Weighted_Bias": (
            round(mean_predicted - mean_actual, 6)
            if np.isfinite(mean_actual) and np.isfinite(mean_predicted)
            else np.nan
        ),
    }


def governance_flag_summary(df, ead_col="ead"):
    """
    Summarise governance proxy/fallback columns when present.
    """
    if len(df) == 0:
        return pd.DataFrame(columns=[
            "column", "value", "loan_count", "total_ead", "ead_share"
        ])

    work = df.copy()
    if ead_col not in work.columns:
        work[ead_col] = 1.0

    candidate_cols = [
        col for col in work.columns
        if col.endswith("_source")
        or col.endswith("_flag")
        or col in {"unemployment_year_bucket", "arrears_stage_proxy", "repayment_behaviour_proxy"}
    ]
    if not candidate_cols:
        return pd.DataFrame(columns=[
            "column", "value", "loan_count", "total_ead", "ead_share"
        ])

    parts = []
    for col in candidate_cols:
        tmp = work[[col, ead_col]].copy()
        tmp[col] = tmp[col].astype(str).fillna("missing")
        grouped = (
            tmp.groupby(col, observed=True)
            .agg(loan_count=(col, "size"), total_ead=(ead_col, "sum"))
            .reset_index()
            .rename(columns={col: "value"})
        )
        total_ead = grouped["total_ead"].sum()
        grouped["ead_share"] = grouped["total_ead"] / total_ead if total_ead > 0 else 0.0
        grouped.insert(0, "column", col)
        parts.append(grouped)

    return pd.concat(parts, ignore_index=True).sort_values(
        ["column", "value"]
    ).reset_index(drop=True)


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

def calibration_by_segment(df, actual_col, predicted_col, segment_col, ead_col="ead"):
    """
    Compare mean predicted vs mean actual LGD by segment.

    Returns DataFrame with:
      - segment, count, mean_actual, mean_predicted, ratio, difference
    """
    if ead_col in df.columns:
        grouped = (
            df.groupby(segment_col, observed=True)
            .apply(
                lambda g: pd.Series(
                    {
                        "count": len(g),
                        "total_ead": pd.to_numeric(g[ead_col], errors="coerce").fillna(0.0).sum(),
                        "mean_actual": _weighted_mean(g[actual_col], g[ead_col]),
                        "mean_predicted": _weighted_mean(g[predicted_col], g[ead_col]),
                    }
                ),
                include_groups=False,
            )
            .reset_index()
        )
    else:
        grouped = df.groupby(segment_col, observed=True).agg(
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

def conservatism_test(actual, predicted, exposure=None, confidence=0.95):
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

    if exposure is not None:
        exp = np.asarray(exposure)
        exp = exp[mask] if len(exp) == len(mask) else np.ones(len(a))
        mean_a = _weighted_mean(a, exp)
        mean_p = _weighted_mean(p, exp)
    else:
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


def add_vintage_columns(
    df,
    date_col="default_date",
    origination_date_col="origination_date",
    seasoning_months_col="seasoning_months",
    fallback_years_on_book=2.5,
    require_observed=False,
):
    """
    Add origination/default year fields with transparent fallback lineage.

    Fallback order:
      1) observed origination_date (if present)
      2) derive using seasoning_months (if present)
      3) proxy years-on-book fallback (constant years by product config)

    Parameters
    ----------
    require_observed : bool
        If True, raise ValueError when any row reaches the constant
        years-on-book fallback (tier 3). Use this in production/OOT contexts
        where imputed vintages would corrupt holdout analysis.
    """
    out = df.copy()
    default_dates = pd.to_datetime(out.get(date_col), errors="coerce")
    out[date_col] = default_dates
    out["default_year"] = default_dates.dt.year.astype("Int64")

    observed_orig = (
        pd.to_datetime(out[origination_date_col], errors="coerce")
        if origination_date_col in out.columns
        else pd.Series(pd.NaT, index=out.index)
    )
    orig_dates = observed_orig.copy()
    source = pd.Series(
        np.where(observed_orig.notna(), "observed_origination_date", "missing"),
        index=out.index,
        dtype="object",
    )

    if seasoning_months_col in out.columns:
        seasoning = pd.to_numeric(out[seasoning_months_col], errors="coerce")
        mask = (
            orig_dates.isna()
            & default_dates.notna()
            & seasoning.notna()
            & (seasoning >= 0)
        )
        if mask.any():
            days = (seasoning.loc[mask] * 30.4375).round().astype(int)
            orig_dates.loc[mask] = default_dates.loc[mask] - pd.to_timedelta(days, unit="D")
            source.loc[mask] = "derived_from_seasoning_months"

    mask = orig_dates.isna() & default_dates.notna()
    if mask.any():
        n_fallback = int(mask.sum())
        if require_observed:
            raise ValueError(
                f"add_vintage_columns: {n_fallback} row(s) have no origination_date and no "
                f"seasoning_months, so the constant {fallback_years_on_book}y proxy would be "
                f"applied. Set require_observed=False to allow this fallback, or supply "
                f"origination_date / seasoning_months for all rows."
            )
        logger.warning(
            "add_vintage_columns: %d row(s) are using the constant %.1fy years-on-book fallback "
            "for origination_date. OOT validation using these rows may be unreliable.",
            n_fallback,
            fallback_years_on_book,
        )
        fallback_days = int(round(max(float(fallback_years_on_book), 0.0) * 365.25))
        orig_dates.loc[mask] = default_dates.loc[mask] - pd.to_timedelta(fallback_days, unit="D")
        source.loc[mask] = f"proxy_years_on_book_{float(fallback_years_on_book):.1f}y"

    source.loc[default_dates.isna()] = "missing_default_date"

    # Log fallback tier summary for audit trail
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "add_vintage_columns origination_date fallback summary: %s",
            source.value_counts().to_dict(),
        )

    out["origination_date_derived"] = orig_dates
    out["origination_year"] = orig_dates.dt.year.astype("Int64")
    out["origination_year_source"] = source
    return out


def _weighted_group_summary(df, group_cols, actual_col, predicted_col, ead_col="ead"):
    """Weighted LGD summary helper by arbitrary grouping columns."""
    if len(df) == 0:
        cols = list(group_cols) + [
            "loan_count",
            "total_ead",
            "ead_weighted_actual_lgd",
            "ead_weighted_predicted_lgd",
            "weighted_lgd_gap_pred_minus_actual",
        ]
        return pd.DataFrame(columns=cols)

    work = df.copy()
    if ead_col not in work.columns:
        work[ead_col] = 1.0
    for col in [actual_col, predicted_col, ead_col]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    grouped = (
        work.groupby(group_cols, observed=True)
        .apply(
            lambda g: pd.Series(
                {
                    "loan_count": len(g),
                    "total_ead": pd.to_numeric(g[ead_col], errors="coerce").fillna(0.0).sum(),
                    "ead_weighted_actual_lgd": _weighted_mean(g[actual_col], g[ead_col]),
                    "ead_weighted_predicted_lgd": _weighted_mean(g[predicted_col], g[ead_col]),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    grouped["weighted_lgd_gap_pred_minus_actual"] = (
        grouped["ead_weighted_predicted_lgd"] - grouped["ead_weighted_actual_lgd"]
    )
    return grouped


def build_vintage_lgd_summary(
    df,
    actual_col="realised_lgd",
    predicted_col="lgd_final",
    ead_col="ead",
    vintage_col="origination_year",
    default_year_col="default_year",
):
    """
    Build exposure-weighted LGD by origination/default year vintage.
    """
    cols = [vintage_col, default_year_col]
    work = df.copy()
    if vintage_col not in work.columns:
        work[vintage_col] = pd.NA
    if default_year_col not in work.columns:
        work[default_year_col] = pd.NA
    out = _weighted_group_summary(work, cols, actual_col, predicted_col, ead_col=ead_col)
    return out.sort_values(cols).reset_index(drop=True)


def build_weighted_lgd_over_time(
    df,
    actual_col="realised_lgd",
    predicted_col="lgd_final",
    ead_col="ead",
    time_col="default_year",
):
    """
    Build exposure-weighted LGD time series using default-year buckets.
    """
    work = df.copy()
    if time_col not in work.columns:
        work[time_col] = pd.NA
    group_cols = [time_col]
    if "product" in work.columns:
        group_cols = ["product", time_col]
    out = _weighted_group_summary(work, group_cols, actual_col, predicted_col, ead_col=ead_col)
    return out.sort_values(group_cols).reset_index(drop=True)


def summarise_oot_stability(
    train_df,
    test_df,
    actual_col="realised_lgd",
    predicted_col="lgd_final",
    ead_col="ead",
):
    """
    Summarise train/test weighted LGD stability for out-of-time validation.
    """
    train_w_actual = _weighted_mean(
        train_df[actual_col],
        train_df[ead_col] if ead_col in train_df.columns else None,
    )
    test_w_actual = _weighted_mean(
        test_df[actual_col],
        test_df[ead_col] if ead_col in test_df.columns else None,
    )
    train_w_pred = _weighted_mean(
        train_df[predicted_col],
        train_df[ead_col] if ead_col in train_df.columns else None,
    )
    test_w_pred = _weighted_mean(
        test_df[predicted_col],
        test_df[ead_col] if ead_col in test_df.columns else None,
    )

    shift = test_w_actual - train_w_actual if np.isfinite(train_w_actual) and np.isfinite(test_w_actual) else np.nan
    rel_shift = (
        shift / train_w_actual
        if np.isfinite(shift) and np.isfinite(train_w_actual) and abs(train_w_actual) > 1e-12
        else np.nan
    )
    return {
        "train_weighted_actual_lgd": round(train_w_actual, 6) if np.isfinite(train_w_actual) else np.nan,
        "test_weighted_actual_lgd": round(test_w_actual, 6) if np.isfinite(test_w_actual) else np.nan,
        "train_weighted_predicted_lgd": round(train_w_pred, 6) if np.isfinite(train_w_pred) else np.nan,
        "test_weighted_predicted_lgd": round(test_w_pred, 6) if np.isfinite(test_w_pred) else np.nan,
        "weighted_actual_shift_test_minus_train": round(shift, 6) if np.isfinite(shift) else np.nan,
        "weighted_actual_shift_pct": round(rel_shift, 6) if np.isfinite(rel_shift) else np.nan,
    }


def ranking_consistency_summary(
    train_df,
    test_df,
    segment_col,
    lgd_col="realised_lgd",
    ead_col="ead",
):
    """
    Compare segment LGD rank ordering between train and out-of-time test periods.
    """
    if segment_col is None:
        return {
            "segment_col": None,
            "common_segments": 0,
            "spearman_rank_corr": np.nan,
            "top_segment_train": None,
            "top_segment_test": None,
            "top_segment_match": np.nan,
        }

    if isinstance(segment_col, (list, tuple)):
        if len(segment_col) != 1:
            return {
                "segment_col": str(segment_col),
                "common_segments": 0,
                "spearman_rank_corr": np.nan,
                "top_segment_train": None,
                "top_segment_test": None,
                "top_segment_match": np.nan,
            }
        segment_col = segment_col[0]

    if segment_col not in train_df.columns or segment_col not in test_df.columns:
        return {
            "segment_col": segment_col,
            "common_segments": 0,
            "spearman_rank_corr": np.nan,
            "top_segment_train": None,
            "top_segment_test": None,
            "top_segment_match": np.nan,
        }

    train_seg = _weighted_group_summary(
        train_df, [segment_col], lgd_col, lgd_col, ead_col=ead_col
    ).rename(columns={"ead_weighted_actual_lgd": "train_weighted_lgd"})
    test_seg = _weighted_group_summary(
        test_df, [segment_col], lgd_col, lgd_col, ead_col=ead_col
    ).rename(columns={"ead_weighted_actual_lgd": "test_weighted_lgd"})

    merged = train_seg[[segment_col, "train_weighted_lgd"]].merge(
        test_seg[[segment_col, "test_weighted_lgd"]],
        on=segment_col,
        how="inner",
    )
    merged = merged.dropna(subset=["train_weighted_lgd", "test_weighted_lgd"])
    n_common = len(merged)

    if n_common >= 2:
        corr, _ = stats.spearmanr(merged["train_weighted_lgd"], merged["test_weighted_lgd"])
    else:
        corr = np.nan

    top_train = None
    top_test = None
    if n_common >= 1:
        top_train = str(
            merged.sort_values("train_weighted_lgd", ascending=False).iloc[0][segment_col]
        )
        top_test = str(
            merged.sort_values("test_weighted_lgd", ascending=False).iloc[0][segment_col]
        )

    return {
        "segment_col": segment_col,
        "common_segments": int(n_common),
        "spearman_rank_corr": round(corr, 6) if np.isfinite(corr) else np.nan,
        "top_segment_train": top_train,
        "top_segment_test": top_test,
        "top_segment_match": bool(top_train == top_test) if top_train is not None and top_test is not None else np.nan,
    }


def out_of_time_backtest(train_df, test_df, lgd_col="realised_lgd",
                         segment_col=None, ead_col="ead"):
    """
    Backtest: use training-period segment averages to predict test-period LGD.

    Returns dict with accuracy metrics and calibration.
    """
    if segment_col is None:
        # Portfolio-level: use training exposure-weighted LGD as prediction
        train_mean = _weighted_mean(
            train_df[lgd_col],
            train_df[ead_col] if ead_col in train_df.columns else None,
        )
        test_df = test_df.copy()
        test_df["predicted_lgd"] = train_mean
    else:
        # Segment-level
        if isinstance(segment_col, str):
            segment_col = [segment_col]
        if ead_col in train_df.columns:
            seg_means = (
                train_df.groupby(segment_col, observed=True)
                .apply(
                    lambda g: _weighted_mean(g[lgd_col], g[ead_col]),
                    include_groups=False,
                )
                .reset_index(name="predicted_lgd")
            )
        else:
            seg_means = (
                train_df.groupby(segment_col, observed=True)
                .apply(lambda g: _weighted_mean(g[lgd_col]), include_groups=False)
                .reset_index(name="predicted_lgd")
            )
        test_df = test_df.merge(seg_means, on=segment_col, how="left")
        # Fill missing segments with portfolio average
        portfolio_lgd = _weighted_mean(
            train_df[lgd_col],
            train_df[ead_col] if ead_col in train_df.columns else None,
        )
        test_df["predicted_lgd"] = test_df["predicted_lgd"].fillna(portfolio_lgd)

    metrics = accuracy_metrics(test_df[lgd_col], test_df["predicted_lgd"])
    cons = conservatism_test(
        test_df[lgd_col],
        test_df["predicted_lgd"],
        exposure=test_df[ead_col] if ead_col in test_df.columns else None,
    )
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

def generate_validation_report(
    df,
    actual_col="realised_lgd",
    predicted_col="lgd_final",
    segment_col=None,
    date_col="default_date",
    oot_holdout_start="2023-01-01",
    fallback_years_on_book=2.5,
    ranking_segment_col=None,
):
    """
    Generate a comprehensive validation report for an LGD model.

    Returns dict with all validation components.
    """
    report = {}
    df = add_vintage_columns(
        df,
        date_col=date_col,
        fallback_years_on_book=fallback_years_on_book,
    )
    ead_col = "ead" if "ead" in df.columns else None

    # Accuracy
    report["accuracy"] = accuracy_metrics(df[actual_col], df[predicted_col])
    report["weighted_accuracy"] = weighted_accuracy_metrics(
        df[actual_col],
        df[predicted_col],
        exposure=df[ead_col] if ead_col is not None else None,
    )

    # Discriminatory power
    report["discriminatory_power"] = discriminatory_power(
        df[actual_col], df[predicted_col]
    )

    # Conservatism
    report["conservatism"] = conservatism_test(
        df[actual_col],
        df[predicted_col],
        exposure=df[ead_col] if ead_col is not None else None,
    )

    # Calibration by segment
    if segment_col:
        report["calibration"] = calibration_by_segment(
            df, actual_col, predicted_col, segment_col, ead_col="ead"
        )

    # Vintage and time-series weighted LGD views
    report["vintage_summary"] = build_vintage_lgd_summary(
        df,
        actual_col=actual_col,
        predicted_col=predicted_col,
        ead_col=ead_col or "ead",
        vintage_col="origination_year",
        default_year_col="default_year",
    )
    report["weighted_lgd_over_time"] = build_weighted_lgd_over_time(
        df,
        actual_col=actual_col,
        predicted_col=predicted_col,
        ead_col=ead_col or "ead",
        time_col="default_year",
    )
    if "origination_year_source" in df.columns:
        source_df = df.copy()
        if ead_col is None:
            source_df["ead"] = 1.0
        report["origination_year_source_summary"] = (
            source_df.groupby("origination_year_source", observed=True)
            .agg(
                loan_count=("origination_year_source", "size"),
                total_ead=("ead", "sum"),
            )
            .reset_index()
            .rename(columns={"origination_year_source": "source"})
            .sort_values("source")
            .reset_index(drop=True)
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
        train, test = out_of_time_split(df, date_col, holdout_start=oot_holdout_start)
        if len(test) > 5 and len(train) > 5:
            oot = out_of_time_backtest(
                train,
                test,
                lgd_col=actual_col,
                segment_col=segment_col,
                ead_col="ead",
            )
            oot["holdout_start"] = str(pd.Timestamp(oot_holdout_start).date())
            oot["train_period_start"] = (
                str(pd.to_datetime(train[date_col]).min().date())
                if len(train)
                else None
            )
            oot["train_period_end"] = (
                str(pd.to_datetime(train[date_col]).max().date())
                if len(train)
                else None
            )
            oot["test_period_start"] = (
                str(pd.to_datetime(test[date_col]).min().date())
                if len(test)
                else None
            )
            oot["test_period_end"] = (
                str(pd.to_datetime(test[date_col]).max().date())
                if len(test)
                else None
            )
            oot["stability_summary"] = summarise_oot_stability(
                train,
                test,
                actual_col=actual_col,
                predicted_col=predicted_col,
                ead_col="ead",
            )
            ranking_col = ranking_segment_col if ranking_segment_col is not None else segment_col
            oot["ranking_consistency"] = ranking_consistency_summary(
                train,
                test,
                segment_col=ranking_col,
                lgd_col=actual_col,
                ead_col="ead",
            )
            report["out_of_time"] = oot

    # Calibration by industry (if available)
    if "industry_risk_band" in df.columns:
        report["calibration_by_industry"] = calibration_by_segment(
            df, actual_col, predicted_col, "industry_risk_band", ead_col="ead"
        )
    elif "industry" in df.columns:
        report["calibration_by_industry"] = calibration_by_segment(
            df, actual_col, predicted_col, "industry", ead_col="ead"
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

    # Governance and fallback transparency section
    report["governance_flags"] = governance_flag_summary(
        df, ead_col="ead" if "ead" in df.columns else "ead"
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
    has_ead = "ead" in df_clean.columns

    if len(df_clean) < 10:
        return {"error": "Insufficient data for attribution analysis"}

    le_ind = LabelEncoder()
    ind_encoded = le_ind.fit_transform(df_clean[industry_col])

    # Industry alone: group-mean model (EAD-weighted when available)
    if has_ead:
        ind_lookup = (
            df_clean.groupby(industry_col, observed=True)
            .apply(
                lambda g: _weighted_mean(g[actual_col], g["ead"]),
                include_groups=False,
            )
            .to_dict()
        )
        ind_means = df_clean[industry_col].map(ind_lookup)
        portfolio_mean = _weighted_mean(df_clean[actual_col], df_clean["ead"])
    else:
        ind_means = df_clean.groupby(industry_col)[actual_col].transform("mean")
        portfolio_mean = df_clean[actual_col].mean()

    ss_total = ((df_clean[actual_col] - portfolio_mean) ** 2).sum()
    ss_resid_ind = ((df_clean[actual_col] - ind_means) ** 2).sum()
    r2_industry = 1 - ss_resid_ind / ss_total if ss_total > 0 else 0.0
    result["r2_industry_alone"] = round(r2_industry, 6)

    # Security type alone (if available)
    if security_col in df_clean.columns:
        if has_ead:
            sec_lookup = (
                df_clean.groupby(security_col, observed=True)
                .apply(
                    lambda g: _weighted_mean(g[actual_col], g["ead"]),
                    include_groups=False,
                )
                .to_dict()
            )
            sec_means = df_clean[security_col].map(sec_lookup)
        else:
            sec_means = df_clean.groupby(security_col)[actual_col].transform("mean")
        ss_resid_sec = ((df_clean[actual_col] - sec_means) ** 2).sum()
        r2_security = 1 - ss_resid_sec / ss_total if ss_total > 0 else 0.0
        result["r2_security_alone"] = round(r2_security, 6)

        # Combined: security + industry group means
        if has_ead:
            combo_lookup = (
                df_clean.groupby([security_col, industry_col], observed=True)
                .apply(
                    lambda g: _weighted_mean(g[actual_col], g["ead"]),
                    include_groups=False,
                )
                .to_dict()
            )
            combined_means = pd.Series(
                [
                    combo_lookup.get((sec, ind), np.nan)
                    for sec, ind in zip(df_clean[security_col], df_clean[industry_col])
                ],
                index=df_clean.index,
            )
        else:
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
        industry_col,
        ead_col="ead",
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
    cal = calibration_by_segment(
        df, actual_col, predicted_col, band_col, ead_col="ead"
    )
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
        if "ead" in df.columns:
            el_table = (
                df.groupby(band_col, observed=True)
                .apply(
                    lambda g: pd.Series(
                        {
                            "mean_pd": _weighted_mean(g[pd_col], g["ead"]),
                            "mean_lgd": _weighted_mean(g[predicted_col], g["ead"]),
                            "total_ead": pd.to_numeric(g["ead"], errors="coerce").fillna(0.0).sum(),
                            "count": len(g),
                        }
                    ),
                    include_groups=False,
                )
                .reset_index()
            )
        else:
            el_table = (
                df.groupby(band_col, observed=True)
                .apply(
                    lambda g: pd.Series(
                        {
                            "mean_pd": _weighted_mean(g[pd_col]),
                            "mean_lgd": _weighted_mean(g[predicted_col]),
                            "count": len(g),
                        }
                    ),
                    include_groups=False,
                )
                .reset_index()
            )
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
    if "ead" in df_clean.columns:
        band_stats = (
            df_clean.groupby(band_col, observed=True)
            .apply(
                lambda g: pd.Series(
                    {
                        "n": len(g),
                        "mean_pd": _weighted_mean(g[pd_col], g["ead"]),
                        "mean_lgd": _weighted_mean(g[lgd_col], g["ead"]),
                        "total_ead": pd.to_numeric(g["ead"], errors="coerce").fillna(0.0).sum(),
                    }
                ),
                include_groups=False,
            )
            .reset_index()
        )
    else:
        band_stats = (
            df_clean.groupby(band_col, observed=True)
            .apply(
                lambda g: pd.Series(
                    {
                        "n": len(g),
                        "mean_pd": _weighted_mean(g[pd_col]),
                        "mean_lgd": _weighted_mean(g[lgd_col]),
                        "total_ead": len(g),
                    }
                ),
                include_groups=False,
            )
            .reset_index()
        )
    band_stats["implied_el"] = band_stats["mean_pd"] * band_stats["mean_lgd"]
    band_stats[band_col] = pd.Categorical(
        band_stats[band_col], categories=band_order, ordered=True
    )
    result["el_by_band"] = band_stats.sort_values(band_col)

    # 3. Portfolio-level implied EL
    if "ead" in df_clean.columns:
        portfolio_el = _weighted_mean(
            df_clean[pd_col] * df_clean[lgd_col],
            df_clean["ead"],
        )
    else:
        portfolio_el = _weighted_mean(df_clean[pd_col] * df_clean[lgd_col])
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
    exposure = df["ead"] if "ead" in df.columns else None
    cons_base = conservatism_test(df[actual_col], df[baseline_col], exposure=exposure)
    cons_enh = conservatism_test(df[actual_col], df[enhanced_col], exposure=exposure)
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


# ---------------------------------------------------------------------------
# Extended IRB validation metrics (APS 113 s.66-68)
# ---------------------------------------------------------------------------

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
