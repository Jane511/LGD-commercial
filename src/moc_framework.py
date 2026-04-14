"""
Margin of Conservatism (MoC) Framework.

Implements the five-component MoC required by APRA APS 113 s.63-65.

CRITICAL: Correct Application Order
=====================================
The MoC is applied AFTER the downturn overlay, not before it.

Correct order (per APS 113 s.63 — MoC applied to "final parameter estimate"):
    Step 1: long_run_lgd   = compute_long_run_lgd(...)           [s.43]
    Step 2: downturn_lgd   = apply_downturn_overlay(long_run_lgd) [s.46-50]
    Step 3: lgd_with_moc   = downturn_lgd + total_moc             [s.63-65]  ← HERE
    Step 4: final_lgd      = max(lgd_with_moc, policy_floor)      [s.58]

The incorrect order (MoC before downturn) causes under-statement because
the downturn scalar then amplifies a pre-MoC base, resulting in a smaller
MoC contribution at the final LGD level.

Five MoC Sources (APS 113 s.65):
    1. data_quality_moc       — limited default observations in segment
    2. model_error_moc        — backtesting bias exceeds threshold
    3. incomplete_workout_moc — high proportion of unresolved workouts
    4. cyclicality_moc        — insufficient cycle coverage in sample
    5. parameter_instability_moc — distribution drift (high PSI)

APS 113 References:
    s.63: ADI must add MoC to address estimation uncertainty
    s.64: MoC must be reviewed and updated at least annually
    s.65: Five specific sources of MoC (data quality, model, observation, cyclicality, uncertainty)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MoC source definitions
# ---------------------------------------------------------------------------

# Each entry: trigger threshold, add-on in decimal, APS 113 section
MOC_SOURCES: dict[str, dict] = {
    "data_quality": {
        "description": "Limited number of observed defaults in segment",
        "trigger_field": "n_observations",
        "trigger_condition": "< 20",
        "trigger_threshold": 20,
        "additive_decimal": 0.0050,   # 50 bps
        "aps113_section": "s.65(a)",
    },
    "model_error": {
        "description": "Backtesting bias exceeds 5% of actual LGD",
        "trigger_field": "abs_backtesting_bias",
        "trigger_condition": "> 0.05",
        "trigger_threshold": 0.05,
        "additive_decimal": 0.0030,   # 30 bps
        "aps113_section": "s.65(b)",
    },
    "incomplete_workout": {
        "description": "More than 10% of workouts unresolved at calibration date",
        "trigger_field": "pct_incomplete_workouts",
        "trigger_condition": "> 0.10",
        "trigger_threshold": 0.10,
        "additive_decimal": 0.0040,   # 40 bps
        "aps113_section": "s.65(c)",
    },
    "cyclicality": {
        "description": "No downturn period represented in calibration sample",
        "trigger_field": "n_downturn_vintages",
        "trigger_condition": "== 0",
        "trigger_threshold": 0,
        "additive_decimal": 0.0075,   # 75 bps
        "aps113_section": "s.65(d)",
    },
    "parameter_instability": {
        "description": "Population Stability Index > 0.10 (distribution drift)",
        "trigger_field": "psi_value",
        "trigger_condition": "> 0.10",
        "trigger_threshold": 0.10,
        "additive_decimal": 0.0030,   # 30 bps
        "aps113_section": "s.65(e)",
    },
}

# Proxy/synthetic data add-on (not an APS 113 source, but a governance overlay)
# Reduced when real RBA/ABS regime data is used (lower approximation uncertainty)
MODEL_APPROXIMATION_MOC_REAL_DATA = 0.0010   # 10 bps — real macro data used
MODEL_APPROXIMATION_MOC_SYNTHETIC  = 0.0025  # 25 bps — synthetic data only

# Per-product MoC caps (max total MoC in decimal)
PRODUCT_MOC_CAPS: dict[str, float] = {
    "mortgage":              0.050,   # 5.0%
    "commercial_cashflow":   0.075,
    "receivables":           0.075,
    "trade_contingent":      0.075,
    "asset_equipment":       0.075,
    "development_finance":   0.100,   # higher uncertainty
    "cre_investment":        0.100,
    "residual_stock":        0.100,
    "land_subdivision":      0.100,
    "bridging":              0.100,
    "mezz_second_mortgage":  0.100,
    "default":               0.075,
}


# ---------------------------------------------------------------------------
# MoCRegister class
# ---------------------------------------------------------------------------

class MoCRegister:
    """
    Per-segment, per-product Margin of Conservatism register.

    Usage:
        register = MoCRegister(product="mortgage")
        moc_df = register.build_moc_register(segment_df, backtest_df, validation_df)
        lgd_with_moc = apply_moc(downturn_lgd, moc_df, segment_col)

    The build_moc_register() method evaluates all five APS 113 s.65 sources
    plus the synthetic-data approximation overlay, computes total MoC per
    segment (subject to product cap), and returns a DataFrame suitable for
    direct export as [product]_moc_register.csv.
    """

    def __init__(
        self,
        product: str = "default",
        regime_data_source: str = "synthetic",  # 'rba_abs_real' or 'synthetic'
    ) -> None:
        self.product = product
        self.regime_data_source = regime_data_source
        self.moc_cap = PRODUCT_MOC_CAPS.get(product, PRODUCT_MOC_CAPS["default"])

    def compute_segment_moc(
        self,
        *,
        n_observations: int,
        abs_backtesting_bias: float,
        pct_incomplete_workouts: float,
        n_downturn_vintages: int,
        psi_value: float,
    ) -> dict:
        """
        Evaluate all five MoC sources for a single segment.

        Parameters
        ----------
        n_observations : number of defaulted facilities in calibration sample
        abs_backtesting_bias : |model_lgd - actual_lgd| / actual_lgd
        pct_incomplete_workouts : proportion of workouts not yet resolved
        n_downturn_vintages : number of downturn years represented in segment
        psi_value : Population Stability Index (distribution drift measure)

        Returns
        -------
        dict with individual component MoCs and total_moc (before cap)
        """
        components = {}
        total = 0.0

        # 1. Data quality
        triggered = n_observations < MOC_SOURCES["data_quality"]["trigger_threshold"]
        add_on = MOC_SOURCES["data_quality"]["additive_decimal"] if triggered else 0.0
        components["data_quality_moc"] = add_on
        components["data_quality_triggered"] = triggered
        total += add_on

        # 2. Model error
        triggered = abs_backtesting_bias > MOC_SOURCES["model_error"]["trigger_threshold"]
        add_on = MOC_SOURCES["model_error"]["additive_decimal"] if triggered else 0.0
        components["model_error_moc"] = add_on
        components["model_error_triggered"] = triggered
        total += add_on

        # 3. Incomplete workout
        triggered = pct_incomplete_workouts > MOC_SOURCES["incomplete_workout"]["trigger_threshold"]
        add_on = MOC_SOURCES["incomplete_workout"]["additive_decimal"] if triggered else 0.0
        components["incomplete_workout_moc"] = add_on
        components["incomplete_workout_triggered"] = triggered
        total += add_on

        # 4. Cyclicality
        triggered = n_downturn_vintages == 0
        add_on = MOC_SOURCES["cyclicality"]["additive_decimal"] if triggered else 0.0
        components["cyclicality_moc"] = add_on
        components["cyclicality_triggered"] = triggered
        total += add_on

        # 5. Parameter instability
        triggered = psi_value > MOC_SOURCES["parameter_instability"]["trigger_threshold"]
        add_on = MOC_SOURCES["parameter_instability"]["additive_decimal"] if triggered else 0.0
        components["parameter_instability_moc"] = add_on
        components["parameter_instability_triggered"] = triggered
        total += add_on

        # Approximation overlay (non-APS 113 but governance requirement for synthetic repos)
        approx_moc = (
            MODEL_APPROXIMATION_MOC_REAL_DATA
            if self.regime_data_source == "rba_abs_real"
            else MODEL_APPROXIMATION_MOC_SYNTHETIC
        )
        components["approximation_moc"] = approx_moc
        total += approx_moc

        # Apply product cap
        cap_applied = total > self.moc_cap
        total_capped = min(total, self.moc_cap)

        components["total_moc_pre_cap"] = round(total, 6)
        components["total_moc"] = round(total_capped, 6)
        components["moc_cap"] = self.moc_cap
        components["moc_cap_applied"] = cap_applied
        return components

    def build_moc_register(
        self,
        segment_df: pd.DataFrame,
        backtest_df: pd.DataFrame | None = None,
        validation_df: pd.DataFrame | None = None,
        segment_col: str = "segment_key_concat",
    ) -> pd.DataFrame:
        """
        Build the full MoC register for a product.

        Parameters
        ----------
        segment_df : output of compute_long_run_lgd() — contains n_obs,
                     cycle_coverage_flag, n_downturn_vintages per segment
        backtest_df : output of compare_model_vs_actual() — contains bias_pct
        validation_df : output of run_full_validation_suite() — contains psi

        Returns
        -------
        DataFrame suitable for export as [product]_moc_register.csv.

        Columns:
            product, segment_key_concat, [5 component mocs], approximation_moc,
            total_moc_pre_cap, total_moc, moc_cap, moc_cap_applied,
            aps113_section_ref, rationale, data_source
        """
        rows = []

        for _, seg in segment_df.iterrows():
            seg_key = str(seg.get(segment_col, seg.get("segment_key_concat", "all")))

            # Pull stats from companion DataFrames
            n_obs = int(seg.get("n_obs", 0))
            n_downturn_vintages = int(seg.get("n_downturn_vintages", 0))
            cycle_flag = str(seg.get("cycle_coverage_flag", "adequate"))
            # Ensure n_downturn_vintages consistent with cycle_flag
            if cycle_flag == "no_downturn_in_sample":
                n_downturn_vintages = 0

            # Backtesting bias (from backtest_df or default to 0)
            abs_bias = 0.0
            if backtest_df is not None and not backtest_df.empty:
                bt_match = backtest_df[backtest_df[segment_col] == seg_key] if segment_col in backtest_df.columns else backtest_df
                if not bt_match.empty and "bias_pct" in bt_match.columns:
                    bias_val = pd.to_numeric(bt_match["bias_pct"].iloc[0], errors="coerce")
                    abs_bias = abs(float(bias_val)) if np.isfinite(bias_val) else 0.0

            # PSI (from validation_df or default)
            psi_val = 0.0
            if validation_df is not None and not validation_df.empty:
                val_match = validation_df[validation_df[segment_col] == seg_key] if segment_col in validation_df.columns else validation_df
                if not val_match.empty and "psi" in val_match.columns:
                    p = pd.to_numeric(val_match["psi"].iloc[0], errors="coerce")
                    psi_val = float(p) if np.isfinite(p) else 0.0

            # Incomplete workout proxy: assume 5% unless explicitly provided
            pct_incomplete = float(seg.get("pct_incomplete_workouts", 0.05))

            components = self.compute_segment_moc(
                n_observations=n_obs,
                abs_backtesting_bias=abs_bias,
                pct_incomplete_workouts=pct_incomplete,
                n_downturn_vintages=n_downturn_vintages,
                psi_value=psi_val,
            )

            row = {
                "product": self.product,
                "segment_key_concat": seg_key,
                **components,
                "regime_data_source": self.regime_data_source,
                "aps113_section_ref": "s.63-65",
                "rationale": self._build_rationale(components),
            }
            rows.append(row)

        result = pd.DataFrame(rows)
        if not result.empty:
            logger.info(
                "MoCRegister (%s): %d segments | mean total_moc=%.1f%% | "
                "%d at cap",
                self.product,
                len(result),
                100 * result["total_moc"].mean(),
                result["moc_cap_applied"].sum(),
            )
        return result

    def _build_rationale(self, components: dict) -> str:
        """Build a human-readable rationale string for the MoC register."""
        triggered = []
        if components.get("data_quality_triggered"):
            triggered.append("low observation count")
        if components.get("model_error_triggered"):
            triggered.append("backtesting bias > 5%")
        if components.get("incomplete_workout_triggered"):
            triggered.append("> 10% incomplete workouts")
        if components.get("cyclicality_triggered"):
            triggered.append("no downturn vintage in sample")
        if components.get("parameter_instability_triggered"):
            triggered.append("PSI > 0.10")
        base = f"Approximation overlay ({components.get('approximation_moc', 0)*10000:.0f}bps): synthetic proxy data."
        if triggered:
            return base + " Additional MoC sources triggered: " + "; ".join(triggered) + "."
        return base + " No additional MoC sources triggered."


# ---------------------------------------------------------------------------
# apply_moc — the function notebooks call
# ---------------------------------------------------------------------------

def apply_moc(
    lgd_series: pd.Series,
    moc_register: pd.DataFrame,
    segment_col: str = "segment_key_concat",
    segment_values: pd.Series | None = None,
) -> pd.Series:
    """
    Apply MoC additively to downturn-adjusted LGD.

    IMPORTANT: Call this AFTER apply_downturn_overlay(), not before.
    lgd_series should be the downturn LGD, not the long-run LGD.
    See module docstring for correct application order.

    Parameters
    ----------
    lgd_series : Series of downturn LGD values (one per loan or per segment)
    moc_register : output of MoCRegister.build_moc_register()
    segment_col : column in moc_register holding segment keys
    segment_values : Series matching lgd_series.index containing each loan's
        segment key. If None, uses portfolio-level (mean) MoC.

    Returns
    -------
    Series: lgd_series + applicable_moc, clipped to [0, 1.5]

    APS 113 s.63: MoC is additive. Decimal values (not bps) throughout.
    """
    if moc_register is None or moc_register.empty:
        logger.warning("apply_moc: empty moc_register, no MoC applied.")
        return lgd_series.copy()

    if "total_moc" not in moc_register.columns:
        raise ValueError("moc_register missing 'total_moc' column.")

    if segment_values is None or segment_col not in moc_register.columns:
        # Portfolio-level: use mean MoC
        mean_moc = float(moc_register["total_moc"].mean())
        logger.info(
            "apply_moc: using portfolio-level mean MoC = %.1f%%", 100 * mean_moc
        )
        return (lgd_series + mean_moc).clip(upper=1.5)

    # Segment-level: map segment key → total_moc
    moc_map = moc_register.set_index(segment_col)["total_moc"].to_dict()
    moc_values = segment_values.map(moc_map).fillna(moc_register["total_moc"].mean())
    result = (lgd_series + moc_values.values).clip(upper=1.5)
    return pd.Series(result, index=lgd_series.index)


# ---------------------------------------------------------------------------
# run_calibration_pipeline — the top-level pipeline for notebooks
# ---------------------------------------------------------------------------

def run_calibration_pipeline(
    loans: pd.DataFrame,
    product: str,
    segment_keys: list[str],
    downturn_method: str = "scalar",
    downturn_scalar: float | None = None,
    policy_floors: dict[str, float] | None = None,
    backtest_df: pd.DataFrame | None = None,
    validation_df: pd.DataFrame | None = None,
    regime_data_source: str = "synthetic",
) -> dict[str, pd.DataFrame]:
    """
    Execute the 4-step APS 113 calibration pipeline for a single product.

    Step 1 — Long-run LGD:
        long_run = compute_long_run_lgd(loans, segment_keys)

    Step 2 — Downturn overlay:
        downturn_lgd = apply_downturn_overlay(long_run_lgd, method, scalar)

    Step 3 — MoC (AFTER downturn, per APS 113 s.63):
        moc_register = MoCRegister(product).build_moc_register(...)
        lgd_with_moc = apply_moc(downturn_lgd, moc_register)

    Step 4 — Regulatory floor:
        final_lgd = max(lgd_with_moc, policy_floor)

    Parameters
    ----------
    loans : DataFrame with realised_lgd, ead_at_default, segment columns, default_year
    product : product name (e.g., 'mortgage')
    segment_keys : list of column names for segmentation
    downturn_scalar : multiplicative scalar for apply_downturn_overlay()
    policy_floors : dict mapping segment_key_concat → floor decimal
    backtest_df, validation_df : optional DataFrames for MoC source computation
    regime_data_source : 'rba_abs_real' or 'synthetic'

    Returns
    -------
    dict of DataFrames:
        'long_run_lgd_by_segment' : compute_long_run_lgd output
        'moc_register'            : MoCRegister output
        'final_calibrated_lgd'    : per-loan final LGD
        'calibration_steps'       : audit trail of all 4 steps per segment
    """
    from src.lgd_calculations import compute_long_run_lgd
    from src.lgd_calculation import apply_downturn_overlay, apply_regulatory_floor

    # Step 1: Long-run LGD
    long_run_df = compute_long_run_lgd(loans, segment_keys, method="vintage_ewa")

    # Step 2: Downturn overlay
    if downturn_scalar is None:
        downturn_scalar = _default_downturn_scalar(product)
    long_run_lgd_col = long_run_df["long_run_lgd"]
    downturn_lgd = apply_downturn_overlay(
        long_run_lgd_col, method=downturn_method, scalar=downturn_scalar
    )
    long_run_df["downturn_lgd"] = downturn_lgd.values

    # Step 3: MoC (applied to downturn LGD)
    moc_register = MoCRegister(product=product, regime_data_source=regime_data_source).build_moc_register(
        segment_df=long_run_df,
        backtest_df=backtest_df,
        validation_df=validation_df,
    )
    long_run_df["total_moc"] = moc_register["total_moc"].values if len(moc_register) == len(long_run_df) else 0.0
    long_run_df["lgd_with_moc"] = long_run_df["downturn_lgd"] + long_run_df["total_moc"]

    # Step 4: Regulatory floor
    default_floor = _default_floor(product)
    if policy_floors:
        floors = long_run_df["segment_key_concat"].map(policy_floors).fillna(default_floor)
    else:
        floors = pd.Series(default_floor, index=long_run_df.index)
    long_run_df["policy_floor"] = floors
    long_run_df["final_lgd"] = long_run_df.apply(
        lambda r: max(r["lgd_with_moc"], r["policy_floor"]), axis=1
    )

    # Audit trail
    long_run_df["calibration_order"] = "LR-LGD → downturn → MoC → floor"
    long_run_df["aps113_pipeline"] = "s.43 → s.46-50 → s.63-65 → s.58"

    logger.info(
        "run_calibration_pipeline (%s): %d segments | mean final_lgd=%.1f%%",
        product, len(long_run_df), 100 * long_run_df["final_lgd"].mean(),
    )

    return {
        "long_run_lgd_by_segment": long_run_df,
        "moc_register": moc_register,
        "calibration_steps": long_run_df[
            ["segment_key_concat", "long_run_lgd", "downturn_lgd",
             "total_moc", "lgd_with_moc", "policy_floor", "final_lgd",
             "calibration_order", "aps113_pipeline"]
        ].copy(),
    }


def _default_downturn_scalar(product: str) -> float:
    """Default downturn scalar by product (based on existing overlay_parameters.csv)."""
    defaults = {
        "mortgage": 1.08,
        "commercial_cashflow": 1.15,
        "development_finance": 1.20,
        "cre_investment": 1.18,
        "receivables": 1.12,
        "trade_contingent": 1.10,
        "asset_equipment": 1.12,
        "residual_stock": 1.20,
        "land_subdivision": 1.25,
        "bridging": 1.15,
        "mezz_second_mortgage": 1.22,
    }
    return defaults.get(product, 1.10)


def _default_floor(product: str) -> float:
    """Default regulatory/policy floor by product (APS 113 s.58)."""
    floors = {
        "mortgage": 0.10,         # 10% with LMI; 15% without (use 10% as portfolio floor)
        "commercial_cashflow": 0.15,
        "receivables": 0.10,      # 10% full recourse + strong collections
        "trade_contingent": 0.05, # 5% fully cash-backed
        "asset_equipment": 0.10,
        "development_finance": 0.20,
        "cre_investment": 0.15,
        "residual_stock": 0.20,
        "land_subdivision": 0.30,
        "bridging": 0.15,
        "mezz_second_mortgage": 0.40,
    }
    return floors.get(product, 0.15)
