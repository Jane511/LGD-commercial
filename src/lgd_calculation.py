"""
LGD calculation engines for Australian bank lending products.

Provides product-specific LGD computation, segmentation, regulatory overlays,
and the full pipeline from realised LGD through to final APRA-compliant estimates.

Products:
  1. Residential Mortgage  (two-stage cure model + APRA overlays)
  2. Commercial Cash Flow   (security-type segmentation + APS 112)
  3. Development Finance    (completion-stage scenario model)

Common pipeline:
  Realised LGD -> Exposure-weighted long-run -> Downturn adjustment -> MoC -> Regulatory overlays
"""
import numpy as np
import pandas as pd
from scipy.special import expit  # logistic function

from .industry_risk_integration import (
    compute_industry_downturn_scalar,
    compute_industry_moc_adjustment,
    compute_industry_recovery_haircut,
    compute_working_capital_lgd_adjustment,
)


# ==========================================================================
# COMMON UTILITIES
# ==========================================================================

def exposure_weighted_average(df, lgd_col, ead_col="ead", group_col=None):
    """
    Compute exposure-weighted average LGD.

    Formula: LR_LGD_s = Sum(LGD_i * EAD_i) / Sum(EAD_i)

    Parameters
    ----------
    df : DataFrame with LGD and EAD columns
    lgd_col : column name for LGD values
    ead_col : column name for EAD values
    group_col : optional grouping column(s) for segment-level averages

    Returns
    -------
    DataFrame with exposure-weighted LGD per segment, or scalar if no grouping
    """
    if group_col is None:
        total_ead = df[ead_col].sum()
        if total_ead == 0:
            return 0.0
        return (df[lgd_col] * df[ead_col]).sum() / total_ead

    def _ewa(g):
        total = g[ead_col].sum()
        if total == 0:
            return 0.0
        return (g[lgd_col] * g[ead_col]).sum() / total

    if isinstance(group_col, str):
        group_col = [group_col]

    result = (
        df.groupby(group_col, observed=True)
        .apply(_ewa, include_groups=False)
        .reset_index()
    )
    result.columns = list(group_col) + ["lgd_long_run"]
    counts = df.groupby(group_col, observed=True).size().reset_index(name="count")
    eads = df.groupby(group_col, observed=True)[ead_col].sum().reset_index(name="total_ead")
    result = result.merge(counts, on=group_col).merge(eads, on=group_col)
    return result


def apply_downturn_overlay(lgd_series, method="scalar", scalar=1.10, addon=0.0):
    """
    Apply downturn adjustment to long-run LGD.

    Methods:
      - 'scalar':  Downturn_LGD = LR_LGD * scalar
      - 'additive': Downturn_LGD = LR_LGD + addon
      - 'combined': Downturn_LGD = LR_LGD * scalar + addon
    """
    if method == "scalar":
        return lgd_series * scalar
    elif method == "additive":
        return lgd_series + addon
    elif method == "combined":
        return lgd_series * scalar + addon
    raise ValueError(f"Unknown downturn method: {method}")


def add_margin_of_conservatism(lgd_series, margin):
    """Add margin of conservatism (additive, in decimal form e.g. 0.02 = 2pp)."""
    return lgd_series + margin


# ==========================================================================
# 1. RESIDENTIAL MORTGAGE LGD ENGINE
# ==========================================================================

class MortgageLGDEngine:
    """
    Australian residential mortgage LGD engine.

    Implements:
    - Two-stage model: P(Cure) then E[LGD|Loss]
    - Exposure-weighted long-run LGD by segment
    - Downturn overlay (scalar or macro-linked)
    - Margin of conservatism
    - APRA overlays: LMI recognition, 10%/15% floor, standard/non-standard
    """

    # APRA regulatory parameters
    LGD_FLOOR_STANDARD = 0.10
    LGD_FLOOR_NON_STANDARD = 0.15
    LMI_BENEFIT_FACTOR = 0.20  # 20% LGD reduction for eligible LMI
    APRA_SCALAR = 1.10  # applied to RWA, not LGD

    # Default model parameters
    DEFAULT_DOWNTURN_SCALAR = 1.15
    DEFAULT_MOC = 0.02  # 2 percentage points

    def __init__(self, downturn_scalar=None, moc=None):
        self.downturn_scalar = downturn_scalar or self.DEFAULT_DOWNTURN_SCALAR
        self.moc = moc or self.DEFAULT_MOC

    @staticmethod
    def segment_loans(df):
        """
        Create segmentation columns for mortgage portfolio.

        Segments:
          Level 1: mortgage_class (Standard / Non-Standard)
          Level 2: ltv_band (LTV at default)
          Level 3: lmi_eligible
        """
        out = df.copy()

        # LTV bands at default
        bins = [0, 0.60, 0.70, 0.80, 0.90, float("inf")]
        labels = ["<60%", "60-70%", "70-80%", "80-90%", "90%+"]
        out["ltv_band"] = pd.cut(
            out["ltv_at_default"], bins=bins, labels=labels, right=True
        )

        # Credit score bands
        score_bins = [0, 550, 620, 680, 740, 900]
        score_labels = ["<550", "550-620", "620-680", "680-740", "740+"]
        out["score_band"] = pd.cut(
            out["credit_score"], bins=score_bins, labels=score_labels, right=True
        )

        return out

    def compute_long_run_lgd(self, df, segments=None):
        """
        Compute exposure-weighted long-run LGD by segment.

        Default segments: mortgage_class x ltv_band
        """
        if segments is None:
            segments = ["mortgage_class", "ltv_band"]
        segmented = self.segment_loans(df)
        return exposure_weighted_average(
            segmented, lgd_col="realised_lgd", ead_col="ead", group_col=segments
        )

    def apply_apra_overlays(self, df):
        """
        Apply full APRA overlay chain to loan-level LGD estimates.

        Pipeline:
          1. Start with realised_lgd (or model-predicted LGD)
          2. Apply downturn scalar
          3. Add margin of conservatism
          4. Apply LMI benefit (for eligible loans)
          5. Apply LGD floor (10% standard, 15% non-standard)

        Returns DataFrame with intermediate and final columns.
        """
        out = df.copy()

        # Step 1: Downturn
        out["lgd_downturn"] = apply_downturn_overlay(
            out["realised_lgd"], method="scalar", scalar=self.downturn_scalar
        )

        # Step 2: MoC
        out["lgd_with_moc"] = add_margin_of_conservatism(
            out["lgd_downturn"], self.moc
        )

        # Step 3: LMI adjustment
        out["lgd_after_lmi"] = np.where(
            out["lmi_eligible"] == 1,
            out["lgd_with_moc"] * (1 - self.LMI_BENEFIT_FACTOR),
            out["lgd_with_moc"],
        )

        # Step 4: Floor
        floor = np.where(
            out["mortgage_class"] == "Standard",
            self.LGD_FLOOR_STANDARD,
            self.LGD_FLOOR_NON_STANDARD,
        )
        out["lgd_final"] = np.maximum(out["lgd_after_lmi"], floor)

        # Cap at 100%
        out["lgd_final"] = out["lgd_final"].clip(0, 1)

        return out

    def compute_illustrative_rwa(self, df, pd_estimate=0.02, maturity=5):
        """
        Compute illustrative RWA using simplified IRB risk-weight function.

        K = LGD * [N(sqrt(1/(1-R)) * G(PD) + sqrt(R/(1-R)) * G(0.999)) - PD]
            * (1 + (M - 2.5) * b) / (1 - 1.5 * b)
        RWA = K * 12.5 * EAD
        RWA_APRA = RWA * 1.10
        """
        from scipy.stats import norm

        lgd = df["lgd_final"].values
        ead = df["ead"].values
        pd_val = pd_estimate

        # Residential mortgage correlation (Basel)
        R = 0.15

        # Maturity adjustment
        b = (0.11852 - 0.05478 * np.log(pd_val)) ** 2
        maturity_adj = (1 + (maturity - 2.5) * b) / (1 - 1.5 * b)

        # Capital requirement
        K = lgd * (
            norm.cdf(
                np.sqrt(1 / (1 - R)) * norm.ppf(pd_val)
                + np.sqrt(R / (1 - R)) * norm.ppf(0.999)
            )
            - pd_val
        ) * maturity_adj

        rwa = K * 12.5 * ead
        rwa_apra = rwa * self.APRA_SCALAR

        out = df.copy()
        out["capital_K"] = np.round(K, 6)
        out["rwa"] = np.round(rwa, 2)
        out["rwa_after_apra_scalar"] = np.round(rwa_apra, 2)
        return out


# ==========================================================================
# 2. COMMERCIAL CASH FLOW LGD ENGINE
# ==========================================================================

class CommercialLGDEngine:
    """
    Australian commercial cash-flow lending LGD engine.

    Implements:
    - Security-type segmentation (Property, PPSR, GSR)
    - Coverage-ratio-based analysis
    - Industry segmentation
    - Exposure-weighted long-run LGD
    - Downturn overlay by security type
    - Margin of conservatism
    - APS 112 supervisory LGD fallback
    """

    # Supervisory LGD (APS 112 / standardised approach fallback)
    SUPERVISORY_LGD = {
        "Senior Secured": 0.35,
        "Senior Unsecured": 0.45,
        "Subordinated": 0.75,
    }

    # Downturn scalars by security type
    DOWNTURN_SCALARS = {
        "Property": 1.15,
        "PPSR - P&E": 1.20,
        "PPSR - Receivables": 1.20,
        "PPSR - Mixed": 1.20,
        "GSR Only": 1.15,
    }

    DEFAULT_MOC = 0.03  # 3 percentage points

    def __init__(self, moc=None):
        self.moc = moc or self.DEFAULT_MOC

    @staticmethod
    def segment_loans(df):
        """
        Create segmentation columns for commercial portfolio.

        Segments:
          Level 1: security_type
          Level 2: coverage_band
          Level 3: industry_risk_band (if industry_risk_score present)
          Level 4: borrower_size
        """
        out = df.copy()

        # Security coverage bands
        bins = [0, 0.50, 0.80, 1.00, 1.20, float("inf")]
        labels = ["<50%", "50-80%", "80-100%", "100-120%", "120%+"]
        out["coverage_band"] = pd.cut(
            out["security_coverage_ratio"], bins=bins, labels=labels, right=True
        )

        # Borrower size
        rev_bins = [0, 5e6, 20e6, 75e6, float("inf")]
        rev_labels = ["Micro (<$5M)", "Small ($5-20M)", "Mid ($20-75M)", "Large (>$75M)"]
        out["borrower_size"] = pd.cut(
            out["annual_revenue"], bins=rev_bins, labels=rev_labels, right=True
        )

        # Industry risk band (from industry analysis integration)
        if "industry_risk_score" in out.columns:
            risk_bins = [0, 2.5, 3.0, 5.0]
            risk_labels = ["Low", "Medium", "Elevated"]
            out["industry_risk_band"] = pd.cut(
                out["industry_risk_score"], bins=risk_bins,
                labels=risk_labels, right=True
            )

        return out

    def compute_long_run_lgd(self, df, segments=None):
        """Compute exposure-weighted long-run LGD by segment."""
        if segments is None:
            segments = ["security_type", "coverage_band"]
        segmented = self.segment_loans(df)
        return exposure_weighted_average(
            segmented, lgd_col="realised_lgd", ead_col="ead", group_col=segments
        )

    def apply_overlays(self, df):
        """
        Apply downturn, MoC, and regulatory overlays with industry risk adjustments.

        Enhanced pipeline:
          1. Apply industry recovery haircut to realised LGD (if available)
          2. Map base downturn scalar by security type, then adjust for industry risk
          3. Apply industry-adjusted MoC
          4. Add working capital LGD adjustment (if available)
          5. Floor to supervisory LGD by seniority
          6. Cap at 100%
        """
        out = df.copy()
        has_industry = "industry_risk_score" in out.columns

        # Step 1: Industry recovery haircut (raises effective LGD pre-overlays)
        if has_industry:
            out["industry_recovery_haircut"] = compute_industry_recovery_haircut(
                out["industry_risk_score"]
            )
            out["lgd_industry_adjusted"] = (
                out["realised_lgd"] + out["industry_recovery_haircut"]
            )
        else:
            out["lgd_industry_adjusted"] = out["realised_lgd"]

        # Step 2: Downturn scalar (base from security type, adjusted by industry)
        base_scalar = out["security_type"].map(self.DOWNTURN_SCALARS).fillna(1.15)
        if has_industry:
            out["downturn_scalar"] = compute_industry_downturn_scalar(
                out["industry_risk_score"], base_scalar
            )
        else:
            out["downturn_scalar"] = base_scalar
        out["lgd_downturn"] = out["lgd_industry_adjusted"] * out["downturn_scalar"]

        # Step 3: MoC (adjusted by industry risk)
        if has_industry:
            out["industry_moc"] = compute_industry_moc_adjustment(
                out["industry_risk_score"], self.moc
            )
        else:
            out["industry_moc"] = self.moc
        out["lgd_with_moc"] = out["lgd_downturn"] + out["industry_moc"]

        # Step 4: Working capital overlay (additive)
        if "wc_lgd_overlay_score" in out.columns:
            out["wc_lgd_adjustment"] = compute_working_capital_lgd_adjustment(
                out["wc_lgd_overlay_score"]
            )
            out["lgd_with_moc"] = out["lgd_with_moc"] + out["wc_lgd_adjustment"]

        # Step 5: Supervisory LGD floor by seniority
        out["supervisory_lgd"] = out["seniority"].map(self.SUPERVISORY_LGD).fillna(0.45)

        # Step 6: Final -- cap at 100%
        out["lgd_final"] = out["lgd_with_moc"].clip(0, 1)

        return out

    @staticmethod
    def sme_firm_size_adjustment(revenue_millions):
        """
        Basel/APRA SME firm-size adjustment to correlation parameter.

        Adjustment = -0.04 * (1 - (S - 5) / 45)
        where S = annual revenue in AUD millions, capped at 5-50.
        """
        s = np.clip(revenue_millions, 5, 50)
        return -0.04 * (1 - (s - 5) / 45)


# ==========================================================================
# 3. DEVELOPMENT FINANCE LGD ENGINE
# ==========================================================================

class DevelopmentLGDEngine:
    """
    Australian development finance LGD engine.

    Implements:
    - Completion-stage segmentation (primary LGD driver)
    - Fund-to-complete scenario analysis
    - Pre-sale rescission risk
    - Downturn overlay by completion stage
    - Margin of conservatism (highest of all products)
    - Specialised lending / slotting framework
    """

    # Downturn scalars by completion stage
    DOWNTURN_SCALARS = {
        "Pre-Construction": 1.20,
        "Early Construction": 1.25,
        "Mid-Construction": 1.30,
        "Near-Complete": 1.15,
        "Complete Unsold": 1.20,
    }

    DEFAULT_MOC = 0.05  # 5 percentage points (highest due to data limitations)

    # APRA slotting categories (supervisory risk weights)
    SLOTTING_RW = {
        "Strong": 0.70,
        "Good": 0.90,
        "Satisfactory": 1.15,
        "Weak": 2.50,
        "Default": 0.0,  # deducted from capital
    }

    # HVCRE higher correlation multiplier
    HVCRE_CORRELATION_MULTIPLIER = 1.25

    def __init__(self, moc=None):
        self.moc = moc or self.DEFAULT_MOC

    @staticmethod
    def segment_loans(df):
        """
        Create segmentation columns for development portfolio.

        Segments:
          Level 1: completion_stage (primary driver)
          Level 2: development_type
          Level 3: presale_band
          Level 4: lvr_band
          Level 5: industry_risk_band (if available)
        """
        out = df.copy()

        # Pre-sale coverage bands
        ps_bins = [0, 0.30, 0.60, 0.80, 1.0, float("inf")]
        ps_labels = ["<30%", "30-60%", "60-80%", "80-100%", "100%+"]
        out["presale_band"] = pd.cut(
            out["presale_coverage"], bins=ps_bins, labels=ps_labels, right=True
        )

        # LVR bands (as-if-complete)
        lvr_bins = [0, 0.55, 0.65, 0.75, float("inf")]
        lvr_labels = ["<55%", "55-65%", "65-75%", "75%+"]
        out["lvr_band"] = pd.cut(
            out["lvr_as_if_complete"], bins=lvr_bins, labels=lvr_labels, right=True
        )

        # Industry risk band (from development type -> industry mapping)
        if "industry_risk_score" in out.columns:
            risk_bins = [0, 2.5, 3.0, 5.0]
            risk_labels = ["Low", "Medium", "Elevated"]
            out["industry_risk_band"] = pd.cut(
                out["industry_risk_score"], bins=risk_bins,
                labels=risk_labels, right=True
            )

        return out

    def compute_long_run_lgd(self, df, segments=None):
        """Compute exposure-weighted long-run LGD by segment."""
        if segments is None:
            segments = ["completion_stage"]
        segmented = self.segment_loans(df)
        return exposure_weighted_average(
            segmented, lgd_col="realised_lgd", ead_col="ead", group_col=segments
        )

    def apply_overlays(self, df):
        """
        Apply downturn, MoC, and regulatory overlays with industry risk adjustments.

        Enhanced pipeline:
          1. Apply industry recovery haircut (if available)
          2. Map base downturn scalar by completion stage, adjust for industry risk
          3. Apply industry-adjusted MoC
          4. Cap at 100%
        """
        out = df.copy()
        has_industry = "industry_risk_score" in out.columns

        # Step 1: Industry recovery haircut
        if has_industry:
            out["industry_recovery_haircut"] = compute_industry_recovery_haircut(
                out["industry_risk_score"]
            )
            out["lgd_industry_adjusted"] = (
                out["realised_lgd"] + out["industry_recovery_haircut"]
            )
        else:
            out["lgd_industry_adjusted"] = out["realised_lgd"]

        # Step 2: Downturn scalar (base from completion stage, adjusted by industry)
        base_scalar = out["completion_stage"].map(
            self.DOWNTURN_SCALARS
        ).fillna(1.25)
        if has_industry:
            out["downturn_scalar"] = compute_industry_downturn_scalar(
                out["industry_risk_score"], base_scalar
            )
        else:
            out["downturn_scalar"] = base_scalar
        out["lgd_downturn"] = out["lgd_industry_adjusted"] * out["downturn_scalar"]

        # Step 3: MoC (adjusted by industry risk)
        if has_industry:
            out["industry_moc"] = compute_industry_moc_adjustment(
                out["industry_risk_score"], self.moc
            )
        else:
            out["industry_moc"] = self.moc
        out["lgd_with_moc"] = out["lgd_downturn"] + out["industry_moc"]

        # Step 4: Cap at 100%
        out["lgd_final"] = out["lgd_with_moc"].clip(0, 1)

        return out

    def assign_slotting(self, df):
        """
        Assign APRA slotting categories based on project characteristics.

        Heuristic scoring:
          - Completion stage
          - Pre-sale coverage
          - LVR
          - Sponsor track record proxy (years_in_business not available, use unit_count as scale proxy)

        Returns DataFrame with 'slotting_category' and 'slotting_rw' columns.
        """
        out = df.copy()
        score = np.zeros(len(df))

        # Completion: later is better
        stage_scores = {
            "Pre-Construction": 0,
            "Early Construction": 1,
            "Mid-Construction": 2,
            "Near-Complete": 4,
            "Complete Unsold": 3,
        }
        score += out["completion_stage"].map(stage_scores).fillna(1).values

        # Pre-sales: higher is better
        score += np.where(out["presale_coverage"] > 0.80, 3,
                 np.where(out["presale_coverage"] > 0.60, 2,
                 np.where(out["presale_coverage"] > 0.30, 1, 0)))

        # LVR: lower is better
        score += np.where(out["lvr_as_if_complete"] < 0.55, 3,
                 np.where(out["lvr_as_if_complete"] < 0.65, 2,
                 np.where(out["lvr_as_if_complete"] < 0.75, 1, 0)))

        # Industry risk: lower risk is better (0-2 points)
        if "industry_risk_score" in out.columns:
            score += np.where(out["industry_risk_score"] < 2.5, 2,
                     np.where(out["industry_risk_score"] < 3.0, 1, 0))

        # Map to categories
        out["slotting_score"] = score
        out["slotting_category"] = np.where(
            score >= 9, "Strong",
            np.where(score >= 7, "Good",
            np.where(score >= 4, "Satisfactory",
                     "Weak"))
        )
        out["slotting_rw"] = out["slotting_category"].map(self.SLOTTING_RW)

        return out

    def scenario_analysis(self, df, grv_decline=0.0, cost_overrun=0.0,
                          rescission_rate=0.0, sales_extension_months=0):
        """
        Run scenario analysis on development LGD.

        Adjusts key drivers and recomputes approximate LGD impact:
          - grv_decline: fractional decline in GRV (e.g., -0.20 for 20% drop)
          - cost_overrun: fractional increase in cost-to-complete (e.g., 0.10 for 10%)
          - rescission_rate: fraction of pre-sales that rescind
          - sales_extension_months: additional months to sell (increases holding/discount cost)

        Returns DataFrame with scenario LGD estimates.
        """
        out = df.copy()

        stressed_grv = out["grv"] * (1 + grv_decline)
        stressed_ctc = out["cost_to_complete"] * (1 + cost_overrun)
        stressed_presale = out["presale_value"] * (1 - rescission_rate)

        # Approximate stressed recovery
        recovery_if_ftc = np.where(
            out["fund_to_complete"] == 1,
            stressed_grv * 0.90 - stressed_ctc,
            out["as_is_value"] * (1 + grv_decline) * 0.75,
        )
        recovery_if_ftc = np.maximum(recovery_if_ftc, 0)

        # Additional discount for extended sales period
        extra_discount = (1 + out["discount_rate"]) ** (sales_extension_months / 12)
        recovery_pv = recovery_if_ftc / extra_discount

        # Approximate stressed costs (holding + receiver + legal)
        stressed_costs = out["ead"] * 0.08  # ~8% total costs

        econ_loss = out["ead"] + stressed_costs - recovery_pv
        out["scenario_lgd"] = (econ_loss / out["ead"]).clip(0, 1)

        out["scenario_description"] = (
            f"GRV {grv_decline*100:+.0f}%, CTC {cost_overrun*100:+.0f}%, "
            f"Rescission {rescission_rate*100:.0f}%, +{sales_extension_months}mo sales"
        )

        return out


# ==========================================================================
# FULL PIPELINE: Run all products
# ==========================================================================

def run_full_pipeline(datasets):
    """
    Run the complete LGD pipeline for all three products.

    Parameters
    ----------
    datasets : dict from generate_all_datasets()

    Returns
    -------
    dict with keys 'mortgage', 'commercial', 'development',
    each containing 'loans_with_overlays' and 'segment_summary' DataFrames.
    """
    results = {}

    # --- Mortgage ---
    mtg = MortgageLGDEngine()
    mtg_loans = datasets["mortgage"]["loans"]
    mtg_with_overlays = mtg.apply_apra_overlays(mtg_loans)
    mtg_with_overlays = mtg.compute_illustrative_rwa(mtg_with_overlays)
    mtg_segments = mtg.compute_long_run_lgd(mtg_loans)
    results["mortgage"] = {
        "loans_with_overlays": mtg_with_overlays,
        "segment_summary": mtg_segments,
    }

    # --- Commercial ---
    com = CommercialLGDEngine()
    com_loans = datasets["commercial"]["loans"]
    com_with_overlays = com.apply_overlays(com_loans)
    com_segments = com.compute_long_run_lgd(com_loans)
    results["commercial"] = {
        "loans_with_overlays": com_with_overlays,
        "segment_summary": com_segments,
    }

    # --- Development ---
    dev = DevelopmentLGDEngine()
    dev_loans = datasets["development"]["loans"]
    dev_with_overlays = dev.apply_overlays(dev_loans)
    dev_with_overlays = dev.assign_slotting(dev_with_overlays)
    dev_segments = dev.compute_long_run_lgd(dev_loans)
    results["development"] = {
        "loans_with_overlays": dev_with_overlays,
        "segment_summary": dev_segments,
    }

    return results
