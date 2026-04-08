"""
Industry risk integration for APRA LGD model.

Loads industry analysis outputs (risk scores, benchmarks, stress scenarios,
working capital metrics) and provides functions to compute industry-adjusted
LGD parameters including downturn scalars, margins of conservatism, and
recovery haircuts.

Source: Industry Risk Analysis project (ANZSIC-aligned, ABS/RBA data-driven).
"""
import os
import numpy as np
import pandas as pd


# ==========================================================================
# INDUSTRY NAME MAPPING
# ==========================================================================

# Map LGD model short names -> industry analysis ANZSIC names
INDUSTRY_NAME_MAP = {
    "Agriculture": "Agriculture, Forestry And Fishing",
    "Manufacturing": "Manufacturing",
    "Retail Trade": "Retail Trade",
    "Construction": "Construction",
    "Transport": "Transport, Postal And Warehousing",
    "Professional Services": "Professional, Scientific And Technical Services",
    "Accommodation & Food": "Accommodation And Food Services",
    "Health Care": "Health Care and Social Assistance",
    "Wholesale Trade": "Wholesale Trade",
    # Mining has no match in industry analysis -- conservative default applied
    "Mining": None,
}

# Reverse map: ANZSIC name -> LGD model short name
ANZSIC_TO_SHORT = {v: k for k, v in INDUSTRY_NAME_MAP.items() if v is not None}

# Conservative defaults for unmapped industries (e.g. Mining)
UNMAPPED_DEFAULTS = {
    "industry_base_risk_score": 3.5,
    "industry_base_risk_level": "Elevated",
    "classification_risk_score": 3.75,
    "macro_risk_score": 3.2,
    "working_capital_lgd_overlay_score": 3.0,
    "working_capital_lgd_overlay_band": "Elevated",
    "debt_to_ebitda_benchmark": 3.5,
    "icr_benchmark": 2.9,
    "esg_sensitive_sector": True,
}


# ==========================================================================
# INDUSTRY RISK LOADER
# ==========================================================================

class IndustryRiskLoader:
    """
    Load and cache industry analysis CSV outputs.

    All methods normalise industry names to the LGD model's short-name
    convention via INDUSTRY_NAME_MAP, and fill unmapped industries with
    conservative defaults.
    """

    def __init__(self, base_path):
        """
        Parameters
        ----------
        base_path : str
            Path to the directory containing industry analysis CSV outputs.
            Typically: <industry_analysis_project>/output/tables/
        """
        self.base_path = base_path
        self._cache = {}

    def _load_csv(self, filename):
        """Load a CSV with caching."""
        if filename not in self._cache:
            path = os.path.join(self.base_path, filename)
            self._cache[filename] = pd.read_csv(path)
        return self._cache[filename].copy()

    def _normalise_industry_col(self, df, col="industry"):
        """Replace ANZSIC names with LGD model short names."""
        df[col] = df[col].map(ANZSIC_TO_SHORT).fillna(df[col])
        return df

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def load_base_risk_scorecard(self):
        """
        Load industry base risk scorecard.

        Returns DataFrame with columns:
            industry, classification_risk_score, macro_risk_score,
            industry_base_risk_score, industry_base_risk_level
        """
        df = self._load_csv("industry_base_risk_scorecard.csv")
        df = self._normalise_industry_col(df)
        keep = [
            "industry", "classification_risk_score", "macro_risk_score",
            "industry_base_risk_score", "industry_base_risk_level",
        ]
        return df[[c for c in keep if c in df.columns]]

    def load_benchmarks(self):
        """
        Load industry-generated financial benchmarks.

        Returns DataFrame with columns:
            industry, debt_to_ebitda_benchmark, icr_benchmark,
            ar_days_benchmark, ap_days_benchmark, inventory_days_benchmark,
            ebitda_margin_pct_latest
        """
        df = self._load_csv("industry_generated_benchmarks.csv")
        df = self._normalise_industry_col(df)
        keep = [
            "industry", "ebitda_margin_pct_latest",
            "debt_to_ebitda_benchmark", "icr_benchmark",
            "ar_days_benchmark", "ar_days_stress_benchmark",
            "ap_days_benchmark", "ap_days_stress_benchmark",
            "inventory_days_benchmark",
        ]
        return df[[c for c in keep if c in df.columns]]

    def load_working_capital_metrics(self):
        """
        Load industry working capital risk metrics.

        Returns DataFrame with LGD-relevant overlay scores.
        """
        df = self._load_csv("industry_working_capital_risk_metrics.csv")
        df = self._normalise_industry_col(df)
        keep = [
            "industry",
            "cash_conversion_cycle_benchmark_days",
            "cash_conversion_cycle_stress_days",
            "working_capital_lgd_overlay_score",
            "working_capital_lgd_overlay_band",
            "lgd_primary_driver",
        ]
        return df[[c for c in keep if c in df.columns]]

    def load_stress_matrix(self):
        """
        Load industry stress test matrix.

        Returns DataFrame with scenario-level stress deltas.
        """
        df = self._load_csv("industry_stress_test_matrix.csv")
        df = self._normalise_industry_col(df)
        return df

    def load_credit_appetite(self):
        """
        Load industry credit appetite strategy.

        Returns DataFrame with lending parameters by sector.
        """
        df = self._load_csv("industry_credit_appetite_strategy.csv")
        df = self._normalise_industry_col(df)
        return df

    def load_esg_overlay(self):
        """
        Load industry ESG sensitivity overlay.

        Returns DataFrame with ESG risk flags.
        """
        df = self._load_csv("industry_esg_sensitivity_overlay.csv")
        df = self._normalise_industry_col(df)
        return df

    def load_concentration_limits(self):
        """
        Load portfolio concentration limits by industry.

        Returns DataFrame with exposure caps and breach flags.
        """
        df = self._load_csv("concentration_limits.csv")
        df = self._normalise_industry_col(df)
        return df

    def get_risk_score_lookup(self):
        """
        Build a dict: industry_short_name -> industry_base_risk_score.

        Includes conservative defaults for unmapped industries.
        """
        sc = self.load_base_risk_scorecard()
        lookup = dict(zip(sc["industry"], sc["industry_base_risk_score"]))
        # Fill unmapped
        for short_name, anzsic_name in INDUSTRY_NAME_MAP.items():
            if anzsic_name is None and short_name not in lookup:
                lookup[short_name] = UNMAPPED_DEFAULTS["industry_base_risk_score"]
        return lookup

    def get_wc_lgd_overlay_lookup(self):
        """
        Build a dict: industry_short_name -> working_capital_lgd_overlay_score.
        """
        wc = self.load_working_capital_metrics()
        lookup = dict(zip(wc["industry"], wc["working_capital_lgd_overlay_score"]))
        for short_name, anzsic_name in INDUSTRY_NAME_MAP.items():
            if anzsic_name is None and short_name not in lookup:
                lookup[short_name] = UNMAPPED_DEFAULTS[
                    "working_capital_lgd_overlay_score"
                ]
        return lookup


# ==========================================================================
# INDUSTRY-ADJUSTED LGD COMPUTATION FUNCTIONS
# ==========================================================================

def compute_industry_downturn_scalar(risk_score, base_scalar, alpha=0.15):
    """
    Compute industry-adjusted downturn scalar.

    Formula:
        adjusted = base_scalar * (1 + alpha * (risk_score - 2.5) / 2.0)

    This keeps adjustments within +/- 7-10% of the base scalar, which is
    defensible under APRA APS 113 (reflects differential industry cyclicality).

    Parameters
    ----------
    risk_score : float or array-like
        Industry base risk score (1-5 scale).
    base_scalar : float or array-like
        Base downturn scalar from security type or completion stage.
    alpha : float
        Sensitivity parameter (default 0.15 = 15% max swing).

    Returns
    -------
    float or array-like
        Industry-adjusted downturn scalar.

    Examples
    --------
    >>> compute_industry_downturn_scalar(3.5, 1.20)  # Elevated
    1.29
    >>> compute_industry_downturn_scalar(2.14, 1.20)  # Low-Medium
    1.1731...
    """
    risk_score = np.asarray(risk_score, dtype=float)
    base_scalar = np.asarray(base_scalar, dtype=float)
    return base_scalar * (1 + alpha * (risk_score - 2.5) / 2.0)


def compute_industry_moc_adjustment(risk_score, base_moc, beta=0.20):
    """
    Compute industry-adjusted margin of conservatism.

    Formula:
        adjusted = base_moc * (1 + beta * (risk_score - 2.5) / 2.5)

    Higher-risk industries get larger MoC (reflecting greater data uncertainty
    and model risk in those sectors).

    Parameters
    ----------
    risk_score : float or array-like
        Industry base risk score (1-5 scale).
    base_moc : float
        Base margin of conservatism (e.g. 0.03 = 3pp).
    beta : float
        Sensitivity parameter (default 0.20 = 20% max swing).

    Returns
    -------
    float or array-like
        Industry-adjusted MoC.
    """
    risk_score = np.asarray(risk_score, dtype=float)
    return base_moc * (1 + beta * (risk_score - 2.5) / 2.5)


def compute_industry_recovery_haircut(risk_score):
    """
    Compute additional recovery haircut based on industry risk.

    Higher-risk industries experience deeper asset value declines during
    distressed workout periods (e.g. fire-sale discounts on specialised
    equipment, inventory obsolescence).

    Formula:
        haircut = 0.02 * max(risk_score - 2.0, 0)

    Returns 0% for score <= 2.0, up to 6% for score = 5.0.
    Applied multiplicatively to recovery PV.

    Parameters
    ----------
    risk_score : float or array-like

    Returns
    -------
    float or array-like
        Recovery haircut (0.0 to 0.06).
    """
    risk_score = np.asarray(risk_score, dtype=float)
    return 0.02 * np.maximum(risk_score - 2.0, 0)


def compute_working_capital_lgd_adjustment(wc_lgd_overlay_score):
    """
    Compute additive LGD adjustment from working capital risk.

    Industries with poor working capital profiles (high CCC, slow collections,
    illiquid inventory) face higher LGD because administration recoveries
    are lower when working capital is under stress.

    Formula:
        adjustment = 0.01 * max(wc_overlay_score - 2.0, 0)

    Returns 0 to ~1.9 percentage points additive LGD adjustment.

    Parameters
    ----------
    wc_lgd_overlay_score : float or array-like
        Working capital LGD overlay score (typically 1.89 to 3.89).

    Returns
    -------
    float or array-like
        Additive LGD adjustment.
    """
    wc_lgd_overlay_score = np.asarray(wc_lgd_overlay_score, dtype=float)
    return 0.01 * np.maximum(wc_lgd_overlay_score - 2.0, 0)


# ==========================================================================
# DEVELOPMENT TYPE -> INDUSTRY MAPPING
# ==========================================================================

DEVELOPMENT_TYPE_TO_INDUSTRY = {
    "Residential Apartments": "Construction",
    "Residential Houses/Lots": "Construction",
    "Commercial Office": "Professional Services",
    "Mixed-Use": "Construction",
    "Industrial": "Manufacturing",
}


def map_development_type_to_industry(development_type):
    """
    Map development project types to the most relevant industry sector.

    Residential and mixed-use projects map to Construction (builder/developer
    risk). Commercial office maps to Professional Services (tenant demand).
    Industrial maps to Manufacturing (end-user demand).

    Parameters
    ----------
    development_type : str or Series

    Returns
    -------
    str or Series
    """
    if isinstance(development_type, pd.Series):
        return development_type.map(DEVELOPMENT_TYPE_TO_INDUSTRY).fillna(
            "Construction"
        )
    return DEVELOPMENT_TYPE_TO_INDUSTRY.get(development_type, "Construction")


# ==========================================================================
# MASTER ENRICHMENT FUNCTION
# ==========================================================================

def enrich_loans_with_industry_risk(loans_df, loader, product_type="commercial"):
    """
    Enrich a loans DataFrame with industry risk features.

    Joins industry risk scores, benchmarks, and working capital overlays onto
    each loan based on its industry column. For development loans, first maps
    development_type to an industry.

    Parameters
    ----------
    loans_df : DataFrame
        Loan-level data. Must have 'industry' column (commercial) or
        'development_type' column (development).
    loader : IndustryRiskLoader
        Initialised loader pointing to industry analysis outputs.
    product_type : str
        'commercial' or 'development'.

    Returns
    -------
    DataFrame
        Input DataFrame with additional industry risk columns.
    """
    out = loans_df.copy()

    # For development loans, derive industry from development type
    if product_type == "development" and "industry" not in out.columns:
        out["industry"] = map_development_type_to_industry(out["development_type"])

    if "industry" not in out.columns:
        return out

    # Load lookups
    risk_lookup = loader.get_risk_score_lookup()
    wc_lookup = loader.get_wc_lgd_overlay_lookup()

    # Load full scorecard for risk level
    scorecard = loader.load_base_risk_scorecard()
    level_lookup = dict(zip(scorecard["industry"], scorecard["industry_base_risk_level"]))
    for short_name, anzsic_name in INDUSTRY_NAME_MAP.items():
        if anzsic_name is None and short_name not in level_lookup:
            level_lookup[short_name] = UNMAPPED_DEFAULTS["industry_base_risk_level"]

    # Load benchmarks
    benchmarks = loader.load_benchmarks()
    debt_ebitda_lookup = dict(zip(benchmarks["industry"], benchmarks["debt_to_ebitda_benchmark"]))
    icr_lookup = dict(zip(benchmarks["industry"], benchmarks["icr_benchmark"]))

    # Load ESG
    esg = loader.load_esg_overlay()
    esg_lookup = dict(zip(esg["industry"], esg["esg_sensitive_sector"]))

    # Map onto loans
    out["industry_risk_score"] = out["industry"].map(risk_lookup).fillna(
        UNMAPPED_DEFAULTS["industry_base_risk_score"]
    )
    out["industry_risk_level"] = out["industry"].map(level_lookup).fillna(
        UNMAPPED_DEFAULTS["industry_base_risk_level"]
    )
    out["wc_lgd_overlay_score"] = out["industry"].map(wc_lookup).fillna(
        UNMAPPED_DEFAULTS["working_capital_lgd_overlay_score"]
    )
    out["industry_debt_to_ebitda_benchmark"] = out["industry"].map(
        debt_ebitda_lookup
    ).fillna(UNMAPPED_DEFAULTS["debt_to_ebitda_benchmark"])
    out["industry_icr_benchmark"] = out["industry"].map(icr_lookup).fillna(
        UNMAPPED_DEFAULTS["icr_benchmark"]
    )
    esg_mapped = out["industry"].map(esg_lookup)
    out["industry_esg_sensitive"] = esg_mapped.where(
        esg_mapped.notna(), UNMAPPED_DEFAULTS["esg_sensitive_sector"]
    ).astype(bool)

    # Derived LGD adjustment columns
    out["industry_recovery_haircut"] = compute_industry_recovery_haircut(
        out["industry_risk_score"]
    )
    out["wc_lgd_adjustment"] = compute_working_capital_lgd_adjustment(
        out["wc_lgd_overlay_score"]
    )

    return out
