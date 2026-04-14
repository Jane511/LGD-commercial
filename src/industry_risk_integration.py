"""
Industry risk integration for APRA-style LGD model.

This module consumes the compact upstream industry-analysis contract under
`data/exports` and keeps LGD-specific dependencies intentionally narrow.

Canonical upstream files:
- industry_risk_scores.parquet
- macro_regime_flags.parquet
- downturn_overlay_table.parquet
- property_market_overlays.parquet (optional; required for property-backed logic)
"""
from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd


# ===========================================================================
# INDUSTRY NAME MAPPING
# ===========================================================================

# Map LGD model short names -> upstream ANZSIC-like names
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
    # Mining has no mapped upstream industry in this demo project
    "Mining": None,
    "Education & Training": "Education And Training",
    "Financial Services": "Financial And Insurance Services",
    "Information Technology": "Information Media And Telecommunications",
    "Real Estate": "Rental, Hiring And Real Estate Services",
    "Arts & Recreation": "Arts And Recreation Services",
    "Utilities": "Electricity, Gas, Water And Waste Services",
}


def _canonical_industry_name(value) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    txt = str(value).strip().lower()
    txt = txt.replace("&", "and")
    txt = re.sub(r"[^a-z0-9]+", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt or None


_CANONICAL_ANZSIC_TO_SHORT = {
    _canonical_industry_name(v): k for k, v in INDUSTRY_NAME_MAP.items() if v is not None
}

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


def _risk_level_from_score(score: pd.Series | np.ndarray | float) -> pd.Series:
    values = pd.to_numeric(score, errors="coerce")
    bins = [-np.inf, 2.2, 3.0, 3.8, np.inf]
    labels = ["Low", "Medium", "Elevated", "High"]
    out = pd.cut(values, bins=bins, labels=labels)
    return out.astype("object").fillna(UNMAPPED_DEFAULTS["industry_base_risk_level"])


def _wc_band_from_score(score: pd.Series | np.ndarray | float) -> pd.Series:
    values = pd.to_numeric(score, errors="coerce")
    bins = [-np.inf, 2.3, 3.1, 3.9, np.inf]
    labels = ["Low", "Medium", "Elevated", "High"]
    out = pd.cut(values, bins=bins, labels=labels)
    return out.astype("object").fillna(UNMAPPED_DEFAULTS["working_capital_lgd_overlay_band"])


# ===========================================================================
# INDUSTRY RISK LOADER
# ===========================================================================


class IndustryRiskLoader:
    """
    Load and cache compact upstream parquet outputs.

    The loader intentionally keeps only LGD-relevant fields and derives any
    missing broad strategy outputs locally rather than recreating the old
    legacy CSV interface end-to-end.
    """

    DEFAULT_EXPORTS_PATH = Path(__file__).resolve().parents[1] / "data" / "exports"

    REQUIRED_FILES = (
        "industry_risk_scores.parquet",
        "macro_regime_flags.parquet",
        "downturn_overlay_table.parquet",
    )
    OPTIONAL_FILES = ("property_market_overlays.parquet",)

    def __init__(
        self,
        exports_path: str | Path | None = None,
        validate_contract: bool = True,
        require_property_overlays: bool = False,
    ):
        """
        Parameters
        ----------
        exports_path : str | Path | None
            Path to upstream compact parquet exports. Defaults to
            `<repo_root>/data/exports`.
        validate_contract : bool
            If True, validate required files/columns at initialisation.
        require_property_overlays : bool
            If True, require `property_market_overlays.parquet`.
        """
        self.exports_path = Path(exports_path or self.DEFAULT_EXPORTS_PATH)
        self._cache: dict[str, pd.DataFrame] = {}

        if validate_contract:
            self.validate_upstream_contract(
                require_property_overlays=require_property_overlays
            )

    def _path_for(self, filename: str) -> Path:
        return self.exports_path / filename

    def _load_parquet(self, filename: str) -> pd.DataFrame:
        if filename not in self._cache:
            path = self._path_for(filename)
            try:
                self._cache[filename] = pd.read_parquet(path)
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"Upstream export file not found: {path}"
                ) from e
            except Exception as e:
                raise ValueError(
                    f"Error reading upstream export file {path}: {e}"
                ) from e
        return self._cache[filename].copy()

    def _normalise_industry_col(self, df: pd.DataFrame, col: str = "industry") -> pd.DataFrame:
        if col not in df.columns:
            return df
        canon = df[col].map(_canonical_industry_name)
        mapped = canon.map(_CANONICAL_ANZSIC_TO_SHORT)
        df[col] = mapped.fillna(df[col])
        return df

    @staticmethod
    def _find_first_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
        for col in candidates:
            if col in df.columns:
                return col
        return None

    @staticmethod
    def _coerce_bool(series: pd.Series) -> pd.Series:
        if series.dtype == bool:
            return series
        vals = series.astype(str).str.strip().str.lower()
        true_set = {"1", "true", "t", "yes", "y"}
        return vals.isin(true_set)

    def validate_upstream_contract(self, require_property_overlays: bool = False) -> None:
        """
        Validate required upstream files and minimum required columns.

        Raises
        ------
        ValueError
            If required files/columns are missing.
        """
        issues: list[str] = []

        files_to_check = list(self.REQUIRED_FILES)
        if require_property_overlays:
            files_to_check.append("property_market_overlays.parquet")

        for filename in files_to_check:
            if not self._path_for(filename).exists():
                issues.append(f"missing required file: {filename}")

        if issues:
            raise ValueError("Invalid upstream industry contract: " + "; ".join(issues))

        # industry_risk_scores.parquet
        scores = self._load_parquet("industry_risk_scores.parquet")
        score_cols = {
            "industry_base_risk_score",
            "industry_risk_score",
            "risk_score",
        }
        if "industry" not in scores.columns:
            issues.append("industry_risk_scores.parquet missing required column: industry")
        if not score_cols.intersection(scores.columns):
            issues.append(
                "industry_risk_scores.parquet missing one of: "
                "industry_base_risk_score/industry_risk_score/risk_score"
            )

        # macro_regime_flags.parquet
        macro = self._load_parquet("macro_regime_flags.parquet")
        regime_col = self._find_first_column(macro, ["macro_regime", "regime", "scenario_name"])
        flag_col = self._find_first_column(
            macro,
            ["is_downturn", "downturn_flag", "regime_flag", "stress_regime_flag"],
        )
        if regime_col is None:
            issues.append(
                "macro_regime_flags.parquet missing one of: macro_regime/regime/scenario_name"
            )
        if flag_col is None:
            issues.append(
                "macro_regime_flags.parquet missing one of: "
                "is_downturn/downturn_flag/regime_flag/stress_regime_flag"
            )

        # downturn_overlay_table.parquet
        downturn = self._load_parquet("downturn_overlay_table.parquet")
        overlay_col = self._find_first_column(
            downturn,
            [
                "stress_delta",
                "lgd_overlay_addon",
                "downturn_scalar",
                "downturn_lgd_scalar",
                "pd_overlay_multiplier",
            ],
        )
        if "industry" not in downturn.columns:
            issues.append("downturn_overlay_table.parquet missing required column: industry")
        if overlay_col is None:
            issues.append(
                "downturn_overlay_table.parquet missing overlay field (expected one of "
                "stress_delta/lgd_overlay_addon/downturn_scalar/downturn_lgd_scalar/pd_overlay_multiplier)"
            )

        # property_market_overlays.parquet (only validate if present or required)
        property_path = self._path_for("property_market_overlays.parquet")
        if require_property_overlays or property_path.exists():
            prop = self._load_parquet("property_market_overlays.parquet")
            if "industry" not in prop.columns and "property_type" not in prop.columns:
                issues.append(
                    "property_market_overlays.parquet missing required identifier "
                    "(industry or property_type)"
                )

        if issues:
            raise ValueError("Invalid upstream industry contract: " + "; ".join(issues))

    def _load_compact_scores(self) -> pd.DataFrame:
        """Load and standardise LGD-relevant columns from industry_risk_scores."""
        raw = self._load_parquet("industry_risk_scores.parquet")
        raw = self._normalise_industry_col(raw)

        score_col = self._find_first_column(
            raw,
            ["industry_base_risk_score", "industry_risk_score", "risk_score"],
        )
        if score_col is None:
            raise ValueError("industry_risk_scores.parquet is missing a risk score column")

        out = pd.DataFrame()
        out["industry"] = raw["industry"].astype(str)
        out["industry_base_risk_score"] = pd.to_numeric(raw[score_col], errors="coerce")

        cls_col = self._find_first_column(raw, ["classification_risk_score"])
        out["classification_risk_score"] = (
            pd.to_numeric(raw[cls_col], errors="coerce")
            if cls_col
            else out["industry_base_risk_score"]
        )

        macro_col = self._find_first_column(raw, ["macro_risk_score"])
        out["macro_risk_score"] = (
            pd.to_numeric(raw[macro_col], errors="coerce")
            if macro_col
            else out["industry_base_risk_score"]
        )

        level_col = self._find_first_column(
            raw,
            ["industry_base_risk_level", "industry_risk_level", "risk_level"],
        )
        out["industry_base_risk_level"] = (
            raw[level_col].astype(str) if level_col else _risk_level_from_score(out["industry_base_risk_score"])
        )

        wc_col = self._find_first_column(
            raw,
            ["working_capital_lgd_overlay_score", "wc_lgd_overlay_score", "lgd_overlay_score"],
        )
        if wc_col:
            out["working_capital_lgd_overlay_score"] = pd.to_numeric(raw[wc_col], errors="coerce")
        else:
            # Keep WC overlay simple and local: anchored to base risk score.
            out["working_capital_lgd_overlay_score"] = (
                out["industry_base_risk_score"] + 0.30
            ).clip(1.0, 5.0)

        debt_col = self._find_first_column(
            raw,
            ["debt_to_ebitda_benchmark", "net_debt_to_ebitda_benchmark"],
        )
        out["debt_to_ebitda_benchmark"] = (
            pd.to_numeric(raw[debt_col], errors="coerce") if debt_col else np.nan
        )

        icr_col = self._find_first_column(
            raw,
            ["icr_benchmark", "interest_cover_benchmark", "interest_coverage_benchmark"],
        )
        out["icr_benchmark"] = pd.to_numeric(raw[icr_col], errors="coerce") if icr_col else np.nan

        esg_col = self._find_first_column(
            raw,
            ["esg_sensitive_sector", "esg_sensitive", "esg_flag"],
        )
        if esg_col:
            out["esg_sensitive_sector"] = self._coerce_bool(raw[esg_col])
        else:
            out["esg_sensitive_sector"] = out["industry_base_risk_score"] >= 3.2

        out["working_capital_lgd_overlay_band"] = _wc_band_from_score(
            out["working_capital_lgd_overlay_score"]
        )

        # One row per industry, last entry wins if duplicates exist upstream.
        out = out.dropna(subset=["industry"]).drop_duplicates("industry", keep="last")
        return out.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Loaders (kept for compatibility with existing notebook code)
    # ------------------------------------------------------------------

    def load_base_risk_scorecard(self) -> pd.DataFrame:
        compact = self._load_compact_scores()
        keep = [
            "industry",
            "classification_risk_score",
            "macro_risk_score",
            "industry_base_risk_score",
            "industry_base_risk_level",
        ]
        return compact[keep].copy()

    def load_benchmarks(self) -> pd.DataFrame:
        compact = self._load_compact_scores()
        out = compact[[
            "industry",
            "debt_to_ebitda_benchmark",
            "icr_benchmark",
        ]].copy()

        # Keep lightweight optional benchmark passthrough if provided upstream.
        raw = self._normalise_industry_col(self._load_parquet("industry_risk_scores.parquet"))
        optional_cols = [
            "ebitda_margin_pct_latest",
            "ar_days_benchmark",
            "ar_days_stress_benchmark",
            "ap_days_benchmark",
            "ap_days_stress_benchmark",
            "inventory_days_benchmark",
        ]
        available = [c for c in optional_cols if c in raw.columns]
        if available:
            extra = raw[["industry", *available]].copy()
            out = out.merge(extra, on="industry", how="left")
        return out

    def load_working_capital_metrics(self) -> pd.DataFrame:
        compact = self._load_compact_scores()
        out = compact[[
            "industry",
            "working_capital_lgd_overlay_score",
            "working_capital_lgd_overlay_band",
        ]].copy()

        raw = self._normalise_industry_col(self._load_parquet("industry_risk_scores.parquet"))
        for col in [
            "cash_conversion_cycle_benchmark_days",
            "cash_conversion_cycle_stress_days",
            "lgd_primary_driver",
        ]:
            out[col] = raw[col].values if col in raw.columns else np.nan

        out["lgd_primary_driver"] = out["lgd_primary_driver"].fillna("Base industry risk score")
        return out

    def load_stress_matrix(self) -> pd.DataFrame:
        """
        Build a lightweight stress matrix from compact downturn + macro flags.

        Replaces legacy `industry_stress_test_matrix.csv` dependency.
        """
        compact = self._load_compact_scores()[["industry", "industry_base_risk_score"]].copy()
        compact = compact.rename(columns={"industry_base_risk_score": "base_industry_risk_score"})

        down = self._normalise_industry_col(self._load_parquet("downturn_overlay_table.parquet"))
        macro = self._load_parquet("macro_regime_flags.parquet")

        scenario_col_macro = self._find_first_column(macro, ["scenario_name", "macro_regime", "regime"])
        flag_col = self._find_first_column(
            macro,
            ["is_downturn", "downturn_flag", "regime_flag", "stress_regime_flag"],
        )

        if scenario_col_macro is None:
            scenarios = pd.DataFrame({"scenario_name": ["Downturn"], "is_downturn": [True]})
        else:
            scenarios = pd.DataFrame(
                {
                    "scenario_name": macro[scenario_col_macro].astype(str),
                    "is_downturn": self._coerce_bool(macro[flag_col]) if flag_col else True,
                }
            ).drop_duplicates()

        scenario_col_down = self._find_first_column(down, ["scenario_name", "macro_regime", "regime"])
        overlay_col = self._find_first_column(
            down,
            [
                "stress_delta",
                "lgd_overlay_addon",
                "downturn_scalar",
                "downturn_lgd_scalar",
                "pd_overlay_multiplier",
            ],
        )

        if overlay_col is None:
            raise ValueError("downturn_overlay_table.parquet missing an overlay field")

        stress = down[["industry", overlay_col] + ([scenario_col_down] if scenario_col_down else [])].copy()
        if overlay_col in {"downturn_scalar", "downturn_lgd_scalar", "pd_overlay_multiplier"}:
            stress["stress_delta"] = pd.to_numeric(stress[overlay_col], errors="coerce") - 1.0
        else:
            stress["stress_delta"] = pd.to_numeric(stress[overlay_col], errors="coerce")

        if scenario_col_down:
            stress = stress.rename(columns={scenario_col_down: "scenario_name"})
        else:
            stress["_k"] = 1
            scenarios = scenarios.copy()
            scenarios["_k"] = 1
            stress = stress.merge(scenarios, on="_k", how="left").drop(columns=["_k"])

        out = stress.merge(compact, on="industry", how="left")
        out["base_industry_risk_score"] = out["base_industry_risk_score"].fillna(
            UNMAPPED_DEFAULTS["industry_base_risk_score"]
        )
        out["stressed_industry_risk_score"] = out["base_industry_risk_score"] + out["stress_delta"]

        out["base_macro_risk_score"] = 2.5
        if "is_downturn" in out.columns:
            out["stressed_macro_risk_score"] = out["base_macro_risk_score"] + np.where(
                out["is_downturn"], 0.35, 0.10
            )
        else:
            out["stressed_macro_risk_score"] = out["base_macro_risk_score"] + 0.35

        out["implied_monitoring_action"] = np.where(
            out["stress_delta"] >= 0.25,
            "Escalate sector review",
            np.where(out["stress_delta"] >= 0.10, "Heightened monitoring", "Monitor through BAU cycle"),
        )

        keep = [
            "industry",
            "scenario_name",
            "base_macro_risk_score",
            "stressed_macro_risk_score",
            "base_industry_risk_score",
            "stressed_industry_risk_score",
            "stress_delta",
            "implied_monitoring_action",
        ]
        return out[[c for c in keep if c in out.columns]]

    def load_credit_appetite(self) -> pd.DataFrame:
        """
        Derive compact credit appetite guidance from risk score.

        Replaces legacy `industry_credit_appetite_strategy.csv` dependency.
        """
        scorecard = self.load_base_risk_scorecard()
        esg = self.load_esg_overlay()[["industry", "esg_sensitive_sector", "esg_focus_area"]]
        out = scorecard.merge(esg, on="industry", how="left")

        tenor_map = {"Low": 7, "Medium": 5, "Elevated": 3, "High": 2}
        stance_map = {
            "Low": "Open",
            "Medium": "Selective",
            "Elevated": "Selective",
            "High": "Restricted",
        }

        out["credit_appetite_stance"] = out["industry_base_risk_level"].map(stance_map).fillna("Selective")
        out["max_tenor_years"] = out["industry_base_risk_level"].map(tenor_map).fillna(3)
        out["covenant_intensity"] = np.where(
            out["industry_base_risk_score"] >= 3.8,
            "Enhanced covenant package",
            "Standard covenant package",
        )
        out["review_frequency"] = np.where(
            out["industry_base_risk_score"] >= 3.2,
            "Quarterly",
            "Semi-Annual",
        )
        out["portfolio_action"] = np.where(
            out["industry_base_risk_score"] >= 3.8,
            "Reduce new exposure",
            "Maintain within appetite",
        )

        keep = [
            "industry",
            "industry_base_risk_level",
            "industry_base_risk_score",
            "credit_appetite_stance",
            "max_tenor_years",
            "covenant_intensity",
            "review_frequency",
            "portfolio_action",
            "esg_sensitive_sector",
            "esg_focus_area",
        ]
        return out[keep]

    def load_esg_overlay(self) -> pd.DataFrame:
        compact = self._load_compact_scores()
        out = compact[["industry", "esg_sensitive_sector"]].copy()
        out["esg_focus_area"] = np.where(
            out["esg_sensitive_sector"],
            "Sector-level ESG sensitivity review",
            "Standard ESG monitoring",
        )
        out["credit_policy_overlay"] = np.where(
            out["esg_sensitive_sector"],
            "Restricted appetite or enhanced due diligence",
            "Standard ESG due diligence",
        )
        return out

    def load_concentration_limits(
        self,
        exposure_by_industry: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Derive concentration limits from industry risk bands.

        Parameters
        ----------
        exposure_by_industry : DataFrame | None
            Optional DataFrame with `industry` and either `exposure` or `ead`.
            If supplied, utilisation and breach flags are calculated.
        """
        scorecard = self.load_base_risk_scorecard().copy()
        limit_map = {"Low": 30.0, "Medium": 25.0, "Elevated": 20.0, "High": 15.0}

        out = scorecard.rename(
            columns={
                "industry_base_risk_level": "risk_level",
                "industry_base_risk_score": "industry_base_risk_score",
            }
        )[["industry", "risk_level", "industry_base_risk_score"]]
        out["concentration_limit_pct"] = out["risk_level"].map(limit_map).fillna(20.0)

        if exposure_by_industry is None or exposure_by_industry.empty:
            out["current_exposure_pct"] = np.nan
            out["headroom_pct"] = np.nan
            out["breach"] = False
            out["utilisation_pct"] = np.nan
            return out

        exp = exposure_by_industry.copy()
        if "industry" not in exp.columns:
            raise ValueError("exposure_by_industry must include an 'industry' column")

        exposure_col = "exposure" if "exposure" in exp.columns else "ead" if "ead" in exp.columns else None
        if exposure_col is None:
            raise ValueError("exposure_by_industry must include an 'exposure' or 'ead' column")

        exp = self._normalise_industry_col(exp)
        exp = exp.groupby("industry", as_index=False)[exposure_col].sum()
        total = pd.to_numeric(exp[exposure_col], errors="coerce").fillna(0.0).sum()
        exp["current_exposure_pct"] = (
            pd.to_numeric(exp[exposure_col], errors="coerce").fillna(0.0) / total * 100.0
            if total > 0
            else 0.0
        )

        out = out.merge(exp[["industry", "current_exposure_pct"]], on="industry", how="left")
        out["current_exposure_pct"] = out["current_exposure_pct"].fillna(0.0)
        out["headroom_pct"] = out["concentration_limit_pct"] - out["current_exposure_pct"]
        out["breach"] = out["headroom_pct"] < 0
        out["utilisation_pct"] = (
            out["current_exposure_pct"] / out["concentration_limit_pct"].replace(0, np.nan) * 100.0
        ).fillna(0.0)
        return out

    def get_risk_score_lookup(self) -> dict[str, float]:
        scorecard = self.load_base_risk_scorecard()
        lookup = dict(zip(scorecard["industry"], scorecard["industry_base_risk_score"]))
        for short_name, mapped in INDUSTRY_NAME_MAP.items():
            if mapped is None and short_name not in lookup:
                lookup[short_name] = UNMAPPED_DEFAULTS["industry_base_risk_score"]
        return lookup

    def get_wc_lgd_overlay_lookup(self) -> dict[str, float]:
        wc = self.load_working_capital_metrics()
        lookup = dict(zip(wc["industry"], wc["working_capital_lgd_overlay_score"]))
        for short_name, mapped in INDUSTRY_NAME_MAP.items():
            if mapped is None and short_name not in lookup:
                lookup[short_name] = UNMAPPED_DEFAULTS["working_capital_lgd_overlay_score"]
        return lookup


# ===========================================================================
# INDUSTRY-ADJUSTED LGD COMPUTATION FUNCTIONS
# ===========================================================================


def compute_industry_downturn_scalar(risk_score, base_scalar, alpha=0.15):
    """
    Compute industry-adjusted downturn scalar.

    Formula:
        adjusted = base_scalar * (1 + alpha * (risk_score - 2.5) / 2.0)
    """
    risk_score = np.asarray(risk_score, dtype=float)
    base_scalar = np.asarray(base_scalar, dtype=float)
    return base_scalar * (1 + alpha * (risk_score - 2.5) / 2.0)


def compute_industry_moc_adjustment(risk_score, base_moc, beta=0.20):
    """
    Compute industry-adjusted margin of conservatism.

    Formula:
        adjusted = base_moc * (1 + beta * (risk_score - 2.5) / 2.5)
    """
    risk_score = np.asarray(risk_score, dtype=float)
    return base_moc * (1 + beta * (risk_score - 2.5) / 2.5)


def compute_industry_recovery_haircut(risk_score):
    """
    Compute additional recovery haircut based on industry risk.

    Formula:
        haircut = 0.02 * max(risk_score - 2.0, 0)
    """
    risk_score = np.asarray(risk_score, dtype=float)
    return 0.02 * np.maximum(risk_score - 2.0, 0)


def compute_working_capital_lgd_adjustment(wc_lgd_overlay_score):
    """
    Compute additive LGD adjustment from working capital risk.

    Formula:
        adjustment = 0.01 * max(wc_overlay_score - 2.0, 0)
    """
    wc_lgd_overlay_score = np.asarray(wc_lgd_overlay_score, dtype=float)
    return 0.01 * np.maximum(wc_lgd_overlay_score - 2.0, 0)


# ===========================================================================
# DEVELOPMENT TYPE -> INDUSTRY MAPPING
# ===========================================================================

DEVELOPMENT_TYPE_TO_INDUSTRY = {
    "Residential Apartments": "Construction",
    "Residential Houses/Lots": "Construction",
    "Commercial Office": "Professional Services",
    "Mixed-Use": "Construction",
    "Industrial": "Manufacturing",
}


def map_development_type_to_industry(development_type):
    if isinstance(development_type, pd.Series):
        return development_type.map(DEVELOPMENT_TYPE_TO_INDUSTRY).fillna("Construction")
    return DEVELOPMENT_TYPE_TO_INDUSTRY.get(development_type, "Construction")


# ===========================================================================
# MASTER ENRICHMENT FUNCTION
# ===========================================================================


def enrich_loans_with_industry_risk(loans_df, loader: IndustryRiskLoader, product_type="commercial"):
    """
    Enrich a loans DataFrame with LGD-relevant industry risk features.

    Uses only the compact upstream contract and internal lightweight derivations.
    """
    out = loans_df.copy()

    if product_type == "development" and "industry" not in out.columns:
        out["industry"] = map_development_type_to_industry(out["development_type"])

    if "industry" not in out.columns:
        return out

    risk_lookup = loader.get_risk_score_lookup()
    wc_lookup = loader.get_wc_lgd_overlay_lookup()

    scorecard = loader.load_base_risk_scorecard()
    level_lookup = dict(zip(scorecard["industry"], scorecard["industry_base_risk_level"]))
    for short_name, mapped in INDUSTRY_NAME_MAP.items():
        if mapped is None and short_name not in level_lookup:
            level_lookup[short_name] = UNMAPPED_DEFAULTS["industry_base_risk_level"]

    benchmarks = loader.load_benchmarks()
    debt_ebitda_lookup = dict(zip(benchmarks["industry"], benchmarks["debt_to_ebitda_benchmark"]))
    icr_lookup = dict(zip(benchmarks["industry"], benchmarks["icr_benchmark"]))

    esg = loader.load_esg_overlay()
    esg_lookup = dict(zip(esg["industry"], esg["esg_sensitive_sector"]))

    out["industry_risk_score"] = out["industry"].map(risk_lookup).fillna(
        UNMAPPED_DEFAULTS["industry_base_risk_score"]
    )
    out["industry_risk_level"] = out["industry"].map(level_lookup).fillna(
        UNMAPPED_DEFAULTS["industry_base_risk_level"]
    )
    out["wc_lgd_overlay_score"] = out["industry"].map(wc_lookup).fillna(
        UNMAPPED_DEFAULTS["working_capital_lgd_overlay_score"]
    )
    out["industry_debt_to_ebitda_benchmark"] = out["industry"].map(debt_ebitda_lookup).fillna(
        UNMAPPED_DEFAULTS["debt_to_ebitda_benchmark"]
    )
    out["industry_icr_benchmark"] = out["industry"].map(icr_lookup).fillna(
        UNMAPPED_DEFAULTS["icr_benchmark"]
    )

    esg_mapped = out["industry"].map(esg_lookup)
    out["industry_esg_sensitive"] = esg_mapped.where(
        esg_mapped.notna(), UNMAPPED_DEFAULTS["esg_sensitive_sector"]
    ).astype(bool)

    out["industry_recovery_haircut"] = compute_industry_recovery_haircut(out["industry_risk_score"])
    out["wc_lgd_adjustment"] = compute_working_capital_lgd_adjustment(out["wc_lgd_overlay_score"])

    return out