from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


REQUIRED_COLUMNS = [
    "parameter_group",
    "product_scope",
    "segment_scope",
    "parameter_name",
    "value",
    "effective_date",
    "version",
    "owner",
    "rationale",
]


@dataclass(frozen=True)
class ParameterMeta:
    version: str
    parameter_hash: str
    csv_path: str


class OverlayParameterManager:
    """Load, validate, and serve versioned overlay parameter tables."""

    def __init__(
        self,
        csv_path: str | Path | None = None,
        manifest_path: str | Path | None = None,
    ):
        repo_root = Path(__file__).resolve().parents[1]
        self.csv_path = Path(csv_path or repo_root / "data" / "config" / "overlay_parameters.csv")
        self.manifest_path = Path(
            manifest_path or repo_root / "data" / "config" / "overlay_parameters_manifest.json"
        )
        logger.info("Loading overlay parameters from: %s", self.csv_path)
        self._df = self._load_and_validate()
        self.meta = self._build_meta()
        self._validate_manifest()
        logger.info(
            "Overlay parameters loaded — version=%s, %d rows, hash=%s",
            self.meta.version,
            len(self._df),
            self.meta.parameter_hash[:12],
        )

    def _load_and_validate(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.csv_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Overlay parameter file not found: {self.csv_path}"
            ) from e
        except Exception as e:
            raise ValueError(
                f"Error reading overlay parameter file {self.csv_path}: {e}"
            ) from e

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"overlay parameter table missing required columns: {missing}")

        key_cols = [
            "parameter_group",
            "product_scope",
            "segment_scope",
            "parameter_name",
            "effective_date",
            "version",
        ]
        if df.duplicated(subset=key_cols).any():
            dups = df.loc[df.duplicated(subset=key_cols, keep=False), key_cols]
            raise ValueError(f"overlay parameter table has duplicate keys: {dups.head(5).to_dict('records')}")

        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        if df["value"].isna().any():
            bad = df.loc[df["value"].isna(), ["parameter_name", "product_scope", "segment_scope"]]
            raise ValueError(f"overlay parameter table has non-numeric value rows: {bad.head(5).to_dict('records')}")

        # Lightweight guardrails against implausible parameter values.
        if ((df["value"] < -5) | (df["value"] > 5)).any():
            bad = df.loc[(df["value"] < -5) | (df["value"] > 5), ["parameter_name", "value"]]
            raise ValueError(f"overlay parameter table has out-of-range values: {bad.head(5).to_dict('records')}")

        scalar_mask = df["parameter_name"].str.contains("scalar", case=False, na=False)
        if (df.loc[scalar_mask, "value"] <= 0).any():
            bad = df.loc[scalar_mask & (df["value"] <= 0), ["parameter_name", "value"]]
            raise ValueError(f"overlay scalar must be > 0: {bad.head(5).to_dict('records')}")

        base_moc_mask = df["parameter_name"].str.contains("base_moc", case=False, na=False)
        if ((df.loc[base_moc_mask, "value"] < 0) | (df.loc[base_moc_mask, "value"] > 1)).any():
            bad = df.loc[
                base_moc_mask & ((df["value"] < 0) | (df["value"] > 1)),
                ["parameter_name", "value"],
            ]
            raise ValueError(f"base_moc parameters must be in [0,1]: {bad.head(5).to_dict('records')}")

        moc_mult_mask = df["parameter_name"].str.contains("moc_pd_multiplier", case=False, na=False)
        if ((df.loc[moc_mult_mask, "value"] < 0) | (df.loc[moc_mult_mask, "value"] > 3)).any():
            bad = df.loc[
                moc_mult_mask & ((df["value"] < 0) | (df["value"] > 3)),
                ["parameter_name", "value"],
            ]
            raise ValueError(
                f"moc_pd_multiplier parameters must be in [0,3]: {bad.head(5).to_dict('records')}"
            )

        return df

    def _build_meta(self) -> ParameterMeta:
        table_hash = hashlib.sha256(self.csv_path.read_bytes()).hexdigest()
        versions = self._df["version"].astype(str).dropna().unique().tolist()
        if len(versions) != 1:
            raise ValueError(f"overlay parameter table must contain exactly one active version, got {versions}")
        return ParameterMeta(version=versions[0], parameter_hash=table_hash, csv_path=str(self.csv_path))

    def _validate_manifest(self) -> None:
        if not self.manifest_path.exists():
            return
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        expected_version = payload.get("expected_version")
        expected_hash = payload.get("expected_sha256")
        if expected_version and expected_version != self.meta.version:
            raise ValueError(
                f"parameter version mismatch: expected {expected_version}, got {self.meta.version}"
            )
        if expected_hash and expected_hash != self.meta.parameter_hash:
            raise ValueError("parameter hash mismatch against manifest")

    @property
    def frame(self) -> pd.DataFrame:
        return self._df.copy()

    def get_value(
        self,
        product_scope: str,
        parameter_name: str,
        segment_scope: str = "all",
        default: float | None = None,
    ) -> float | None:
        df = self._df
        mask = (
            (df["parameter_name"] == parameter_name)
            & (df["product_scope"].isin([product_scope, "all"]))
            & (df["segment_scope"].isin([segment_scope, "all"]))
        )
        hits = df.loc[mask].copy()
        if hits.empty:
            logger.debug(
                "get_value: no match for parameter_name=%r product_scope=%r segment_scope=%r — returning default %r",
                parameter_name,
                product_scope,
                segment_scope,
                default,
            )
            return default

        # Deterministic precedence: exact product+segment, exact product+all, all+segment, all+all
        hits["rank"] = 3
        hits.loc[(hits["product_scope"] == product_scope) & (hits["segment_scope"] == segment_scope), "rank"] = 0
        hits.loc[(hits["product_scope"] == product_scope) & (hits["segment_scope"] == "all"), "rank"] = 1
        hits.loc[(hits["product_scope"] == "all") & (hits["segment_scope"] == segment_scope), "rank"] = 2
        hits = hits.sort_values(["rank", "effective_date", "parameter_group"]).reset_index(drop=True)
        return float(hits.iloc[0]["value"])

    def get_map(
        self,
        product_scope: str,
        parameter_name: str,
        prefix: str,
    ) -> dict[str, float]:
        df = self._df
        mask = (
            (df["parameter_name"] == parameter_name)
            & (df["product_scope"].isin([product_scope, "all"]))
            & (df["segment_scope"].astype(str).str.startswith(prefix))
        )
        rows = df.loc[mask, ["product_scope", "segment_scope", "value"]].copy()
        if rows.empty:
            return {}

        rows["key"] = rows["segment_scope"].astype(str).str[len(prefix):]
        rows["rank"] = rows["product_scope"].map({product_scope: 0, "all": 1}).fillna(2)
        rows = rows.sort_values(["key", "rank"]).drop_duplicates(subset=["key"], keep="first")
        return {str(r["key"]): float(r["value"]) for _, r in rows.iterrows()}

    def build_parameter_version_report(self) -> pd.DataFrame:
        checks = [
            {"check": "schema_valid", "status": True, "detail": "required columns present and validated"},
            {"check": "single_active_version", "status": True, "detail": f"version={self.meta.version}"},
            {"check": "manifest_file_present", "status": bool(self.manifest_path.exists()), "detail": str(self.manifest_path)},
        ]
        return pd.DataFrame(
            {
                "parameter_version": [self.meta.version] * len(checks),
                "parameter_hash": [self.meta.parameter_hash] * len(checks),
                "parameter_file": [self.meta.csv_path] * len(checks),
                "check": [c["check"] for c in checks],
                "status": [c["status"] for c in checks],
                "detail": [c["detail"] for c in checks],
            }
        )
