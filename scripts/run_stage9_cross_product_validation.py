from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.dont_write_bytecode = True


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"

NOTEBOOKS = [
    "08_cre_investment_lgd.ipynb",
    "09_residual_stock_lgd.ipynb",
    "10_land_subdivision_lgd.ipynb",
    "11_bridging_loan_lgd.ipynb",
    "12_mezz_second_mortgage_lgd.ipynb",
    "13_cross_product_comparison.ipynb",
]

REQUIRED_OUTPUTS = {
    "cre_investment_loan_level_output.csv": ["ead", "lgd_base", "lgd_downturn", "lgd_final", "time_to_recovery_months_base"],
    "residual_stock_loan_level_output.csv": ["ead", "lgd_base", "lgd_downturn", "lgd_final", "time_to_sale_base"],
    "land_subdivision_loan_level_output.csv": ["ead", "lgd_base", "lgd_downturn", "lgd_final", "time_to_sell_base"],
    "bridging_loan_level_output.csv": ["ead", "lgd_base", "lgd_downturn", "lgd_final", "exit_time_base"],
    "mezz_second_mortgage_loan_level_output.csv": ["mezz_ead", "lgd_base", "lgd_downturn", "lgd_final", "time_to_recovery_base"],
    "cross_product_comparison.csv": [
        "product",
        "ead_weighted_lgd_base",
        "ead_weighted_lgd_downturn",
        "ead_weighted_lgd_final",
        "downturn_sensitivity_pp",
        "mean_recovery_time_months",
    ],
}


def _display(_obj):
    # Keep script output concise; notebook tables are still persisted to CSV outputs.
    return None


def _step_label_for_notebook(nb_name: str) -> str:
    return f"execute_{Path(nb_name).stem}"


def _execute_notebook(nb_name: str) -> dict[str, object]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.show = lambda *args, **kwargs: None

    nb_path = NOTEBOOK_DIR / nb_name
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    env = {"__name__": "__main__", "display": _display}

    t0 = time.time()
    code_cells = 0
    os.environ["LGD_NOTEBOOK_SHOW_PLOTS"] = "0"
    os.chdir(NOTEBOOK_DIR)

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if not src.strip():
            continue
        exec(compile(src, f"{nb_name}::cell", "exec"), env)
        code_cells += 1

    elapsed = time.time() - t0
    return {"step": _step_label_for_notebook(nb_name), "status": "PASS", "detail": f"code_cells={code_cells}; elapsed_sec={elapsed:.2f}"}


def _validate_output(filename: str, required_cols: list[str]) -> dict[str, object]:
    path = TABLE_DIR / filename
    if not path.exists():
        return {"step": f"validate_{filename}", "status": "FAIL", "detail": "missing file"}
    if path.stat().st_size <= 0:
        return {"step": f"validate_{filename}", "status": "FAIL", "detail": "empty file"}

    df = pd.read_csv(path)
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        return {"step": f"validate_{filename}", "status": "FAIL", "detail": f"missing columns: {missing_cols}"}
    if len(df) == 0:
        return {"step": f"validate_{filename}", "status": "FAIL", "detail": "zero rows"}

    return {"step": f"validate_{filename}", "status": "PASS", "detail": f"rows={len(df)}; cols={len(df.columns)}"}


def _validate_cross_product_metrics() -> dict[str, object]:
    df = pd.read_csv(TABLE_DIR / "cross_product_comparison.csv")
    key_cols = [
        "ead_weighted_lgd_base",
        "ead_weighted_lgd_downturn",
        "ead_weighted_lgd_final",
        "mean_recovery_time_months",
    ]
    nan_counts = {c: int(df[c].isna().sum()) for c in key_cols}

    cre_row = df[df["product"] == "CRE Investment"]
    cre_missing = True
    if not cre_row.empty:
        cre_missing = bool(cre_row["mean_recovery_time_months"].isna().iloc[0])

    passed = all(v == 0 for v in nan_counts.values()) and not cre_missing
    detail = f"nan_counts={nan_counts}; cre_recovery_missing={cre_missing}"
    return {"step": "validate_cross_product_metrics", "status": "PASS" if passed else "FAIL", "detail": detail}


def main():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for nb_name in NOTEBOOKS:
        try:
            rows.append(_execute_notebook(nb_name))
        except Exception as exc:  # pragma: no cover - defensive reporting path
            rows.append({"step": _step_label_for_notebook(nb_name), "status": "FAIL", "detail": f"{type(exc).__name__}: {exc}"})

    for filename, required_cols in REQUIRED_OUTPUTS.items():
        rows.append(_validate_output(filename, required_cols))

    rows.append(_validate_cross_product_metrics())

    report = pd.DataFrame(rows)
    report_path = TABLE_DIR / "stage9_cross_product_validation_report.csv"
    report.to_csv(report_path, index=False)

    passed = int((report["status"] == "PASS").sum())
    total = len(report)
    print(f"Stage 9 validation completed: {passed}/{total} checks passed.")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
