from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

import matplotlib

sys.dont_write_bytecode = True


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "11_bridging_loan_lgd.ipynb"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

EXPECTED_TABLES = [
    "bridging_loan_level_output.csv",
    "bridging_segment_summary.csv",
    "bridging_delay_summary.csv",
    "bridging_scenario_summary.csv",
    "bridging_validation_checks.csv",
]
EXPECTED_FIGURES = ["bridging_delay_vs_lgd.png"]


def _load_code_cells(notebook_path: Path) -> list[str]:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    return [
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    ]


def _run_notebook_cells_non_interactive(code_cells: list[str]) -> pd.DataFrame:
    # Force a non-interactive backend for script-safe execution.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: WPS433

    plt.show = lambda *args, **kwargs: None

    timing_rows: list[dict[str, object]] = []
    context: dict[str, object] = {"__name__": "__main__", "display": lambda *args, **kwargs: None}

    notebook_dir = NOTEBOOK_PATH.parent
    original_cwd = Path.cwd()
    try:
        os.chdir(notebook_dir)
        for idx, source in enumerate(code_cells, start=1):
            if not source.strip():
                continue
            print(f"CELL {idx} START", flush=True)
            start = time.perf_counter()
            exec(compile(source, f"{NOTEBOOK_PATH.name}::cell_{idx}", "exec"), context)
            elapsed = time.perf_counter() - start
            print(f"CELL {idx} END elapsed_sec={elapsed:.3f}", flush=True)
            timing_rows.append(
                {
                    "cell_number": idx,
                    "elapsed_sec": round(elapsed, 6),
                    "status": "ok",
                }
            )
    finally:
        os.chdir(original_cwd)

    return pd.DataFrame(timing_rows)


def _build_output_report() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for file_name in EXPECTED_TABLES:
        path = TABLE_DIR / file_name
        rows.append(
            {
                "artifact_type": "table",
                "artifact_name": file_name,
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
            }
        )
    for file_name in EXPECTED_FIGURES:
        path = FIGURE_DIR / file_name
        rows.append(
            {
                "artifact_type": "figure",
                "artifact_name": file_name,
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
            }
        )
    return pd.DataFrame(rows)


def _validation_checks_passed() -> bool:
    checks_path = TABLE_DIR / "bridging_validation_checks.csv"
    if not checks_path.exists():
        return False
    checks = pd.read_csv(checks_path)
    if "passed" not in checks.columns or checks.empty:
        return False
    return bool(checks["passed"].fillna(False).all())


def main() -> int:
    if not NOTEBOOK_PATH.exists():
        print(f"Missing notebook: {NOTEBOOK_PATH}", flush=True)
        return 1

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    code_cells = _load_code_cells(NOTEBOOK_PATH)
    print(f"Stage 7 diagnostic start: {NOTEBOOK_PATH.name}, code_cells={len(code_cells)}", flush=True)
    timing_df = _run_notebook_cells_non_interactive(code_cells)
    timing_path = TABLE_DIR / "bridging_notebook_cell_timings.csv"
    timing_df.to_csv(timing_path, index=False)

    output_report = _build_output_report()
    output_report_path = TABLE_DIR / "bridging_stage7_validation_report.csv"
    output_report.to_csv(output_report_path, index=False)

    missing = output_report.loc[~output_report["exists"], "artifact_name"].tolist()
    checks_ok = _validation_checks_passed()

    print(f"Timings report: {timing_path}", flush=True)
    print(f"Output report:  {output_report_path}", flush=True)
    print(f"Missing artifacts: {missing if missing else 'none'}", flush=True)
    print(f"bridging_validation_checks all passed: {checks_ok}", flush=True)

    if missing or not checks_ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
