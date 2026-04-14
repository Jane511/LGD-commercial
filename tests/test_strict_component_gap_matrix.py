from __future__ import annotations

from pathlib import Path
import importlib.util


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "src" / "governance" / "gap_matrix.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("strict_gap_matrix_builder", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load strict gap matrix builder module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _split_piped(text: str) -> list[str]:
    return [part.strip() for part in str(text).split("|") if part.strip()]


def test_strict_gap_matrix_has_16_rows_and_balanced_docs():
    module = _load_module()
    df = module.build_matrix()

    assert len(df) == 16
    counts = df["doc"].value_counts().to_dict()
    assert counts.get("cashflow_methodology") == 8
    assert counts.get("property_backed_methodology") == 8
    assert set(df["component_id"]) == {"5.1", "5.2", "5.3", "5.4", "5.5", "5.6", "5.7", "5.8"}


def test_implemented_rows_have_full_evidence_fields():
    module = _load_module()
    df = module.build_matrix()
    implemented = df[df["status"] == "Implemented"]

    for _, row in implemented.iterrows():
        assert str(row["evidence_code_paths"]).strip()
        assert str(row["evidence_output_tables"]).strip()
        assert str(row["logic_trace"]).strip()


def test_non_implemented_rows_have_remediation_and_acceptance():
    module = _load_module()
    df = module.build_matrix()
    non_impl = df[df["status"].isin(["Proxy-only", "Missing"])]

    assert len(non_impl) > 0
    for _, row in non_impl.iterrows():
        tasks = _split_piped(row["exact_remediation_tasks"])
        assert len(tasks) >= 3
        assert str(row["acceptance_criteria"]).strip()


def test_strict_option_audit_lists_options_with_pass_fail_tokens():
    module = _load_module()
    df = module.build_matrix()

    for _, row in df.iterrows():
        documented = _split_piped(row["documented_model_options"])
        audit_items = _split_piped(row["strict_option_audit"])

        assert len(audit_items) == len(documented)
        for idx, item in enumerate(audit_items, start=1):
            assert item.startswith(f"{idx}:")
            assert ":PASS:" in item or ":FAIL:" in item
