from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.lgd_scoring import (  # noqa: E402
    score_batch_loans,
    score_batch_from_source,
    score_single_loan,
    score_single_loan_from_source_template,
)


def _read_json_payload(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Single-loan JSON input must be an object/dict")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Score LGD for single or batch loan inputs.")
    parser.add_argument("--product-type", required=True)
    parser.add_argument("--scenario-id", default="baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-mode", choices=["generated", "controlled"], default="generated")
    parser.add_argument("--controlled-root", default="data/controlled")
    parser.add_argument("--single-json", help="Path to single-loan JSON payload.")
    parser.add_argument("--input-csv", help="Path to batch CSV payload.")
    parser.add_argument("--batch-from-source", action="store_true", help="Ignore --input-csv and score full product loans from selected source adapter.")
    parser.add_argument("--use-source-template", action="store_true", help="For --single-json: merge payload onto a source-template loan row before scoring.")
    parser.add_argument("--output", required=True, help="Path to output JSON/CSV.")
    parser.add_argument("--full-output", action="store_true", help="Include full engine columns.")
    args = parser.parse_args()

    if not args.batch_from_source and bool(args.single_json) == bool(args.input_csv):
        raise ValueError("Provide exactly one of --single-json or --input-csv (unless --batch-from-source is used)")
    if args.batch_from_source and (args.single_json or args.input_csv):
        raise ValueError("--batch-from-source cannot be combined with --single-json or --input-csv")

    output_path = Path(args.output)

    if args.single_json:
        payload = _read_json_payload(Path(args.single_json))
        if args.use_source_template:
            result = score_single_loan_from_source_template(
                payload=payload,
                product_type=args.product_type,
                source_mode=args.source_mode,
                controlled_root=args.controlled_root,
                scenario_id=args.scenario_id,
                seed=args.seed,
                return_full=args.full_output,
            )
        else:
            result = score_single_loan(
                payload=payload,
                product_type=args.product_type,
                scenario_id=args.scenario_id,
                seed=args.seed,
                source_mode=args.source_mode,
                return_full=args.full_output,
            )
        if output_path.suffix.lower() == ".json":
            output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        else:
            pd.DataFrame([result]).to_csv(output_path, index=False)
    else:
        if args.batch_from_source:
            scored = score_batch_from_source(
                product_type=args.product_type,
                source_mode=args.source_mode,
                controlled_root=args.controlled_root,
                scenario_id=args.scenario_id,
                seed=args.seed,
                return_full=args.full_output,
            )
        else:
            frame = pd.read_csv(args.input_csv)
            scored = score_batch_loans(
                df=frame,
                product_type=args.product_type,
                scenario_id=args.scenario_id,
                seed=args.seed,
                source_mode=args.source_mode,
                return_full=args.full_output,
            )
        if output_path.suffix.lower() == ".json":
            output_path.write_text(scored.to_json(orient="records", indent=2), encoding="utf-8")
        else:
            scored.to_csv(output_path, index=False)

    print(f"product_type={args.product_type}")
    print(f"output={output_path.resolve()}")


if __name__ == "__main__":
    main()
