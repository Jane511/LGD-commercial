from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_source_adapter import export_controlled_input_templates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create controlled-system input template files from canonical LGD contract."
    )
    parser.add_argument("--output-root", default="data/controlled/templates")
    args = parser.parse_args()

    report = export_controlled_input_templates(output_root=args.output_root)
    print(f"output_root={report['output_root']}")
    print(f"files_written={len(report['files_written'])}")


if __name__ == "__main__":
    main()
