from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_generation import generate_all_datasets  # noqa: E402
from src.lgd_calculation import run_full_pipeline  # noqa: E402


def test_core_products_have_base_downturn_weighted_outputs():
    datasets = generate_all_datasets()
    results = run_full_pipeline(datasets, include_reporting=False)

    required_cols = {
        "facility_count",
        "total_ead",
        "ead_weighted_lgd_base",
        "ead_weighted_lgd_downturn",
        "ead_weighted_lgd_final",
    }

    for product in ["mortgage", "commercial", "development"]:
        assert product in results
        assert "weighted_output" in results[product]
        w = results[product]["weighted_output"]
        assert not w.empty
        assert required_cols.issubset(set(w.columns))
