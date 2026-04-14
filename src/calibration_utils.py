"""
Calibration Utilities — thin re-export layer for notebook ergonomics.

Notebooks import everything they need from this single module:

    from src.calibration_utils import (
        compute_realised_lgd,
        segment_lgd,
        compute_long_run_lgd,
        compare_model_vs_actual,
        compute_calibration_adjustment,
        apply_regulatory_floor,
        MoCRegister,
        apply_moc,
        run_calibration_pipeline,
        classify_economic_regime,
        assign_regime_to_workouts,
        export_regime_classification,
        run_full_validation_suite,
        compute_gini_coefficient,
        hosmer_lemeshow_test,
        generate_compliance_map,
        validate_observation_periods,
        export_compliance_map,
        load_rba_lending_rates,
        get_discount_rate_for_loan,
        build_discount_rate_register,
        load_apra_adi_benchmarks,
        generate_benchmark_comparison,
        estimate_lgd_pd_correlation,
        apply_correlation_adjustment,
        build_lgd_pd_annual_series,
        CALIBRATION_STEP_ORDER,
    )

This module contains NO logic — it is a pure re-export wrapper.
Any calibration logic belongs in the respective src/ module, not here.

APS 113 pipeline step order (for reference in notebooks):
    1. compute_realised_lgd()          — s.32-34, s.49-51
    2. classify_economic_regime()      — s.43, s.46-50
    3. assign_regime_to_workouts()     — s.43
    4. segment_lgd()                   — s.52
    5. compute_long_run_lgd()          — s.43-44
    6. compare_model_vs_actual()       — s.60-62
    7. apply_downturn_overlay()        — s.46-50
    8. apply_correlation_adjustment()  — s.55-57 (Frye-Jacobs)
    9. MoCRegister + apply_moc()       — s.63-65 (AFTER downturn)
    10. apply_regulatory_floor()       — s.58
    11. run_full_validation_suite()    — s.66-68
"""
from src.lgd_calculations import (
    compute_realised_lgd,
    segment_lgd,
    compute_long_run_lgd,
    compare_model_vs_actual,
    compute_calibration_adjustment,
    apply_regulatory_floor,
)

from src.moc_framework import (
    MoCRegister,
    apply_moc,
    run_calibration_pipeline,
    MOC_SOURCES,
    PRODUCT_MOC_CAPS,
)

from src.regime_classifier import (
    classify_economic_regime,
    assign_regime_to_workouts,
    export_regime_classification,
)

from src.validation_suite import (
    run_full_validation_suite,
    compute_gini_coefficient,
    hosmer_lemeshow_test,
)

from src.aps113_compliance import (
    generate_compliance_map,
    validate_observation_periods,
    export_compliance_map,
    APS113_REQUIREMENTS,
)

from src.rba_rates_loader import (
    load_rba_lending_rates,
    get_discount_rate_for_loan,
    build_discount_rate_register,
    export_discount_rate_register,
)

from src.apra_benchmarks import (
    load_apra_adi_benchmarks,
    generate_benchmark_comparison,
    export_benchmark_comparison,
)

from src.lgd_pd_correlation import (
    estimate_lgd_pd_correlation,
    apply_correlation_adjustment,
    build_lgd_pd_annual_series,
    export_correlation_report,
)

# Reuse from existing proxy engine (not re-implemented)
from src.lgd_calculation import (
    apply_downturn_overlay,
    exposure_weighted_average,
    build_weighted_lgd_output,
)

# APS 113 pipeline step order — for notebook documentation
CALIBRATION_STEP_ORDER = [
    ("1", "compute_realised_lgd",         "s.32-34, s.49-51"),
    ("2", "classify_economic_regime",     "s.43, s.46-50"),
    ("3", "assign_regime_to_workouts",    "s.43"),
    ("4", "segment_lgd",                  "s.52"),
    ("5", "compute_long_run_lgd",         "s.43-44"),
    ("6", "compare_model_vs_actual",      "s.60-62"),
    ("7", "apply_downturn_overlay",       "s.46-50"),
    ("8", "apply_correlation_adjustment", "s.55-57"),
    ("9", "MoCRegister + apply_moc",      "s.63-65"),
    ("10", "apply_regulatory_floor",      "s.58"),
    ("11", "run_full_validation_suite",   "s.66-68"),
]

__all__ = [
    "compute_realised_lgd",
    "segment_lgd",
    "compute_long_run_lgd",
    "compare_model_vs_actual",
    "compute_calibration_adjustment",
    "apply_regulatory_floor",
    "MoCRegister",
    "apply_moc",
    "run_calibration_pipeline",
    "MOC_SOURCES",
    "PRODUCT_MOC_CAPS",
    "classify_economic_regime",
    "assign_regime_to_workouts",
    "export_regime_classification",
    "run_full_validation_suite",
    "compute_gini_coefficient",
    "hosmer_lemeshow_test",
    "generate_compliance_map",
    "validate_observation_periods",
    "export_compliance_map",
    "APS113_REQUIREMENTS",
    "load_rba_lending_rates",
    "get_discount_rate_for_loan",
    "build_discount_rate_register",
    "export_discount_rate_register",
    "load_apra_adi_benchmarks",
    "generate_benchmark_comparison",
    "export_benchmark_comparison",
    "estimate_lgd_pd_correlation",
    "apply_correlation_adjustment",
    "build_lgd_pd_annual_series",
    "export_correlation_report",
    "apply_downturn_overlay",
    "exposure_weighted_average",
    "build_weighted_lgd_output",
    "CALIBRATION_STEP_ORDER",
]
