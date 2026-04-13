from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[1]
REPO_NAME='LGD-Cashflow-and-Property-Lending'
PIPELINE_KIND='lgd'
EXPECTED_OUTPUTS=['lgd_segment_summary.csv', 'recovery_waterfall.csv', 'downturn_lgd_output.csv', 'lgd_validation_report.csv', 'policy_parameter_register.csv']

# Governance: policy parameters that should be visible in reporting outputs.
# These are portfolio-policy settings and documentation hooks, not a claim of
# production calibration quality.
POLICY_PARAMETER_REGISTER = [
    {
        'group': 'aggregation',
        'parameter': 'lgd_aggregation_method',
        'value': 'ead_weighted',
        'description': 'Portfolio LGD uses Sum(LGD_i*EAD_i)/Sum(EAD_i).',
        'fallback_hierarchy': 'not_applicable',
        'category': 'methodology_standard',
        'status': 'active',
        'calibration_status': 'n/a_rule_based',
    },
    {
        'group': 'discount_rate_policy',
        'parameter': 'discount_rate_rule',
        'value': 'max(contract_rate_proxy, cost_of_funds_proxy)',
        'description': 'Workout discount rate uses contractual return floor and funding floor.',
        'fallback_hierarchy': 'discount_rate',
        'category': 'policy_parameter',
        'status': 'active',
        'calibration_status': 'proxy_rule_not_internally_calibrated',
    },
    {
        'group': 'mortgage_cure_proxy',
        'parameter': 'cure_lgd_formula',
        'value': 'lgd = (1 - cure_rate) * liquidation_loss',
        'description': 'Two-stage mortgage LGD with explicit cure channel.',
        'fallback_hierarchy': 'cure_model_inputs',
        'category': 'methodology_standard',
        'status': 'active',
        'calibration_status': 'proxy_rule_not_internally_calibrated',
    },
    {
        'group': 'mortgage_cure_proxy',
        'parameter': 'mortgage_stage1_probability_of_cure',
        'value': 'proxy_logit_drivers_lvr_arrears_borrower_behaviour',
        'description': 'Stage 1 estimates P(cure) from transparent proxy drivers.',
        'fallback_hierarchy': 'cure_model_inputs',
        'category': 'stage_1_cure',
        'status': 'active',
        'calibration_status': 'proxy_not_calibrated_to_internal_collections_tape',
    },
    {
        'group': 'mortgage_cure_proxy',
        'parameter': 'mortgage_stage2_liquidation_loss_if_not_cured',
        'value': 'proxy_liquidation_loss_non_cure_channel',
        'description': 'Stage 2 estimates liquidation loss conditional on non-cure.',
        'fallback_hierarchy': 'cure_model_inputs',
        'category': 'stage_2_liquidation',
        'status': 'active',
        'calibration_status': 'proxy_not_calibrated_to_internal_workout_tape',
    },
    {
        'group': 'mortgage_cure_proxy',
        'parameter': 'arrears_stage_proxy',
        'value': '30-59/60-89/90+ dpd proxy from lvr,dti,credit_score',
        'description': 'Proxy arrears stage when detailed delinquency tape is unavailable.',
        'fallback_hierarchy': 'arrears_proxy',
        'category': 'proxy_input',
        'status': 'active',
        'calibration_status': 'proxy_not_calibrated_to_internal_collections_tape',
    },
    {
        'group': 'mortgage_cure_proxy',
        'parameter': 'repayment_behaviour_proxy',
        'value': 'weak/stable/strong from pii, score, seasoning, dti, lvr',
        'description': 'Behaviour proxy for cure likelihood in mortgage workout.',
        'fallback_hierarchy': 'behaviour_proxy',
        'category': 'proxy_input',
        'status': 'active',
        'calibration_status': 'proxy_not_calibrated_to_internal_collections_tape',
    },
    {
        'group': 'secured_cure_overlay',
        'parameter': 'commercial_cure_rate_proxy_cap',
        'value': '0.30',
        'description': 'Cap on simplified secured-commercial cure overlay.',
        'fallback_hierarchy': 'secured_cure_overlay',
        'category': 'cure_overlay',
        'status': 'active',
        'calibration_status': 'proxy_not_calibrated_to_internal_workout_tape',
    },
    {
        'group': 'secured_cure_overlay',
        'parameter': 'development_cure_rate_proxy_cap',
        'value': '0.25',
        'description': 'Cap on simplified development cure overlay.',
        'fallback_hierarchy': 'secured_cure_overlay',
        'category': 'cure_overlay',
        'status': 'active',
        'calibration_status': 'proxy_not_calibrated_to_internal_workout_tape',
    },
    {
        'group': 'validation_wording',
        'parameter': 'governance_scope_note',
        'value': 'demo_portfolio_not_production_calibration',
        'description': 'Validation outputs are demonstration checks, not production model approval.',
        'fallback_hierarchy': 'not_applicable',
        'category': 'governance',
        'status': 'active',
        'calibration_status': 'not_production_validated',
    },
]

# Documentation-only fallback hierarchy used by governance outputs and docs.
FALLBACK_HIERARCHY = [
    {
        'topic': 'discount_rate',
        'priority': 1,
        'rule': 'Use contract_rate_proxy when available.',
    },
    {
        'topic': 'discount_rate',
        'priority': 2,
        'rule': 'Use cost_of_funds_proxy as floor via max(contract_rate_proxy, cost_of_funds_proxy).',
    },
    {
        'topic': 'discount_rate',
        'priority': 3,
        'rule': 'If one proxy is missing, use available proxy and flag assumption.',
    },
    {
        'topic': 'arrears_proxy',
        'priority': 1,
        'rule': 'Use observed arrears stage from collections tape.',
    },
    {
        'topic': 'arrears_proxy',
        'priority': 2,
        'rule': 'Use proxy arrears stage from lvr,dti,credit_score bands.',
    },
    {
        'topic': 'arrears_proxy',
        'priority': 3,
        'rule': 'If proxy inputs missing, apply conservative late-stage arrears bucket.',
    },
    {
        'topic': 'behaviour_proxy',
        'priority': 1,
        'rule': 'Use observed repayment behaviour score.',
    },
    {
        'topic': 'behaviour_proxy',
        'priority': 2,
        'rule': 'Use proxy behaviour from p&i flag, seasoning, dti, lvr, credit score.',
    },
    {
        'topic': 'behaviour_proxy',
        'priority': 3,
        'rule': 'If behaviour inputs missing, default to neutral/stable proxy.',
    },
    {
        'topic': 'secured_cure_overlay',
        'priority': 1,
        'rule': 'Use observed cure rates from workout history by product/segment.',
    },
    {
        'topic': 'secured_cure_overlay',
        'priority': 2,
        'rule': 'Use simplified secured-product cure overlay proxy.',
    },
    {
        'topic': 'secured_cure_overlay',
        'priority': 3,
        'rule': 'If overlay inputs missing, set cure overlay to zero and disclose limitation.',
    },
]
