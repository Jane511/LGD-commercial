"""
Synthetic data generation for Australian bank LGD modelling.

Generates realistic loan-level default and workout data for three products:
  1. Residential Mortgage
  2. Commercial Cash Flow (PPSR + GSR secured)
  3. Development Finance

Each generator produces:
  - loans DataFrame  (one row per defaulted facility)
  - cashflows DataFrame  (one row per recovery / cost cashflow)

All monetary values in AUD. Dates span 2018-2024 to cover a range of
economic conditions (including COVID stress period).
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

STATES = ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"]
STATE_WEIGHTS = [0.32, 0.26, 0.20, 0.10, 0.06, 0.02, 0.02, 0.02]


def _random_dates(start, end, n, rng):
    delta = (end - start).days
    return [start + timedelta(days=int(rng.uniform(0, delta))) for _ in range(n)]


def _discount(amount, days_from_default, annual_rate):
    return amount / (1 + annual_rate) ** (days_from_default / 365.0)


def _build_discount_rate(contract_rate_proxy, cost_of_funds_proxy):
    """
    Bank-style discount-rate rule used across products.

    discount_rate = max(contract_rate_proxy, cost_of_funds_proxy)
    """
    return np.maximum(contract_rate_proxy, cost_of_funds_proxy)


# ---------------------------------------------------------------------------
# 1. RESIDENTIAL MORTGAGE
# ---------------------------------------------------------------------------

def generate_mortgage_data(n_loans=500, seed=42):
    """
    Generate synthetic residential mortgage default & workout data.

    Features align with Australian Big 4 IRB practice:
    - LTV, credit score, DTI, property type, state, occupancy
    - LMI flag for high-LVR loans
    - Standard vs non-standard classification
    - Multiple recovery types: property sale, borrower cure, LMI claim
    - Cost types: legal, agent commission, property preservation, valuation
    """
    rng = np.random.RandomState(seed)

    # --- Loan origination attributes ---
    loan_ids = np.arange(1, n_loans + 1)
    states = rng.choice(STATES, n_loans, p=STATE_WEIGHTS)
    property_types = rng.choice(
        ["House", "Unit", "Townhouse"],
        n_loans, p=[0.55, 0.35, 0.10]
    )
    occupancy = rng.choice(
        ["Owner-Occupier", "Investor"],
        n_loans, p=[0.65, 0.35]
    )
    loan_type = rng.choice(
        ["P&I", "Interest-Only"],
        n_loans, p=[0.70, 0.30]
    )

    # Property values and LTV
    property_value_orig = rng.uniform(350_000, 2_500_000, n_loans)
    original_ltv = rng.uniform(0.50, 0.95, n_loans)
    original_principal = property_value_orig * original_ltv
    credit_score = rng.normal(680, 60, n_loans).clip(450, 850).astype(int)
    dti = rng.uniform(0.20, 0.55, n_loans)

    # Seasoning (months on book before default)
    seasoning_months = rng.randint(6, 120, n_loans)

    # Default dates spanning 2018-2024
    default_dates = _random_dates(
        datetime(2018, 1, 1), datetime(2024, 6, 30), n_loans, rng
    )

    # Property value at default (indexed; some appreciation, some decline)
    hpi_change = rng.normal(0.02, 0.12, n_loans)  # annual HPI change
    years_on_book = seasoning_months / 12.0
    property_value_at_default = property_value_orig * (1 + hpi_change) ** years_on_book

    # Balance at default (amortised for P&I, less so for IO)
    amort_factor = np.where(
        loan_type == "P&I",
        np.maximum(1 - seasoning_months / 360, 0.60),
        np.maximum(1 - seasoning_months / 600, 0.85)
    )
    ead = original_principal * amort_factor
    accrued_interest = ead * rng.uniform(0.005, 0.02, n_loans)
    ead_total = ead + accrued_interest

    # LTV at default
    ltv_at_default = ead_total / property_value_at_default

    # LMI: taken out at origination if LTV > 80%
    lmi_flag = (original_ltv > 0.80).astype(int)
    lmi_eligible = lmi_flag & (rng.random(n_loans) > 0.15)  # 85% of LMI loans are claim-eligible

    # Standard vs non-standard
    is_non_standard = (
        (credit_score < 600) |
        (dti > 0.45) |
        (original_ltv > 0.90)
    ).astype(int)
    mortgage_class = np.where(is_non_standard, "Non-Standard", "Standard")

    # Discount-rate proxies (bank-style floor logic)
    contract_rate_proxy = rng.uniform(0.035, 0.060, n_loans)
    cost_of_funds_proxy = rng.uniform(0.028, 0.052, n_loans)
    discount_rate = _build_discount_rate(contract_rate_proxy, cost_of_funds_proxy)
    house_price_decline = (
        (property_value_orig - property_value_at_default)
        / np.maximum(property_value_orig, 1)
    ).clip(0, 0.40)
    default_years = np.array([d.year for d in default_dates])
    unemployment_shock = np.where(
        np.isin(default_years, [2020, 2021]),
        0.030,
        np.where(np.isin(default_years, [2022, 2023]), 0.015, 0.010),
    )
    rate_shock = np.maximum(discount_rate - contract_rate_proxy, 0.0)

    # --- Resolution outcomes ---
    # Cure probability: higher for lower LTV, better credit score
    cure_score = (
        -2.0
        + 3.0 * (1 - ltv_at_default.clip(0, 1.5))
        + 1.5 * ((credit_score - 450) / 400)
        - 1.0 * dti
        + rng.normal(0, 0.5, n_loans)
    )
    cure_prob = 1 / (1 + np.exp(-cure_score))
    is_cure = rng.random(n_loans) < cure_prob
    resolution_type = np.where(is_cure, "Cure", "Property Sale")

    # --- Build cashflows ---
    all_cashflows = []
    realised_lgd = np.zeros(n_loans)
    workout_months_arr = np.zeros(n_loans)

    for i in range(n_loans):
        loan_cfs = []
        d_date = default_dates[i]

        if is_cure[i]:
            # Cure: borrower pays arrears over 3-6 months
            cure_months = rng.randint(3, 7)
            workout_months_arr[i] = cure_months
            monthly_payment = ead_total[i] / cure_months
            total_pv_recovery = 0.0
            total_pv_cost = 0.0

            for m in range(1, cure_months + 1):
                cf_date = d_date + timedelta(days=30 * m)
                days = (cf_date - d_date).days
                pv = _discount(monthly_payment, days, discount_rate[i])
                total_pv_recovery += pv
                loan_cfs.append({
                    "loan_id": i + 1, "product": "Mortgage",
                    "cashflow_date": cf_date, "days_from_default": days,
                    "cashflow_type": "Borrower Cure",
                    "cashflow_category": "Recovery",
                    "amount": round(monthly_payment, 2),
                    "amount_pv": round(pv, 2),
                })

            # Small workout cost even for cures
            legal = ead_total[i] * rng.uniform(0.002, 0.008)
            legal_date = d_date + timedelta(days=30 * cure_months)
            legal_days = (legal_date - d_date).days
            legal_pv = _discount(legal, legal_days, discount_rate[i])
            total_pv_cost += legal_pv
            loan_cfs.append({
                "loan_id": i + 1, "product": "Mortgage",
                "cashflow_date": legal_date, "days_from_default": legal_days,
                "cashflow_type": "Legal Cost",
                "cashflow_category": "Cost",
                "amount": round(legal, 2),
                "amount_pv": round(legal_pv, 2),
            })

            econ_loss = ead_total[i] + total_pv_cost - total_pv_recovery
            realised_lgd[i] = max(econ_loss / ead_total[i], 0.0)

        else:
            # Property sale resolution
            workout_months = rng.randint(6, 19)
            workout_months_arr[i] = workout_months

            # Sale price: function of property value at default with distressed discount
            distress_discount = rng.uniform(0.05, 0.20)
            sale_price = property_value_at_default[i] * (1 - distress_discount)
            sale_date = d_date + timedelta(days=30 * workout_months)
            sale_days = (sale_date - d_date).days
            sale_pv = _discount(sale_price, sale_days, discount_rate[i])

            loan_cfs.append({
                "loan_id": i + 1, "product": "Mortgage",
                "cashflow_date": sale_date, "days_from_default": sale_days,
                "cashflow_type": "Property Sale",
                "cashflow_category": "Recovery",
                "amount": round(sale_price, 2),
                "amount_pv": round(sale_pv, 2),
            })
            total_pv_recovery = sale_pv

            # LMI claim (if eligible and there is a shortfall)
            lmi_recovery_pv = 0.0
            if lmi_eligible[i] and sale_price < ead_total[i]:
                shortfall = ead_total[i] - sale_price
                lmi_claim = shortfall * rng.uniform(0.70, 0.95)
                lmi_date = sale_date + timedelta(days=rng.randint(60, 180))
                lmi_days = (lmi_date - d_date).days
                lmi_pv = _discount(lmi_claim, lmi_days, discount_rate[i])
                lmi_recovery_pv = lmi_pv
                total_pv_recovery += lmi_pv
                loan_cfs.append({
                    "loan_id": i + 1, "product": "Mortgage",
                    "cashflow_date": lmi_date, "days_from_default": lmi_days,
                    "cashflow_type": "LMI Claim",
                    "cashflow_category": "Recovery",
                    "amount": round(lmi_claim, 2),
                    "amount_pv": round(lmi_pv, 2),
                })

            # Costs
            total_pv_cost = 0.0
            # Legal
            legal = ead_total[i] * rng.uniform(0.01, 0.03)
            legal_date = d_date + timedelta(days=30 * max(workout_months - 2, 1))
            legal_days = (legal_date - d_date).days
            legal_pv = _discount(legal, legal_days, discount_rate[i])
            total_pv_cost += legal_pv
            loan_cfs.append({
                "loan_id": i + 1, "product": "Mortgage",
                "cashflow_date": legal_date, "days_from_default": legal_days,
                "cashflow_type": "Legal Cost",
                "cashflow_category": "Cost",
                "amount": round(legal, 2),
                "amount_pv": round(legal_pv, 2),
            })

            # Agent commission (% of sale price)
            agent = sale_price * rng.uniform(0.02, 0.03)
            agent_pv = _discount(agent, sale_days, discount_rate[i])
            total_pv_cost += agent_pv
            loan_cfs.append({
                "loan_id": i + 1, "product": "Mortgage",
                "cashflow_date": sale_date, "days_from_default": sale_days,
                "cashflow_type": "Agent Commission",
                "cashflow_category": "Cost",
                "amount": round(agent, 2),
                "amount_pv": round(agent_pv, 2),
            })

            # Property preservation
            preservation = ead_total[i] * rng.uniform(0.005, 0.015)
            pres_date = d_date + timedelta(days=30 * (workout_months // 2))
            pres_days = (pres_date - d_date).days
            pres_pv = _discount(preservation, pres_days, discount_rate[i])
            total_pv_cost += pres_pv
            loan_cfs.append({
                "loan_id": i + 1, "product": "Mortgage",
                "cashflow_date": pres_date, "days_from_default": pres_days,
                "cashflow_type": "Property Preservation",
                "cashflow_category": "Cost",
                "amount": round(preservation, 2),
                "amount_pv": round(pres_pv, 2),
            })

            # Valuation
            val_cost = rng.uniform(500, 3000)
            val_date = d_date + timedelta(days=rng.randint(14, 60))
            val_days = (val_date - d_date).days
            val_pv = _discount(val_cost, val_days, discount_rate[i])
            total_pv_cost += val_pv
            loan_cfs.append({
                "loan_id": i + 1, "product": "Mortgage",
                "cashflow_date": val_date, "days_from_default": val_days,
                "cashflow_type": "Valuation Fee",
                "cashflow_category": "Cost",
                "amount": round(val_cost, 2),
                "amount_pv": round(val_pv, 2),
            })

            econ_loss = ead_total[i] + total_pv_cost - total_pv_recovery
            realised_lgd[i] = max(econ_loss / ead_total[i], 0.0)

        all_cashflows.extend(loan_cfs)

    # --- Assemble loans DataFrame ---
    loans = pd.DataFrame({
        "loan_id": loan_ids,
        "product": "Mortgage",
        "state": states,
        "property_type": property_types,
        "occupancy": occupancy,
        "loan_type": loan_type,
        "mortgage_class": mortgage_class,
        "property_value_orig": property_value_orig.round(2),
        "original_ltv": original_ltv.round(4),
        "original_principal": original_principal.round(2),
        "credit_score": credit_score,
        "dti": dti.round(4),
        "seasoning_months": seasoning_months,
        "default_date": default_dates,
        "property_value_at_default": property_value_at_default.round(2),
        "ead": ead_total.round(2),
        "ltv_at_default": ltv_at_default.round(4),
        "lmi_flag": lmi_flag,
        "lmi_eligible": lmi_eligible.astype(int),
        "contract_rate_proxy": contract_rate_proxy.round(4),
        "cost_of_funds_proxy": cost_of_funds_proxy.round(4),
        "discount_rate": discount_rate.round(4),
        "house_price_decline": house_price_decline.round(4),
        "unemployment_shock": unemployment_shock.round(4),
        "rate_shock": rate_shock.round(4),
        "resolution_type": resolution_type,
        "workout_months": workout_months_arr.astype(int),
        "realised_lgd": realised_lgd.round(6),
    })

    cashflows = pd.DataFrame(all_cashflows)
    return loans, cashflows


# ---------------------------------------------------------------------------
# 2. COMMERCIAL CASH FLOW LENDING (PPSR + GSR)
# ---------------------------------------------------------------------------

INDUSTRIES = [
    "Agriculture", "Manufacturing", "Retail Trade", "Construction",
    "Transport", "Professional Services", "Accommodation & Food",
    "Health Care", "Mining", "Wholesale Trade",
    "Education & Training", "Financial Services", "Information Technology",
    "Real Estate", "Arts & Recreation", "Utilities",
]
INDUSTRY_WEIGHTS = [
    0.09, 0.10, 0.10, 0.08, 0.06, 0.09, 0.07, 0.06, 0.05, 0.06,
    0.04, 0.05, 0.04, 0.04, 0.03, 0.04,
]

# Industry risk profiles from industry analysis scorecard (1-5 scale)
# Mining has no match in industry analysis -- conservative Elevated default
INDUSTRY_RISK_PROFILES = {
    "Agriculture":           {"risk_score": 3.50, "risk_level": "Elevated", "debt_to_ebitda": 2.9, "icr": 3.6},
    "Manufacturing":         {"risk_score": 3.50, "risk_level": "Elevated", "debt_to_ebitda": 3.2, "icr": 3.1},
    "Wholesale Trade":       {"risk_score": 3.23, "risk_level": "Elevated", "debt_to_ebitda": 3.5, "icr": 2.9},
    "Retail Trade":          {"risk_score": 3.23, "risk_level": "Elevated", "debt_to_ebitda": 3.1, "icr": 3.2},
    "Accommodation & Food":  {"risk_score": 2.68, "risk_level": "Medium",   "debt_to_ebitda": 2.7, "icr": 3.7},
    "Construction":          {"risk_score": 2.68, "risk_level": "Medium",   "debt_to_ebitda": 3.3, "icr": 3.1},
    "Health Care":           {"risk_score": 2.22, "risk_level": "Medium",   "debt_to_ebitda": 2.1, "icr": 4.5},
    "Professional Services": {"risk_score": 2.18, "risk_level": "Medium",   "debt_to_ebitda": 2.3, "icr": 4.3},
    "Transport":             {"risk_score": 2.14, "risk_level": "Medium",   "debt_to_ebitda": 2.6, "icr": 3.8},
    "Mining":                {"risk_score": 3.50, "risk_level": "Elevated", "debt_to_ebitda": 3.5, "icr": 2.9},
    # New industries aligned with PD Scorecard project
    "Education & Training":  {"risk_score": 2.30, "risk_level": "Medium",   "debt_to_ebitda": 1.8, "icr": 5.0},
    "Financial Services":    {"risk_score": 2.45, "risk_level": "Medium",   "debt_to_ebitda": 2.5, "icr": 4.0},
    "Information Technology": {"risk_score": 2.55, "risk_level": "Medium",   "debt_to_ebitda": 2.0, "icr": 4.8},
    "Real Estate":           {"risk_score": 2.90, "risk_level": "Medium",   "debt_to_ebitda": 3.8, "icr": 2.7},
    "Arts & Recreation":     {"risk_score": 3.10, "risk_level": "Elevated", "debt_to_ebitda": 2.4, "icr": 3.3},
    "Utilities":             {"risk_score": 1.90, "risk_level": "Low",      "debt_to_ebitda": 3.0, "icr": 3.5},
}

# ---------------------------------------------------------------------------
# CASH FLOW LENDING PRODUCTS (aligned with PD Scorecard project)
# ---------------------------------------------------------------------------

CASHFLOW_PRODUCTS = [
    "Business Term Loan",
    "Working Capital Facility",
    "Trade Finance",
    "Equipment Finance",
    "Invoice Finance",
    "Merchant Cash Advance",
    "Business Line of Credit",
    "Professional Practice Loan",
]
CASHFLOW_PRODUCT_WEIGHTS = [0.20, 0.18, 0.10, 0.15, 0.10, 0.05, 0.12, 0.10]

# PD score bands from WoE logistic regression scorecard
PD_SCORE_BANDS = {
    "A": {"pd_range": (0.000, 0.025), "label": "Very Low"},
    "B": {"pd_range": (0.025, 0.050), "label": "Low"},
    "C": {"pd_range": (0.050, 0.075), "label": "Medium"},
    "D": {"pd_range": (0.075, 0.100), "label": "High"},
    "E": {"pd_range": (0.100, 0.250), "label": "Very High"},
}

# Conduct classification for overlay
CONDUCT_CLASSES = ["Green", "Amber", "Red"]
CONDUCT_WEIGHTS = [0.70, 0.20, 0.10]

# Development type -> industry mapping for risk score lookup
_DEV_TYPE_INDUSTRY = {
    "Residential Apartments": "Construction",
    "Residential Houses/Lots": "Construction",
    "Commercial Office": "Professional Services",
    "Mixed-Use": "Construction",
    "Industrial": "Manufacturing",
}


def generate_commercial_data(n_loans=300, seed=43):
    """
    Generate synthetic commercial cash-flow lending default & workout data.

    Features aligned with Australian bank practice:
    - Borrower financials (revenue, EBITDA, leverage, ICR)
    - Security types: property, PPSR (P&E, vehicles, receivables, inventory), GSR
    - Facility structure: term loan, revolving, overdraft
    - Multiple recovery streams by security type
    - Receiver/administrator costs
    """
    rng = np.random.RandomState(seed)

    loan_ids = np.arange(1, n_loans + 1)
    states = rng.choice(STATES, n_loans, p=STATE_WEIGHTS)
    industries = rng.choice(INDUSTRIES, n_loans, p=INDUSTRY_WEIGHTS)

    # Borrower characteristics
    annual_revenue = rng.lognormal(np.log(5_000_000), 0.8, n_loans).clip(500_000, 200_000_000)
    ebitda_margin = rng.uniform(0.05, 0.25, n_loans)
    ebitda = annual_revenue * ebitda_margin
    years_in_business = rng.randint(2, 40, n_loans)

    # Facility details
    facility_types = rng.choice(
        ["Term Loan", "Revolving Credit", "Overdraft"],
        n_loans, p=[0.55, 0.30, 0.15]
    )
    facility_limit = rng.uniform(200_000, 15_000_000, n_loans)
    drawn_pct = np.where(
        facility_types == "Term Loan",
        rng.uniform(0.80, 1.00, n_loans),
        rng.uniform(0.50, 0.95, n_loans),
    )
    drawn_balance = facility_limit * drawn_pct
    undrawn = facility_limit - drawn_balance

    # CCF for undrawn portion
    ccf = np.where(facility_types == "Term Loan", 1.0,
           np.where(facility_types == "Revolving Credit",
                    rng.uniform(0.50, 0.75, n_loans),
                    rng.uniform(0.75, 1.00, n_loans)))
    ead = drawn_balance + ccf * undrawn

    # Leverage and coverage
    total_debt = ead * rng.uniform(1.0, 2.5, n_loans)
    leverage = total_debt / np.maximum(ebitda, 1)
    icr = ebitda / np.maximum(ead * rng.uniform(0.04, 0.08, n_loans), 1)

    seniority = rng.choice(
        ["Senior Secured", "Senior Unsecured"],
        n_loans, p=[0.80, 0.20]
    )
    has_director_guarantee = (rng.random(n_loans) > 0.30).astype(int)

    # Security structure
    security_types = rng.choice(
        ["Property", "PPSR - P&E", "PPSR - Receivables", "PPSR - Mixed", "GSR Only"],
        n_loans, p=[0.30, 0.15, 0.15, 0.25, 0.15]
    )
    # Security coverage ratio
    base_coverage = np.where(
        security_types == "Property", rng.uniform(0.70, 1.40, n_loans),
        np.where(security_types == "GSR Only", rng.uniform(0.20, 0.60, n_loans),
                 rng.uniform(0.40, 1.10, n_loans))
    )
    collateral_value = ead * base_coverage

    # Default
    default_dates = _random_dates(
        datetime(2018, 1, 1), datetime(2024, 6, 30), n_loans, rng
    )
    default_triggers = rng.choice(
        ["90 DPD", "Covenant Breach", "Voluntary Administration", "Receivership"],
        n_loans, p=[0.35, 0.25, 0.25, 0.15]
    )
    contract_rate_proxy = rng.uniform(0.055, 0.085, n_loans)
    cost_of_funds_proxy = rng.uniform(0.045, 0.070, n_loans)
    discount_rate = _build_discount_rate(contract_rate_proxy, cost_of_funds_proxy)

    # Resolution
    resolution_strategies = rng.choice(
        ["Receivership", "Voluntary Administration", "Workout", "Write-off"],
        n_loans, p=[0.30, 0.25, 0.35, 0.10]
    )

    # Industry risk scores for each loan
    industry_risk_scores = np.array([
        INDUSTRY_RISK_PROFILES[ind]["risk_score"] for ind in industries
    ])

    # --- Build cashflows and compute LGD ---
    all_cashflows = []
    realised_lgd = np.zeros(n_loans)
    workout_months_arr = np.zeros(n_loans, dtype=int)

    for i in range(n_loans):
        loan_cfs = []
        d_date = default_dates[i]
        workout_months = rng.randint(12, 37)
        workout_months_arr[i] = workout_months
        total_pv_recovery = 0.0
        total_pv_cost = 0.0

        # Industry-sensitive recovery penalty: higher-risk industries get deeper discounts
        ind_penalty = 0.02 * max(industry_risk_scores[i] - 2.0, 0)

        # --- Recoveries by security type ---
        sec = security_types[i]

        if sec == "Property":
            # Property sale (industry risk widens distress discount)
            sale_discount = rng.uniform(0.08 + ind_penalty, 0.25 + ind_penalty)
            sale_amt = collateral_value[i] * (1 - sale_discount)
            sale_month = rng.randint(10, workout_months + 1)
            sale_date = d_date + timedelta(days=30 * sale_month)
            sale_days = (sale_date - d_date).days
            sale_pv = _discount(sale_amt, sale_days, discount_rate[i])
            total_pv_recovery += sale_pv
            loan_cfs.append({
                "loan_id": i + 1, "product": "Commercial",
                "cashflow_date": sale_date, "days_from_default": sale_days,
                "cashflow_type": "Property Sale",
                "cashflow_category": "Recovery",
                "amount": round(sale_amt, 2),
                "amount_pv": round(sale_pv, 2),
            })

        elif sec in ("PPSR - P&E", "PPSR - Receivables", "PPSR - Mixed"):
            # PPSR asset realisations (industry risk reduces recovery rate)
            n_tranches = rng.randint(3, 8)
            total_ppsr = collateral_value[i] * rng.uniform(
                max(0.30 - ind_penalty, 0.15), max(0.65 - ind_penalty, 0.35)
            )
            tranche_amts = rng.dirichlet(np.ones(n_tranches)) * total_ppsr
            for t_idx, t_amt in enumerate(tranche_amts):
                t_month = rng.randint(3, workout_months + 1)
                t_date = d_date + timedelta(days=30 * t_month)
                t_days = (t_date - d_date).days
                t_pv = _discount(t_amt, t_days, discount_rate[i])
                total_pv_recovery += t_pv

                cf_type_map = {
                    "PPSR - P&E": "PPSR Plant & Equipment",
                    "PPSR - Receivables": "PPSR Receivables Collection",
                    "PPSR - Mixed": rng.choice(["PPSR Plant & Equipment", "PPSR Receivables Collection", "PPSR Inventory"]),
                }
                loan_cfs.append({
                    "loan_id": i + 1, "product": "Commercial",
                    "cashflow_date": t_date, "days_from_default": t_days,
                    "cashflow_type": cf_type_map[sec],
                    "cashflow_category": "Recovery",
                    "amount": round(t_amt, 2),
                    "amount_pv": round(t_pv, 2),
                })

        else:  # GSR Only
            # GSR sweep - lower recovery (industry risk further reduces)
            gsr_amt = collateral_value[i] * rng.uniform(
                max(0.25 - ind_penalty, 0.10), max(0.55 - ind_penalty, 0.30)
            )
            gsr_month = rng.randint(12, workout_months + 1)
            gsr_date = d_date + timedelta(days=30 * gsr_month)
            gsr_days = (gsr_date - d_date).days
            gsr_pv = _discount(gsr_amt, gsr_days, discount_rate[i])
            total_pv_recovery += gsr_pv
            loan_cfs.append({
                "loan_id": i + 1, "product": "Commercial",
                "cashflow_date": gsr_date, "days_from_default": gsr_days,
                "cashflow_type": "GSR All-Assets Sweep",
                "cashflow_category": "Recovery",
                "amount": round(gsr_amt, 2),
                "amount_pv": round(gsr_pv, 2),
            })

        # Director guarantee recovery
        if has_director_guarantee[i]:
            guar_recovery_rate = rng.uniform(0.05, 0.30)
            guar_amt = ead[i] * guar_recovery_rate * rng.uniform(0.3, 0.8)
            guar_month = rng.randint(6, workout_months + 1)
            guar_date = d_date + timedelta(days=30 * guar_month)
            guar_days = (guar_date - d_date).days
            guar_pv = _discount(guar_amt, guar_days, discount_rate[i])
            total_pv_recovery += guar_pv
            loan_cfs.append({
                "loan_id": i + 1, "product": "Commercial",
                "cashflow_date": guar_date, "days_from_default": guar_days,
                "cashflow_type": "Director Guarantee",
                "cashflow_category": "Recovery",
                "amount": round(guar_amt, 2),
                "amount_pv": round(guar_pv, 2),
            })

        # --- Costs ---
        # Receiver / administrator fees
        recv_fee = ead[i] * rng.uniform(0.03, 0.08)
        recv_date = d_date + timedelta(days=30 * workout_months)
        recv_days = (recv_date - d_date).days
        recv_pv = _discount(recv_fee, recv_days, discount_rate[i])
        total_pv_cost += recv_pv
        loan_cfs.append({
            "loan_id": i + 1, "product": "Commercial",
            "cashflow_date": recv_date, "days_from_default": recv_days,
            "cashflow_type": "Receiver/Administrator Fee",
            "cashflow_category": "Cost",
            "amount": round(recv_fee, 2),
            "amount_pv": round(recv_pv, 2),
        })

        # Legal
        legal = ead[i] * rng.uniform(0.02, 0.05)
        legal_month = rng.randint(2, max(workout_months - 2, 3))
        legal_date = d_date + timedelta(days=30 * legal_month)
        legal_days = (legal_date - d_date).days
        legal_pv = _discount(legal, legal_days, discount_rate[i])
        total_pv_cost += legal_pv
        loan_cfs.append({
            "loan_id": i + 1, "product": "Commercial",
            "cashflow_date": legal_date, "days_from_default": legal_days,
            "cashflow_type": "Legal Cost",
            "cashflow_category": "Cost",
            "amount": round(legal, 2),
            "amount_pv": round(legal_pv, 2),
        })

        # Valuation
        val_cost = rng.uniform(2000, 15000)
        val_date = d_date + timedelta(days=rng.randint(14, 60))
        val_days = (val_date - d_date).days
        val_pv = _discount(val_cost, val_days, discount_rate[i])
        total_pv_cost += val_pv
        loan_cfs.append({
            "loan_id": i + 1, "product": "Commercial",
            "cashflow_date": val_date, "days_from_default": val_days,
            "cashflow_type": "Valuation Fee",
            "cashflow_category": "Cost",
            "amount": round(val_cost, 2),
            "amount_pv": round(val_pv, 2),
        })

        # Asset management (during receivership)
        if resolution_strategies[i] in ("Receivership", "Voluntary Administration"):
            mgmt = ead[i] * rng.uniform(0.01, 0.03)
            mgmt_date = d_date + timedelta(days=30 * (workout_months // 2))
            mgmt_days = (mgmt_date - d_date).days
            mgmt_pv = _discount(mgmt, mgmt_days, discount_rate[i])
            total_pv_cost += mgmt_pv
            loan_cfs.append({
                "loan_id": i + 1, "product": "Commercial",
                "cashflow_date": mgmt_date, "days_from_default": mgmt_days,
                "cashflow_type": "Asset Management Cost",
                "cashflow_category": "Cost",
                "amount": round(mgmt, 2),
                "amount_pv": round(mgmt_pv, 2),
            })

        econ_loss = ead[i] + total_pv_cost - total_pv_recovery
        realised_lgd[i] = max(econ_loss / ead[i], 0.0)

        all_cashflows.extend(loan_cfs)

    # Industry risk columns
    industry_risk_levels = np.array([
        INDUSTRY_RISK_PROFILES[ind]["risk_level"] for ind in industries
    ])
    industry_debt_to_ebitda = np.array([
        INDUSTRY_RISK_PROFILES[ind]["debt_to_ebitda"] for ind in industries
    ])
    industry_icr_bench = np.array([
        INDUSTRY_RISK_PROFILES[ind]["icr"] for ind in industries
    ])

    loans = pd.DataFrame({
        "loan_id": loan_ids,
        "product": "Commercial",
        "state": states,
        "industry": industries,
        "annual_revenue": annual_revenue.round(2),
        "ebitda": ebitda.round(2),
        "leverage_ratio": leverage.round(2),
        "interest_coverage_ratio": icr.round(2),
        "years_in_business": years_in_business,
        "facility_type": facility_types,
        "facility_limit": facility_limit.round(2),
        "drawn_balance": drawn_balance.round(2),
        "undrawn_amount": undrawn.round(2),
        "ccf": ccf.round(4),
        "ead": ead.round(2),
        "seniority": seniority,
        "has_director_guarantee": has_director_guarantee,
        "security_type": security_types,
        "collateral_value": collateral_value.round(2),
        "security_coverage_ratio": base_coverage.round(4),
        "default_date": default_dates,
        "default_trigger": default_triggers,
        "contract_rate_proxy": contract_rate_proxy.round(4),
        "cost_of_funds_proxy": cost_of_funds_proxy.round(4),
        "discount_rate": discount_rate.round(4),
        "resolution_strategy": resolution_strategies,
        "workout_months": workout_months_arr,
        "realised_lgd": realised_lgd.round(6),
        "industry_risk_score": industry_risk_scores.round(2),
        "industry_risk_level": industry_risk_levels,
        "industry_debt_to_ebitda_benchmark": industry_debt_to_ebitda,
        "industry_icr_benchmark": industry_icr_bench,
    })

    cashflows = pd.DataFrame(all_cashflows)
    return loans, cashflows


# ---------------------------------------------------------------------------
# 3. DEVELOPMENT FINANCE
# ---------------------------------------------------------------------------

def generate_development_data(n_loans=200, seed=44):
    """
    Generate synthetic development finance default & workout data.

    Features aligned with Australian bank practice:
    - Completion stage at default (primary LGD driver)
    - Development type, pre-sale coverage, TDC, GRV
    - Fund-to-complete vs sell-as-is decision
    - Cost-to-complete as dominant cost
    - Scenario-based recovery (by completion stage)
    """
    rng = np.random.RandomState(seed)

    loan_ids = np.arange(1, n_loans + 1)
    states = rng.choice(STATES, n_loans, p=STATE_WEIGHTS)

    dev_types = rng.choice(
        ["Residential Apartments", "Residential Houses/Lots",
         "Commercial Office", "Mixed-Use", "Industrial"],
        n_loans, p=[0.35, 0.25, 0.15, 0.15, 0.10]
    )

    # Map development type to industry and look up risk scores
    dev_industries = np.array([_DEV_TYPE_INDUSTRY[dt] for dt in dev_types])
    dev_industry_risk_scores = np.array([
        INDUSTRY_RISK_PROFILES[ind]["risk_score"] for ind in dev_industries
    ])
    unit_count = np.where(
        np.isin(dev_types, ["Residential Apartments", "Mixed-Use"]),
        rng.randint(8, 120, n_loans),
        np.where(dev_types == "Residential Houses/Lots",
                 rng.randint(5, 60, n_loans),
                 rng.randint(1, 10, n_loans))
    )

    # Project financials
    tdc = rng.uniform(3_000_000, 80_000_000, n_loans)  # total development cost
    grv = tdc * rng.uniform(1.15, 1.60, n_loans)  # gross realisation value
    land_value = tdc * rng.uniform(0.25, 0.45, n_loans)

    # Facility
    facility_limit = tdc * rng.uniform(0.55, 0.80, n_loans)
    ltc = facility_limit / tdc  # loan to cost

    # Completion stage at default
    completion_pct = rng.choice(
        [0.0, 0.15, 0.45, 0.75, 1.0],
        n_loans, p=[0.10, 0.20, 0.30, 0.25, 0.15]
    )
    # Add noise
    completion_pct = (completion_pct + rng.uniform(-0.08, 0.08, n_loans)).clip(0, 1)

    completion_stage = np.where(
        completion_pct == 0, "Pre-Construction",
        np.where(completion_pct < 0.30, "Early Construction",
        np.where(completion_pct < 0.70, "Mid-Construction",
        np.where(completion_pct < 0.95, "Near-Complete",
                 "Complete Unsold")))
    )

    # Drawn balance (progressive drawdown)
    drawn_pct = np.where(
        completion_stage == "Pre-Construction", rng.uniform(0.30, 0.45, n_loans),
        np.where(completion_stage == "Early Construction", rng.uniform(0.40, 0.60, n_loans),
        np.where(completion_stage == "Mid-Construction", rng.uniform(0.60, 0.80, n_loans),
        np.where(completion_stage == "Near-Complete", rng.uniform(0.80, 0.95, n_loans),
                 rng.uniform(0.90, 1.00, n_loans))))
    )
    drawn_balance = facility_limit * drawn_pct
    capitalised_interest = drawn_balance * rng.uniform(0.03, 0.08, n_loans)
    ead = drawn_balance + capitalised_interest

    # Pre-sales
    presale_coverage = np.where(
        completion_stage == "Pre-Construction", rng.uniform(0.0, 0.50, n_loans),
        np.where(completion_stage == "Early Construction", rng.uniform(0.20, 0.70, n_loans),
        np.where(completion_stage == "Mid-Construction", rng.uniform(0.40, 0.90, n_loans),
                 rng.uniform(0.50, 1.0, n_loans)))
    )
    presale_value = grv * presale_coverage

    # Cost to complete
    cost_to_complete = tdc * (1 - completion_pct) * rng.uniform(0.90, 1.15, n_loans)

    # As-is value
    as_is_value = np.where(
        completion_stage == "Pre-Construction", land_value,
        np.where(completion_stage == "Early Construction",
                 land_value + (grv - land_value) * completion_pct * rng.uniform(0.4, 0.7, n_loans),
        np.where(completion_stage == "Mid-Construction",
                 land_value + (grv - land_value) * completion_pct * rng.uniform(0.5, 0.8, n_loans),
        np.where(completion_stage == "Near-Complete",
                 grv * rng.uniform(0.75, 0.90, n_loans),
                 grv * rng.uniform(0.85, 0.95, n_loans))))
    )
    lvr_as_if_complete = facility_limit / grv

    # Default details
    default_dates = _random_dates(
        datetime(2018, 1, 1), datetime(2024, 6, 30), n_loans, rng
    )
    default_triggers = rng.choice(
        ["Builder Insolvency", "Cost Overrun", "Pre-sale Shortfall",
         "Interest Reserve Exhausted", "90 DPD"],
        n_loans, p=[0.25, 0.20, 0.20, 0.15, 0.20]
    )
    contract_rate_proxy = rng.uniform(0.075, 0.105, n_loans)
    cost_of_funds_proxy = rng.uniform(0.055, 0.085, n_loans)
    discount_rate = _build_discount_rate(contract_rate_proxy, cost_of_funds_proxy)
    grv_decline = ((grv - as_is_value) / np.maximum(grv, 1)).clip(0, 0.60)

    # Fund-to-complete decision
    # More likely when near-complete (cheaper to finish)
    ftc_prob = np.where(
        completion_stage == "Pre-Construction", 0.10,
        np.where(completion_stage == "Early Construction", 0.25,
        np.where(completion_stage == "Mid-Construction", 0.55,
        np.where(completion_stage == "Near-Complete", 0.85,
                 0.0)))  # complete unsold: nothing to build
    )
    fund_to_complete = rng.random(n_loans) < ftc_prob

    # --- Build cashflows and compute LGD ---
    all_cashflows = []
    realised_lgd = np.zeros(n_loans)
    workout_months_arr = np.zeros(n_loans, dtype=int)

    for i in range(n_loans):
        loan_cfs = []
        d_date = default_dates[i]
        total_pv_recovery = 0.0
        total_pv_cost = 0.0

        if fund_to_complete[i] and completion_stage[i] != "Complete Unsold":
            # --- FUND TO COMPLETE scenario ---
            construction_months = int((1 - completion_pct[i]) * rng.uniform(8, 18))
            sales_months = int(rng.randint(6, 18))
            workout_months = construction_months + sales_months
            workout_months_arr[i] = workout_months

            # Cost to complete (dominant cost)
            ctc = cost_to_complete[i]
            ctc_overrun = ctc * rng.uniform(1.0, 1.20)  # potential overrun
            # Spread cost over construction period
            monthly_ctc = ctc_overrun / max(construction_months, 1)
            for m in range(1, construction_months + 1):
                cf_date = d_date + timedelta(days=30 * m)
                cf_days = (cf_date - d_date).days
                pv = _discount(monthly_ctc, cf_days, discount_rate[i])
                total_pv_cost += pv
                loan_cfs.append({
                    "loan_id": i + 1, "product": "Development",
                    "cashflow_date": cf_date, "days_from_default": cf_days,
                    "cashflow_type": "Cost to Complete",
                    "cashflow_category": "Cost",
                    "amount": round(monthly_ctc, 2),
                    "amount_pv": round(pv, 2),
                })

            # Sale of completed units
            market_stress = rng.uniform(0.85, 1.05)  # market may have moved
            total_sales = grv[i] * market_stress
            # Some pre-sales settle, some don't (sunset risk)
            presale_rescission_rate = rng.uniform(0.0, 0.20)
            effective_presales = presale_value[i] * (1 - presale_rescission_rate)
            open_market_sales = max(total_sales - effective_presales, 0)

            # Pre-sale settlements (around completion)
            if effective_presales > 0:
                ps_date = d_date + timedelta(days=30 * (construction_months + 2))
                ps_days = (ps_date - d_date).days
                ps_pv = _discount(effective_presales, ps_days, discount_rate[i])
                total_pv_recovery += ps_pv
                loan_cfs.append({
                    "loan_id": i + 1, "product": "Development",
                    "cashflow_date": ps_date, "days_from_default": ps_days,
                    "cashflow_type": "Pre-sale Settlement",
                    "cashflow_category": "Recovery",
                    "amount": round(effective_presales, 2),
                    "amount_pv": round(ps_pv, 2),
                })

            # Open market sales (spread over sales period)
            if open_market_sales > 0:
                n_sale_tranches = min(rng.randint(3, 8), max(sales_months // 2, 1))
                tranche_amts = rng.dirichlet(np.ones(n_sale_tranches)) * open_market_sales
                for t_idx, t_amt in enumerate(tranche_amts):
                    t_month = construction_months + int(rng.randint(1, sales_months + 1))
                    t_date = d_date + timedelta(days=int(30 * t_month))
                    t_days = (t_date - d_date).days
                    t_pv = _discount(t_amt, t_days, discount_rate[i])
                    total_pv_recovery += t_pv
                    loan_cfs.append({
                        "loan_id": i + 1, "product": "Development",
                        "cashflow_date": t_date, "days_from_default": t_days,
                        "cashflow_type": "Open Market Sale",
                        "cashflow_category": "Recovery",
                        "amount": round(t_amt, 2),
                        "amount_pv": round(t_pv, 2),
                    })

        else:
            # --- SELL AS-IS scenario ---
            if completion_stage[i] == "Complete Unsold":
                # Sell completed units directly
                workout_months = int(rng.randint(6, 18))
                distress = rng.uniform(0.05, 0.20)
                sale_value = grv[i] * (1 - distress)
            elif completion_stage[i] == "Pre-Construction":
                # Sell land
                workout_months = int(rng.randint(6, 12))
                distress = rng.uniform(0.10, 0.30)
                sale_value = land_value[i] * (1 - distress)
            else:
                # Sell partially built (worst case)
                workout_months = int(rng.randint(9, 24))
                distress = rng.uniform(0.20, 0.45)
                sale_value = as_is_value[i] * (1 - distress)

            workout_months_arr[i] = workout_months
            sale_date = d_date + timedelta(days=int(30 * workout_months))
            sale_days = (sale_date - d_date).days
            sale_pv = _discount(sale_value, sale_days, discount_rate[i])
            total_pv_recovery += sale_pv
            loan_cfs.append({
                "loan_id": i + 1, "product": "Development",
                "cashflow_date": sale_date, "days_from_default": sale_days,
                "cashflow_type": "As-Is Sale" if completion_stage[i] != "Complete Unsold" else "Completed Unit Sales",
                "cashflow_category": "Recovery",
                "amount": round(sale_value, 2),
                "amount_pv": round(sale_pv, 2),
            })

        # --- Common costs ---
        # Receiver fees
        recv_pct = rng.uniform(0.03, 0.05)
        recv = ead[i] * recv_pct
        recv_date = d_date + timedelta(days=int(30 * workout_months_arr[i]))
        recv_days = (recv_date - d_date).days
        recv_pv = _discount(recv, recv_days, discount_rate[i])
        total_pv_cost += recv_pv
        loan_cfs.append({
            "loan_id": i + 1, "product": "Development",
            "cashflow_date": recv_date, "days_from_default": recv_days,
            "cashflow_type": "Receiver Fee",
            "cashflow_category": "Cost",
            "amount": round(recv, 2),
            "amount_pv": round(recv_pv, 2),
        })

        # Legal
        legal = ead[i] * rng.uniform(0.02, 0.05)
        legal_month = int(rng.randint(1, max(int(workout_months_arr[i] // 2), 2)))
        legal_date = d_date + timedelta(days=int(30 * legal_month))
        legal_days = (legal_date - d_date).days
        legal_pv = _discount(legal, legal_days, discount_rate[i])
        total_pv_cost += legal_pv
        loan_cfs.append({
            "loan_id": i + 1, "product": "Development",
            "cashflow_date": legal_date, "days_from_default": legal_days,
            "cashflow_type": "Legal Cost",
            "cashflow_category": "Cost",
            "amount": round(legal, 2),
            "amount_pv": round(legal_pv, 2),
        })

        # Holding costs (rates, insurance, security)
        holding_monthly = ead[i] * rng.uniform(0.002, 0.005)
        for m in range(1, int(workout_months_arr[i]) + 1):
            h_date = d_date + timedelta(days=int(30 * m))
            h_days = (h_date - d_date).days
            h_pv = _discount(holding_monthly, h_days, discount_rate[i])
            total_pv_cost += h_pv
            loan_cfs.append({
                "loan_id": i + 1, "product": "Development",
                "cashflow_date": h_date, "days_from_default": h_days,
                "cashflow_type": "Holding Cost",
                "cashflow_category": "Cost",
                "amount": round(holding_monthly, 2),
                "amount_pv": round(h_pv, 2),
            })

        # Marketing (for fund-to-complete or complete-unsold)
        if fund_to_complete[i] or completion_stage[i] == "Complete Unsold":
            mktg = grv[i] * rng.uniform(0.02, 0.04)
            mktg_month = int(workout_months_arr[i]) - int(rng.randint(1, 4))
            mktg_date = d_date + timedelta(days=int(30 * max(mktg_month, 1)))
            mktg_days = (mktg_date - d_date).days
            mktg_pv = _discount(mktg, mktg_days, discount_rate[i])
            total_pv_cost += mktg_pv
            loan_cfs.append({
                "loan_id": i + 1, "product": "Development",
                "cashflow_date": mktg_date, "days_from_default": mktg_days,
                "cashflow_type": "Marketing & Sales Cost",
                "cashflow_category": "Cost",
                "amount": round(mktg, 2),
                "amount_pv": round(mktg_pv, 2),
            })

        econ_loss = ead[i] + total_pv_cost - total_pv_recovery
        realised_lgd[i] = max(econ_loss / ead[i], 0.0)

        all_cashflows.extend(loan_cfs)

    # Industry risk columns for development loans
    dev_industry_risk_levels = np.array([
        INDUSTRY_RISK_PROFILES[ind]["risk_level"] for ind in dev_industries
    ])

    loans = pd.DataFrame({
        "loan_id": loan_ids,
        "product": "Development",
        "state": states,
        "development_type": dev_types,
        "industry": dev_industries,
        "unit_count": unit_count,
        "tdc": tdc.round(2),
        "grv": grv.round(2),
        "land_value": land_value.round(2),
        "as_is_value": as_is_value.round(2),
        "as_if_complete_value": grv.round(2),
        "facility_limit": facility_limit.round(2),
        "ltc": ltc.round(4),
        "lvr_as_if_complete": lvr_as_if_complete.round(4),
        "drawn_balance": drawn_balance.round(2),
        "capitalised_interest": capitalised_interest.round(2),
        "ead": ead.round(2),
        "completion_pct": completion_pct.round(4),
        "completion_stage": completion_stage,
        "presale_coverage": presale_coverage.round(4),
        "presale_value": presale_value.round(2),
        "cost_to_complete": cost_to_complete.round(2),
        "default_date": default_dates,
        "default_trigger": default_triggers,
        "contract_rate_proxy": contract_rate_proxy.round(4),
        "cost_of_funds_proxy": cost_of_funds_proxy.round(4),
        "discount_rate": discount_rate.round(4),
        "grv_decline": grv_decline.round(4),
        "fund_to_complete": fund_to_complete.astype(int),
        "workout_months": workout_months_arr,
        "realised_lgd": realised_lgd.round(6),
        "industry_risk_score": dev_industry_risk_scores.round(2),
        "industry_risk_level": dev_industry_risk_levels,
    })

    cashflows = pd.DataFrame(all_cashflows)
    return loans, cashflows


# ---------------------------------------------------------------------------
# 4. CASH FLOW LENDING (PD-aligned unsecured / receivables-secured)
# ---------------------------------------------------------------------------

def generate_cashflow_lending_data(n_loans=400, seed=45):
    """
    Generate synthetic cash flow lending default & workout data.

    Aligned with PD Scorecard project:
    - 8 product types (unsecured and receivables-secured)
    - PD score bands A-E from WoE logistic regression scorecard
    - Borrower features: DSCR, bureau score, FCF margin, revenue CAGR
    - Conduct classification (Green / Amber / Red)
    - Recovery driven by cash flow capacity, not hard collateral
    """
    rng = np.random.RandomState(seed)

    loan_ids = np.arange(1, n_loans + 1)
    states = rng.choice(STATES, n_loans, p=STATE_WEIGHTS)
    industries = rng.choice(INDUSTRIES, n_loans, p=INDUSTRY_WEIGHTS)
    products = rng.choice(
        CASHFLOW_PRODUCTS, n_loans, p=CASHFLOW_PRODUCT_WEIGHTS
    )

    # --- Borrower characteristics ---
    annual_revenue = rng.lognormal(np.log(3_000_000), 0.9, n_loans).clip(
        300_000, 100_000_000
    )
    ebitda_margin = rng.uniform(0.04, 0.22, n_loans)
    ebitda = annual_revenue * ebitda_margin
    years_in_business = rng.randint(1, 35, n_loans)

    # Cash flow metrics (key PD drivers)
    dscr = rng.uniform(0.80, 3.50, n_loans)
    bureau_score = rng.normal(650, 70, n_loans).clip(400, 850).astype(int)
    fcf_margin = rng.uniform(-0.05, 0.20, n_loans)
    revenue_cagr = rng.uniform(-0.15, 0.30, n_loans)

    # Conduct classification
    conduct = rng.choice(CONDUCT_CLASSES, n_loans, p=CONDUCT_WEIGHTS)

    # --- PD score band assignment ---
    # Simulate WoE scorecard: composite score from borrower features
    raw_score = (
        0.30 * np.clip((bureau_score - 400) / 450, 0, 1)
        + 0.25 * np.clip((dscr - 0.80) / 2.70, 0, 1)
        + 0.15 * np.clip((fcf_margin + 0.05) / 0.25, 0, 1)
        + 0.10 * np.clip(revenue_cagr / 0.30, 0, 1)
        + 0.10 * np.clip(years_in_business / 30, 0, 1)
        + 0.10 * np.where(conduct == "Green", 0.8,
                 np.where(conduct == "Amber", 0.4, 0.1))
        + rng.normal(0, 0.05, n_loans)
    ).clip(0, 1)

    # Map to PD and score band
    pd_estimate = 0.25 * np.exp(-4.0 * raw_score)  # higher score = lower PD
    pd_estimate = pd_estimate.clip(0.002, 0.25)

    score_band = np.where(
        pd_estimate <= 0.025, "A",
        np.where(pd_estimate <= 0.05, "B",
        np.where(pd_estimate <= 0.075, "C",
        np.where(pd_estimate <= 0.10, "D", "E")))
    )

    # --- Facility details ---
    facility_limit = rng.uniform(100_000, 8_000_000, n_loans)

    # Product-specific drawn behaviour
    is_revolving = np.isin(products, [
        "Working Capital Facility", "Business Line of Credit",
        "Invoice Finance", "Merchant Cash Advance",
    ])
    drawn_pct = np.where(
        is_revolving,
        rng.uniform(0.45, 0.95, n_loans),
        rng.uniform(0.80, 1.00, n_loans),
    )
    drawn_balance = facility_limit * drawn_pct
    undrawn = facility_limit - drawn_balance
    ccf = np.where(is_revolving, rng.uniform(0.50, 0.80, n_loans), 1.0)
    ead = drawn_balance + ccf * undrawn

    # Leverage
    total_debt = ead * rng.uniform(1.0, 2.0, n_loans)
    leverage = total_debt / np.maximum(ebitda, 1)

    # Seniority -- mostly senior unsecured for cashflow lending
    has_receivables_security = np.isin(products, [
        "Invoice Finance", "Trade Finance",
    ]).astype(int)
    seniority = np.where(
        has_receivables_security, "Senior Secured", "Senior Unsecured"
    )

    # Security coverage (low for cashflow lending)
    security_coverage = np.where(
        has_receivables_security,
        rng.uniform(0.40, 0.80, n_loans),
        rng.uniform(0.0, 0.20, n_loans),
    )

    # Default
    default_dates = _random_dates(
        datetime(2018, 1, 1), datetime(2024, 6, 30), n_loans, rng
    )
    default_triggers = rng.choice(
        ["90 DPD", "Covenant Breach", "Voluntary Administration",
         "Cash Flow Shortfall", "Bureau Downgrade"],
        n_loans, p=[0.25, 0.20, 0.15, 0.25, 0.15]
    )
    contract_rate_proxy = rng.uniform(0.065, 0.105, n_loans)
    cost_of_funds_proxy = rng.uniform(0.050, 0.080, n_loans)
    discount_rate = _build_discount_rate(contract_rate_proxy, cost_of_funds_proxy)

    resolution_strategies = rng.choice(
        ["Voluntary Administration", "Workout", "Write-off", "DOCA"],
        n_loans, p=[0.25, 0.35, 0.20, 0.20]
    )

    # Industry risk scores
    industry_risk_scores = np.array([
        INDUSTRY_RISK_PROFILES[ind]["risk_score"] for ind in industries
    ])

    # --- Build cashflows and compute LGD ---
    all_cashflows = []
    realised_lgd = np.zeros(n_loans)
    workout_months_arr = np.zeros(n_loans, dtype=int)

    for i in range(n_loans):
        loan_cfs = []
        d_date = default_dates[i]
        workout_months = rng.randint(6, 30)
        workout_months_arr[i] = workout_months
        total_pv_recovery = 0.0
        total_pv_cost = 0.0

        # Industry risk penalty
        ind_penalty = 0.02 * max(industry_risk_scores[i] - 2.0, 0)

        # Base recovery rate driven by cash flow capacity and PD band
        pd_band_recovery_mod = {
            "A": 0.12, "B": 0.06, "C": 0.0, "D": -0.08, "E": -0.15
        }
        base_recovery = 0.35 + pd_band_recovery_mod[score_band[i]]
        # DSCR effect on recovery
        dscr_effect = 0.05 * max(dscr[i] - 1.0, 0)
        # Conduct effect
        conduct_effect = (
            0.03 if conduct[i] == "Green"
            else -0.02 if conduct[i] == "Red"
            else 0.0
        )
        effective_recovery = max(
            base_recovery + dscr_effect + conduct_effect - ind_penalty, 0.05
        )
        effective_recovery = min(effective_recovery, 0.70)

        # Receivables-secured products get additional asset recovery
        if has_receivables_security[i]:
            recv_recovery = security_coverage[i] * ead[i] * rng.uniform(0.50, 0.80)
            recv_month = rng.randint(3, min(workout_months, 12) + 1)
            recv_date = d_date + timedelta(days=30 * recv_month)
            recv_days = (recv_date - d_date).days
            recv_pv = _discount(recv_recovery, recv_days, discount_rate[i])
            total_pv_recovery += recv_pv
            loan_cfs.append({
                "loan_id": i + 1, "product": "Cashflow Lending",
                "cashflow_date": recv_date, "days_from_default": recv_days,
                "cashflow_type": "Receivables Collection",
                "cashflow_category": "Recovery",
                "amount": round(recv_recovery, 2),
                "amount_pv": round(recv_pv, 2),
            })

        # Cash flow recovery (administration dividend / workout payments)
        cf_recovery = ead[i] * effective_recovery
        if has_receivables_security[i]:
            cf_recovery *= 0.60  # avoid double-counting
        n_tranches = rng.randint(2, 6)
        tranche_amts = rng.dirichlet(np.ones(n_tranches)) * cf_recovery
        for t_idx, t_amt in enumerate(tranche_amts):
            t_month = rng.randint(3, workout_months + 1)
            t_date = d_date + timedelta(days=30 * t_month)
            t_days = (t_date - d_date).days
            t_pv = _discount(t_amt, t_days, discount_rate[i])
            total_pv_recovery += t_pv
            loan_cfs.append({
                "loan_id": i + 1, "product": "Cashflow Lending",
                "cashflow_date": t_date, "days_from_default": t_days,
                "cashflow_type": "Administration Dividend",
                "cashflow_category": "Recovery",
                "amount": round(t_amt, 2),
                "amount_pv": round(t_pv, 2),
            })

        # --- Costs ---
        # Administrator / receiver fees
        admin_fee = ead[i] * rng.uniform(0.04, 0.10)
        admin_date = d_date + timedelta(days=30 * workout_months)
        admin_days = (admin_date - d_date).days
        admin_pv = _discount(admin_fee, admin_days, discount_rate[i])
        total_pv_cost += admin_pv
        loan_cfs.append({
            "loan_id": i + 1, "product": "Cashflow Lending",
            "cashflow_date": admin_date, "days_from_default": admin_days,
            "cashflow_type": "Administrator Fee",
            "cashflow_category": "Cost",
            "amount": round(admin_fee, 2),
            "amount_pv": round(admin_pv, 2),
        })

        # Legal
        legal = ead[i] * rng.uniform(0.02, 0.06)
        legal_month = rng.randint(1, max(workout_months // 2, 2))
        legal_date = d_date + timedelta(days=30 * legal_month)
        legal_days = (legal_date - d_date).days
        legal_pv = _discount(legal, legal_days, discount_rate[i])
        total_pv_cost += legal_pv
        loan_cfs.append({
            "loan_id": i + 1, "product": "Cashflow Lending",
            "cashflow_date": legal_date, "days_from_default": legal_days,
            "cashflow_type": "Legal Cost",
            "cashflow_category": "Cost",
            "amount": round(legal, 2),
            "amount_pv": round(legal_pv, 2),
        })

        econ_loss = ead[i] + total_pv_cost - total_pv_recovery
        realised_lgd[i] = max(econ_loss / ead[i], 0.0)

        all_cashflows.extend(loan_cfs)

    # Industry columns
    industry_risk_levels = np.array([
        INDUSTRY_RISK_PROFILES[ind]["risk_level"] for ind in industries
    ])
    industry_debt_to_ebitda = np.array([
        INDUSTRY_RISK_PROFILES[ind]["debt_to_ebitda"] for ind in industries
    ])
    industry_icr_bench = np.array([
        INDUSTRY_RISK_PROFILES[ind]["icr"] for ind in industries
    ])

    loans = pd.DataFrame({
        "loan_id": loan_ids,
        "product": "Cashflow Lending",
        "cashflow_product": products,
        "state": states,
        "industry": industries,
        "annual_revenue": annual_revenue.round(2),
        "ebitda": ebitda.round(2),
        "ebitda_margin": ebitda_margin.round(4),
        "leverage_ratio": leverage.round(2),
        "dscr": dscr.round(4),
        "bureau_score": bureau_score,
        "fcf_margin": fcf_margin.round(4),
        "revenue_cagr": revenue_cagr.round(4),
        "conduct_classification": conduct,
        "years_in_business": years_in_business,
        "pd_estimate": pd_estimate.round(6),
        "pd_score_band": score_band,
        "facility_type": np.where(is_revolving, "Revolving", "Term"),
        "facility_limit": facility_limit.round(2),
        "drawn_balance": drawn_balance.round(2),
        "undrawn_amount": undrawn.round(2),
        "ccf": ccf.round(4),
        "ead": ead.round(2),
        "seniority": seniority,
        "has_receivables_security": has_receivables_security,
        "security_coverage_ratio": security_coverage.round(4),
        "default_date": default_dates,
        "default_trigger": default_triggers,
        "contract_rate_proxy": contract_rate_proxy.round(4),
        "cost_of_funds_proxy": cost_of_funds_proxy.round(4),
        "discount_rate": discount_rate.round(4),
        "resolution_strategy": resolution_strategies,
        "workout_months": workout_months_arr,
        "realised_lgd": realised_lgd.round(6),
        "industry_risk_score": industry_risk_scores.round(2),
        "industry_risk_level": industry_risk_levels,
        "industry_debt_to_ebitda_benchmark": industry_debt_to_ebitda,
        "industry_icr_benchmark": industry_icr_bench,
    })

    cashflows = pd.DataFrame(all_cashflows)
    return loans, cashflows


# ---------------------------------------------------------------------------
# CONVENIENCE: Generate all products
# ---------------------------------------------------------------------------

def generate_all_datasets():
    """Generate all four product datasets and return as a dict."""
    mortgage_loans, mortgage_cfs = generate_mortgage_data()
    commercial_loans, commercial_cfs = generate_commercial_data()
    development_loans, development_cfs = generate_development_data()
    cashflow_loans, cashflow_cfs = generate_cashflow_lending_data()

    return {
        "mortgage": {"loans": mortgage_loans, "cashflows": mortgage_cfs},
        "commercial": {"loans": commercial_loans, "cashflows": commercial_cfs},
        "development": {"loans": development_loans, "cashflows": development_cfs},
        "cashflow_lending": {"loans": cashflow_loans, "cashflows": cashflow_cfs},
    }


if __name__ == "__main__":
    import os
    datasets = generate_all_datasets()
    os.makedirs("data/raw", exist_ok=True)
    for product, data in datasets.items():
        loans_file = f"data/raw/{product}_loans.csv"
        cfs_file = f"data/raw/{product}_cashflows.csv"
        data["loans"].to_csv(loans_file, index=False)
        data["cashflows"].to_csv(cfs_file, index=False)
        print(f"{product}: {len(data['loans'])} loans, {len(data['cashflows'])} cashflows -> {loans_file}")
