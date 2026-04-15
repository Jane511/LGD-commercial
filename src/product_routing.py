"""
Shared product-type resolver for the LGD framework.

Enforces the 3-family hierarchy:

    mortgage | cashflow_lending | property_backed_lending

``_resolve_product()`` is the single entry point for all product-type
routing.  Import it; do not reimplement the logic elsewhere.
"""
from __future__ import annotations

CANONICAL_FAMILIES: tuple[str, ...] = (
    "mortgage",
    "cashflow_lending",
    "property_backed_lending",
)

FAMILY_SUB_TYPES: dict[str, tuple[str, ...]] = {
    "mortgage": (
        "mortgage",
        "residential_mortgage",
        "property_mortgage",
    ),
    "cashflow_lending": (
        "cashflow_lending",
        "commercial_cashflow",
        "receivables",
        "trade_contingent",
        "asset_equipment",
    ),
    "property_backed_lending": (
        "development_finance",
        "cre_investment",
        "residual_stock",
        "land_subdivision",
        "bridging",
        "mezz_second_mortgage",
    ),
}

PRODUCT_TO_FAMILY: dict[str, str] = {
    sub: family
    for family, subs in FAMILY_SUB_TYPES.items()
    for sub in subs
}

LEGACY_AMBIGUOUS: dict[str, str] = {
    "commercial": (
        "Ambiguous product_type 'commercial'. Use a specific sub-type: "
        "commercial_cashflow, cre_investment, bridging, or mezz_second_mortgage."
    ),
    "commercial_lending": (
        "Ambiguous product_type 'commercial_lending'. Use commercial_cashflow for "
        "DSCR/cashflow-based facilities or cre_investment / bridging for "
        "property-backed facilities."
    ),
    "development": (
        "Deprecated product_type 'development'. Use development_finance instead."
    ),
}


def _resolve_product(product_type: str) -> tuple[str, str]:
    """
    Validate and resolve a product_type string to ``(family, sub_type)``.

    Parameters
    ----------
    product_type:
        Raw product type string supplied by the caller.

    Returns
    -------
    (family, sub_type) where *family* is one of ``CANONICAL_FAMILIES`` and
    *sub_type* is the normalised key present in ``PRODUCT_TO_FAMILY``.

    Raises
    ------
    ValueError
        For ambiguous legacy labels (``commercial``, ``commercial_lending``,
        ``development``) or completely unknown product types.
    """
    key = str(product_type).strip().lower()

    if key in LEGACY_AMBIGUOUS:
        raise ValueError(LEGACY_AMBIGUOUS[key])

    family = PRODUCT_TO_FAMILY.get(key)
    if family is None:
        known = ", ".join(sorted(PRODUCT_TO_FAMILY))
        raise ValueError(
            f"Unsupported product_type '{product_type}'. "
            f"Known sub-types: {known}"
        )
    return family, key
