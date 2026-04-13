import json
from pathlib import Path


def _replace_in_cell_source(cell, replacements):
    if cell.get('cell_type') != 'code':
        return False
    src = ''.join(cell.get('source', []))
    changed = False
    for old, new in replacements:
        if old in src:
            src = src.replace(old, new)
            changed = True
    if changed:
        cell['source'] = src.splitlines(keepends=True)
    return changed


def patch_notebook(path, replacements):
    p = Path(path)
    nb = json.loads(p.read_text(encoding='utf-8'))
    any_changed = False
    for cell in nb.get('cells', []):
        any_changed = _replace_in_cell_source(cell, replacements) or any_changed
    if any_changed:
        p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    return any_changed


def main():
    changes = []

    # 03 Commercial: remove duplicate exports and legacy aliases.
    commercial_replacements = [
        (
            "weighted_lgd_by_segment.to_csv(os.path.join(TABLE_DIR, 'commercial_framework_subsegment_comparison.csv'), index=False)\n",
            "",
        ),
        (
            "framework_export.to_csv(os.path.join(TABLE_DIR, 'commercial_lgd_results.csv'), index=False)\n",
            "",
        ),
        (
            "weighted_lgd_by_segment.to_csv(os.path.join(TABLE_DIR, 'commercial_segment_summary.csv'), index=False)\n",
            "",
        ),
        ("print('- commercial_framework_subsegment_comparison.csv')\n", ""),
        ("print('- commercial_lgd_results.csv (backward-compatible alias)')\n", ""),
        ("print('- commercial_segment_summary.csv (backward-compatible alias)')\n", ""),
    ]
    if patch_notebook('notebooks/03_commercial_cashflow_lgd.ipynb', commercial_replacements):
        changes.append('notebooks/03_commercial_cashflow_lgd.ipynb')

    # 08 CRE: stop exporting the same df twice under two names.
    cre_replacements = [
        (
            "sensitivity_df.to_csv(os.path.join(OUTPUT_DIR, 'tables', 'cre_investment_scenario_summary.csv'), index=False)\n",
            "",
        ),
        ("print('- cre_investment_scenario_summary.csv')\n", ""),
    ]
    if patch_notebook('notebooks/08_cre_investment_lgd.ipynb', cre_replacements):
        changes.append('notebooks/08_cre_investment_lgd.ipynb')

    # 13 Cross-product: remove duplicate weighted comparison export.
    cross_replacements = [
        ("comparison_df.to_csv(TABLE_DIR / 'cross_product_weighted_comparison.csv', index=False)\n", ""),
        ("print('- cross_product_weighted_comparison.csv')\n", ""),
    ]
    if patch_notebook('notebooks/13_cross_product_comparison.ipynb', cross_replacements):
        changes.append('notebooks/13_cross_product_comparison.ipynb')

    if changes:
        print('Patched notebooks:')
        for c in changes:
            print('-', c)
    else:
        print('No notebook changes were required.')


if __name__ == '__main__':
    main()
