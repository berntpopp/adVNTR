"""Rare-unit coverage guard (SPEC Q-RARE, PLAN Task 9).

When sequencing depth is uneven across repeat units (e.g. a rare single-copy repeat
unit receiving few mapped reads while others receive hundreds), the unit's coverage
denominator collapses. Under low coverage, sequencing errors or low-support artifacts
can appear statistically significant against an expected alternative rate.

This module provides relative coverage guarding: a candidate occurring in a repeat
unit whose observed coverage has collapsed below a specified fraction of the locus-wide
average coverage is guarded against false positive calling.

The threshold is evaluated relative to locus depth so that the decision boundary
scales appropriately with sample sequencing depth (e.g. 30x vs 300x) rather than
imposing an arbitrary absolute cutoff.
"""


def compute_locus_coverage(ru_bp_coverage, hmm_match_count, estimated_ru_count, is_haploid=False):
    """Calculate the locus-wide average base-pair coverage across all repeat units.

    Parameters
    ----------
    ru_bp_coverage : dict
        Mapping of pattern index (str) to total observed base pairs mapped to that RU.
    hmm_match_count : dict
        Mapping of pattern index (str) to reference repeat unit length in base pairs.
    estimated_ru_count : dict
        Mapping of pattern index (str) to copy count of that repeat unit in the reference.
    is_haploid : bool
        Whether the organism / chromosome is haploid.

    Returns
    -------
    float
        Average base-pair coverage across the entire VNTR locus.
    """
    total_bps = sum(ru_bp_coverage.values())
    total_expected_bps = sum(
        hmm_match_count[k] * estimated_ru_count[k] for k in estimated_ru_count
    )
    if total_expected_bps <= 0:
        return 0.0
    multiplier = 1.0 if is_haploid else 2.0
    return float(total_bps) / (multiplier * total_expected_bps)


def is_rare_unit_coverage_collapsed(ru_coverage, locus_coverage, min_relative_coverage=None):
    """Return whether a repeat unit's coverage has collapsed relative to locus depth.

    Parameters
    ----------
    ru_coverage : float
        Average base-pair coverage for the specific repeat unit model.
    locus_coverage : float
        Locus-wide average base-pair coverage across all repeat units.
    min_relative_coverage : float or None
        Minimum allowable fraction of locus coverage (e.g. 0.15). If None or locus
        coverage <= 0, the guard is inactive and returns False.

    Returns
    -------
    bool
        True if the unit's coverage is collapsed below the relative threshold;
        False otherwise.
    """
    if min_relative_coverage is None:
        return False
    if locus_coverage <= 0.0:
        return False
    return ru_coverage < locus_coverage * min_relative_coverage
