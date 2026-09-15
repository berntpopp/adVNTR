"""Pure replay traversal using the production visit and statistic functions."""
from advntr.coverage_guard import compute_locus_coverage
from advntr.frameshift_traversal import iter_visits
from advntr.frameshift_visit_decisions import assess_visit, coverage_basis


def evaluate_visits(traversal, reference_order, boundaries, geometry, records, policy, background):
    """Revisit the frozen candidates, including policy-dependent continue behavior.

    Inventory-only evidence never becomes a candidate. Flank borrowing, boundary
    comparisons, mean-coverage association and every native numeric decision are
    delegated to the same functions used by the production adapter.
    """
    if (policy.capture.caller_mode == 'exact') != (background is not None):
        raise ValueError('replay background presence differs from the requested caller mode')
    coverage = dict((index, row['ru_bp_coverage']) for index, row in geometry.items())
    lengths = dict((index, len(row['sequence'])) for index, row in geometry.items())
    copies = dict((index, row['reference_copies']) for index, row in geometry.items())
    locus = compute_locus_coverage(coverage, lengths, copies, policy.capture.is_haploid)
    result = []
    for plan in iter_visits(traversal, reference_order, boundaries[0], boundaries[1], policy.frameshift):
        basis = None
        if plan.disposition == 'ready-to-score':
            index = plan.repeat_unit_index
            basis = coverage_basis(coverage[index], lengths[index], copies[index], policy.capture.is_haploid, locus)
        result.append(assess_visit(plan, basis, policy.frameshift, records, background,
                                    rare_fraction=policy.capture.minimum_relative_ru_coverage,
                                    legacy_error_rate=policy.capture.legacy_error_rate))
    return tuple(result)
