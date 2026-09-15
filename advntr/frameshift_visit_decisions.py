"""Filesystem-free visit scoring shared by production adapters and future replay.

Plans are generated from frozen candidate traversal for the requested policy.
No-call dispositions retain their stage; read support and exact occurrence counts
are separate. Existing legacy statistic callbacks remain possible for library
compatibility, but a custom callback is not an advertised native capture producer.
"""
from collections import namedtuple

from advntr import coverage_guard, exact_caller, frameshift_statistics
from advntr.frameshift_decisions import passes_support, validate_policy


CoverageBasis = namedtuple('CoverageBasis', 'total_bps mean_coverage expected_indels locus_coverage')
VisitAssessment = namedtuple('VisitAssessment', (
    'plan disposition mean_coverage expected_indels sequencing_error_probability '
    'frameshift_probability statistic exact_assessment'))


def coverage_basis(total_bps, unit_length, reference_copies, is_haploid, locus_coverage):
    """Retain source arithmetic association and its zero-geometry failure behavior."""
    if is_haploid:
        mean = float(total_bps) / unit_length / reference_copies
        expected = 0.99 / reference_copies
    else:
        mean = float(total_bps) / unit_length / 2 / reference_copies
        expected = 0.99 / (2 * reference_copies)
    return CoverageBasis(total_bps, mean, expected, locus_coverage)


def assess_visit(plan, basis, policy, records, background, rare_fraction=None,
                 legacy_error_rate=0.01, legacy_statistic=None):
    """Apply coverage and statistic gates to one plan from the same support policy.

    A prescore-suppressed visit needs no geometry or background. The source's
    relative coverage comparison remains strict. Statistic results use the
    existing shared legacy/exact ABI, including NaN noncall and exact k>N refusal.
    """
    validate_policy(policy)
    if plan.disposition == 'outside-boundary':
        return VisitAssessment(plan, plan.disposition, None, None, None, None, None, None)
    supported = passes_support(plan.read_support, policy)
    expected_disposition = 'ready-to-score' if supported else 'insufficient-read-support'
    if plan.disposition != expected_disposition:
        raise ValueError('visit plan does not match the requested supporting-read policy')
    if not supported:
        return VisitAssessment(plan, plan.disposition, None, None, None, None, None, None)
    if coverage_guard.is_rare_unit_coverage_collapsed(basis.mean_coverage, basis.locus_coverage, rare_fraction):
        return VisitAssessment(plan, 'rare-unit-coverage', basis.mean_coverage, basis.expected_indels,
                               None, None, None, None)
    exact = None
    sequencing_error, frameshift_probability = None, None
    if background is None:
        statistic = frameshift_statistics.legacy_statistic if legacy_statistic is None else legacy_statistic
        sequencing_error, frameshift_probability, pvalue = statistic(
            basis.mean_coverage, plan.read_support, basis.expected_indels, error_rate=legacy_error_rate)
        result = frameshift_statistics.legacy_result(pvalue, policy)
    else:
        exact = exact_caller.assess_evidence(records, plan.state, background, policy)
        result = exact.statistic
    return VisitAssessment(plan, result.disposition, basis.mean_coverage, basis.expected_indels,
                           sequencing_error, frameshift_probability, result, exact)


def flank_boundaries(repeat_segments, left_flank, right_flank, read_length):
    """Keep the asymmetric source endpoints, including negative suffix boundaries."""
    first, last = repeat_segments[0][0], repeat_segments[-1][-1]
    suffix, prefix = read_length, 0
    for index in range(1, len(left_flank)):
        if left_flank[-index] == first:
            suffix = read_length - index
        else:
            break
    for index in range(len(right_flank)):
        if right_flank[index] == last:
            prefix = index + 1
        else:
            break
    return suffix, prefix
