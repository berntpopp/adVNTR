"""Pure production statistic results shared with future capture replay.

This ABI deliberately distinguishes read support in the inherited statistic from
occurrence support in the exact tail. It does not apply recruitment, flank,
minimum read support or coverage gates. Those precede these statistics.

``log_tail`` is only populated for exact decisions. An underflowed probability
is zero while its log tail remains finite; a genuinely zero tail is -infinity.
Legacy NaN is retained here for compatibility, never valid JSON capture data:
a future serializer must explicitly encode its disposition and a null statistic.
"""
import math
import numbers
from collections import namedtuple

from advntr.exact_tail import exact_indel_tail_log
from advntr.frameshift_decisions import legacy_call, validate_policy


StatisticDecision = namedtuple(
    'StatisticDecision', ('called', 'pvalue', 'log_tail', 'disposition'))


def legacy_statistic(location_coverage, observed_indel_transitions,
                     expected_indels, error_rate):
    """Return the unchanged inherited likelihood statistic and probabilities.

    The observed count is reads; coverage is a mean per base and copy. Their
    inequality is possible, and the inherited early p=0 return is intentionally
    preserved, not normalized into the exact caller's invalid-trials refusal.
    """
    if observed_indel_transitions > location_coverage:
        return 0, 1.0, 0
    from scipy.stats import binom, chi2
    sequencing_error_prob = binom.pmf(observed_indel_transitions, location_coverage, error_rate)
    frameshift_prob = binom.pmf(observed_indel_transitions, location_coverage, expected_indels)
    chi_square_val = -2 * (binom.logpmf(observed_indel_transitions, location_coverage, error_rate) -
                           binom.logpmf(observed_indel_transitions, location_coverage, expected_indels))
    pval = chi2.sf(chi_square_val, 1)
    return sequencing_error_prob, frameshift_prob, pval


def legacy_result(pvalue, policy):
    """Apply the strict inherited probability cutoff, including NaN suppression."""
    called = legacy_call(pvalue, policy)
    disposition = 'called' if called else 'cutoff'
    if math.isnan(pvalue):
        disposition = 'legacy-nonfinite'
    return StatisticDecision(called, pvalue, None, disposition)


def exact_result(support, opportunities, probability, policy):
    """Score occurrence counts with the strict production log-tail comparator.

    Invalid k>N is an explicit refusal before background lookup in production;
    ``probability`` may therefore be None on that path. All scoreable paths use
    the existing exact-tail input validation, including zero-support cases.
    """
    validate_policy(policy)
    for count in (support, opportunities):
        if (isinstance(count, bool) or not isinstance(count, numbers.Integral)
                or count < 0):
            raise ValueError('exact support and opportunities must be non-negative integers')
    if support > opportunities:
        return StatisticDecision(False, None, None, 'support-exceeds-opportunities')
    log_tail = exact_indel_tail_log(support, opportunities, probability)
    called = log_tail < math.log(policy.cutoff)
    disposition = 'called' if called else 'cutoff'
    if opportunities == 0:
        disposition = 'no-trials'
    elif support == 0:
        disposition = 'no-occurrence-support'
    return StatisticDecision(called, math.exp(log_tail), log_tail, disposition)
