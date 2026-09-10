"""Callable cluster size analysis and validation (Issue #7, Constraint 2).

In the legacy frameshift statistical test (advntr/vntr_finder.py:identify_frameshift),
the expected probability of indel transitions under the alternative hypothesis (H1:
frameshift on one allele) is:

    p_frameshift = 0.99 / (2 * E)   (diploid)
    p_frameshift = 0.99 / E         (haploid)

where E is estimated_ru_count[repeat_unit_index] (the number of times that repeat unit
occurs in the reference VNTR).

This is compared against the null hypothesis (H0: sequencing error):

    p_error = INDEL_ERROR_RATE = 0.01

When E is large:
    p_frameshift <= p_error  <=>  0.99 / (2 * E) <= 0.01  <=>  E >= 49.5 (i.e. E >= 50)

At E >= 50 (or E >= 100 haploid), the likelihood ratio test inverts: the alternative
hypothesis expects FEWER indels than sequencing error. A sufficiently low observed count
then fits the alternative better than the null, causing false-positive frameshift calls
on 3 supporting reads at large E.

This module provides build-time and runtime validation to detect and prevent cluster
inversions.
"""
import logging
import math

from advntr import settings


class CallableClusterError(ValueError):
    """Raised when a repeat-unit cluster size exceeds the callable capacity."""
    pass


def max_callable_cluster_size(is_haploid=False, error_rate=None):
    """Return the theoretical upper bound on cluster occurrences E before inversion.

    At or above this size, p_frameshift <= p_error, inverting the likelihood ratio.
    """
    if error_rate is None:
        error_rate = settings.INDEL_ERROR_RATE
    if error_rate <= 0:
        raise ValueError('error_rate must be positive: %s' % error_rate)
    numerator = 0.99 if is_haploid else (0.99 / 2.0)
    # The crossover occurs where numerator / E <= error_rate, i.e. E >= numerator / error_rate
    # The maximum callable size strictly below crossover is floor((numerator / error_rate) - epsilon)
    bound = numerator / error_rate
    if bound == math.floor(bound):
        return int(bound - 1)
    return int(math.floor(bound))


def practical_callable_cluster_size(is_haploid=False, error_rate=None, min_ratio=1.5):
    """Return practical cluster size bound ensuring p_frameshift is at least min_ratio * p_error."""
    if error_rate is None:
        error_rate = settings.INDEL_ERROR_RATE
    numerator = 0.99 if is_haploid else (0.99 / 2.0)
    target = min_ratio * error_rate
    return int(math.floor(numerator / target))


def is_cluster_callable(cluster_size, is_haploid=False, error_rate=None):
    """Check whether a repeat unit cluster size is callable without likelihood inversion."""
    return cluster_size <= max_callable_cluster_size(is_haploid, error_rate)


def validate_cluster_sizes(pattern_clusters, is_haploid=False, error_rate=None,
                           strict=False):
    """Validate cluster occurrences against theoretical and practical callable limits.

    :param pattern_clusters: list of lists of pattern strings, or dict of cluster_id -> count
    :param is_haploid: boolean indicating haploid vs diploid
    :param error_rate: indel sequencing error rate (default settings.INDEL_ERROR_RATE)
    :param strict: if True, raise CallableClusterError on uncallable clusters
    :return: dict of issues found: {cluster_idx: (cluster_size, max_allowed)}
    """
    max_size = max_callable_cluster_size(is_haploid, error_rate)
    issues = {}

    if isinstance(pattern_clusters, dict):
        iterator = pattern_clusters.items()
    else:
        iterator = enumerate(pattern_clusters)

    for cluster_id, cluster in iterator:
        size = len(cluster) if isinstance(cluster, (list, tuple)) else int(cluster)
        if size > max_size:
            issues[cluster_id] = (size, max_size)
            msg = ('Cluster %s size %d exceeds max callable cluster size %d '
                   '(inverts legacy likelihood ratio test: p_frameshift <= p_error). '
                   'Use --exact-frameshift-caller or partition the cluster.' %
                   (cluster_id, size, max_size))
            if strict:
                raise CallableClusterError(msg)
            else:
                logging.warning(msg)

    return issues
