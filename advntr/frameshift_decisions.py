"""Pure run-local frameshift policy and production decision gates."""
import math
import numbers
from collections import namedtuple

from advntr.exact_tail import tail_below_cutoff


DEFAULT_PVALUE_CUTOFF = 0.001
DEFAULT_MINIMUM_READ_SUPPORT = 3


class FrameshiftPolicy(namedtuple(
        '_FrameshiftPolicy', ('cutoff', 'minimum_read_support'))):
    """Validated immutable caller thresholds for one adVNTR run."""

    __slots__ = ()

    def __new__(cls, cutoff, minimum_read_support):
        _validate_cutoff(cutoff)
        _validate_minimum_support(minimum_read_support)
        return tuple.__new__(cls, (cutoff, minimum_read_support))

    @classmethod
    def _make(cls, iterable):
        return cls(*iterable)

    def _replace(self, **values):
        unknown = set(values) - set(self._fields)
        if unknown:
            raise ValueError('unknown frameshift policy fields: %s' % sorted(unknown))
        return type(self)(*(values.get(field, getattr(self, field))
                            for field in self._fields))


def resolve_policy(cutoff=None, minimum_read_support=None):
    """Resolve optional CLI values to one validated run-local policy."""
    if cutoff is None:
        cutoff = DEFAULT_PVALUE_CUTOFF
    if minimum_read_support is None:
        minimum_read_support = DEFAULT_MINIMUM_READ_SUPPORT
    return FrameshiftPolicy(cutoff, minimum_read_support)


def passes_support(read_support, policy):
    """Whether the existing read-support gate is met, including equality."""
    validate_policy(policy)
    if (isinstance(read_support, bool)
            or not isinstance(read_support, numbers.Integral)
            or read_support < 0):
        raise ValueError('frameshift read support must be a non-negative integer')
    return read_support >= policy.minimum_read_support


def legacy_call(pvalue, policy):
    """Apply the legacy cutoff, preserving its NaN-as-noncall behavior.

    The inherited statistic can produce NaN for malformed numerical evidence;
    extraction of the decision must not turn its prior suppression into a crash.
    """
    validate_policy(policy)
    if (isinstance(pvalue, numbers.Real) and not isinstance(pvalue, bool)
            and math.isnan(pvalue)):
        return False
    if (not isinstance(pvalue, numbers.Real) or isinstance(pvalue, bool)
            or math.isinf(pvalue)
            or not 0.0 <= pvalue <= 1.0):
        raise ValueError('legacy frameshift p-value must be finite and lie in [0, 1]')
    return pvalue < policy.cutoff


def exact_call(support, opportunities, probability, policy):
    """Apply the exact caller's unchanged strict log-tail cutoff."""
    validate_policy(policy)
    return tail_below_cutoff(
        support, opportunities, probability, policy.cutoff)


def validate_policy(policy):
    """Return the same policy after checking a direct Python caller's boundary."""
    if not isinstance(policy, FrameshiftPolicy):
        raise ValueError('frameshift policy must be a FrameshiftPolicy')
    _validate_cutoff(policy.cutoff)
    _validate_minimum_support(policy.minimum_read_support)
    return policy


def _validate_cutoff(cutoff):
    if (not isinstance(cutoff, float) or isinstance(cutoff, bool)
            or math.isnan(cutoff) or math.isinf(cutoff)
            or not 0.0 < cutoff < 1.0):
        raise ValueError('frameshift p-value cutoff must be a finite float in (0, 1)')


def _validate_minimum_support(minimum_read_support):
    if (isinstance(minimum_read_support, bool)
            or not isinstance(minimum_read_support, numbers.Integral)
            or minimum_read_support < 1):
        raise ValueError('minimum frameshift read support must be an integer >= 1')
