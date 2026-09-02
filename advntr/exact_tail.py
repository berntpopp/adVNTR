"""The one-sided exact binomial tail `P(K >= k | N, p0)`, stable into the deep tail.

PLAN Task 8 Steps 1-2. This is the statistic that replaces `identify_frameshift`
(`advntr/vntr_finder.py:186-196`) inside the default-off exact caller; nothing here is
reachable from the shipped decision path. `k` and `N` are Task 7's integer pair off
`VNTRFinder.last_frameshift_opportunities`, and `p0` comes from an operator-supplied
frozen background (`advntr/frameshift_background.py`) -- never from a constant in this
tree.

**There is no alternative rate.** SPEC 3.1: the exact one-sided test is
`P(K >= k | N, p0)` and nothing else. The shipped `expected_indel_transitions` is one
half of the malformed statistic (Q-DENOM/Q-STAT), not an input here, so these functions
take exactly three arguments and a fourth is a `TypeError` rather than a silently
ignored keyword.

**Why the counts are validated instead of coerced.** SciPy 1.2.1 -- the pinned build --
truncates a fractional `n` and merely warns: measured on this machine,
`binom.sf(2, 10.5, 0.001)` returns exactly `binom.sf(2, 10, 0.001)` =
1.1937150990179906e-07 after emitting `RuntimeWarning: floating point number truncated
to an integer`. Truncation is not a definition of `N` (SPEC 3.1), so a non-integer `N`
raises. `k > N` raises for the same reason and is *not* clamped: it mirrors the
invariant Task 7 already enforces by construction
(`advntr/frameshift_opportunities.py:finalise`), and SciPy would answer it silently --
`binom.sf(11, 10, 0.001)` is `0.0`, indistinguishable from a genuinely strong tail.

**What the public surface returns, and where the decision is taken.**

- `exact_indel_tail_log(k, N, p0)` -- the natural log of the tail. This is the honest
  form and the one `tail_below_cutoff` compares. It may be `-inf`, and that is a valid
  strong result (see below), never an error.
- `exact_indel_tail(k, N, p0)` -- the same tail as a probability, for the `Pvalue`
  column and the log line. It is `math.exp` of the log form, so the two can never
  disagree, and it underflows to `0.0` in the deep tail. SPEC 3.1 forbids promising all
  reported p-values remain nonzero, so it is not clamped to a tiny positive number.
- `tail_below_cutoff(k, N, p0, cutoff)` -- the decision, taken as
  `log_tail < log(cutoff)`. Comparing probabilities would collapse every tail below
  ~1e-308 onto `0.0 < cutoff` and lose the ordering among strong results.

**The deep tail.** `binom.logsf` in SciPy 1.2.1 is literally `log(self._sf(x))`
(`scipy/stats/_distn_infrastructure.py:897`), so it inherits `sf`'s underflow: measured
here, `binom.sf(199, 1000, 1e-4)` is exactly `0.0` and `binom.logsf(199, 1000, 1e-4)` is
`-inf` with a divide-by-zero RuntimeWarning. Rather than accept that loss, the tail is
re-summed in log space from `logpmf(k)` whenever `sf` underflows: for `k` far above
`N*p0` the terms fall geometrically, so a few dozen of them carry the whole tail. The
`-inf` return is therefore reserved for a tail that really is zero (`p0 == 0` with
`k >= 1`), and `tail_below_cutoff` still accepts it as below any cutoff.

**Boundaries, pinned by tests rather than left to emerge:**

| case | value | why |
|---|---|---|
| `k == 0` | `1.0` | `P(K >= 0) = 1` for every `N` and `p0`, degenerate ones included |
| `N == 0` | `1.0` (only `k == 0` is legal) | `k > N` raises, so nothing else reaches here |
| `k == N` | `p0 ** N` | the single-term tail |
| `p0 == 0` | `0.0` / `-inf` for `k >= 1` | a genuine impossibility, not an underflow |
| `p0 == 1` | `1.0` | every trial succeeds, so `K == N >= k` |
"""
import math
import numbers


#: Relative size below which a summed tail term stops changing the result. Doubles carry
#: ~2.2e-16, so a term this small cannot move the log tail at the precision anything
#: downstream reads; it is a convergence test, not a tolerance on the answer.
_NEGLIGIBLE_TERM = 1e-18


def _validate(k, n, p0):
    """Reject everything SciPy would silently reinterpret. See the module docstring."""
    for name, value in (('support k', k), ('opportunities N', n)):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise ValueError('%s must be an integer count, got %r (%s). The pinned '
                             'SciPy truncates a fractional trial count and only warns; '
                             'truncation is not a definition of N (SPEC 3.1).'
                             % (name, value, type(value).__name__))
        if value < 0:
            raise ValueError('%s must not be negative, got %d' % (name, value))
    if k > n:
        raise ValueError('support k=%d exceeds opportunities N=%d. This is an invariant '
                         'failure, not a value to clamp: SciPy answers it silently with '
                         '0.0, which is indistinguishable from a strong tail.' % (k, n))
    if not isinstance(p0, numbers.Real) or isinstance(p0, bool):
        raise ValueError('background probability p0 must be a real number, got %r' % (p0,))
    if math.isnan(p0):
        raise ValueError('background probability p0 is NaN')
    if not 0.0 <= p0 <= 1.0:
        raise ValueError('background probability p0 must lie in [0, 1], got %r' % (p0,))


def _log_tail_by_summation(k, n, p0):
    """`log P(K >= k)` summed from `logpmf(k)` upward, for tails `sf` underflows on.

    Terms are generated by the exact ratio `pmf(j+1)/pmf(j) = (n-j)/(j+1) * p0/(1-p0)`
    rather than by re-entering SciPy per term, so the sum costs one `logpmf` call. It is
    written relative to `logpmf(k)`, which keeps every accumulated term near 1.0 and out
    of the underflow that made this function necessary.

    The loop is bounded by `n` twice over -- by the counter and by the negligibility
    test -- because this runs inside a production caller. Callers must have validated
    `0 < p0 < 1` and `1 <= k <= n` already.
    """
    from scipy.stats import binom
    anchor = binom.logpmf(k, n, p0)
    odds = p0 / (1.0 - p0)
    term = 1.0
    total = 1.0
    position = k
    while position < n:
        term *= (n - position) * odds / (position + 1)
        total += term
        if term < _NEGLIGIBLE_TERM * total:
            break
        position += 1
    return float(anchor) + math.log(total)


def exact_indel_tail_log(k, n, p0):
    """`log P(K >= k | N = n, p0)`, the form the decision is taken in.

    Returns `-inf` only for a tail that is genuinely zero; deep-but-nonzero tails come
    back finite (SciPy's own `logsf` does not -- see the module docstring).
    """
    _validate(k, n, p0)
    if k == 0 or p0 == 1.0:
        return 0.0
    if p0 == 0.0:
        return float('-inf')
    from scipy.stats import binom
    tail = binom.sf(k - 1, n, p0)
    if tail > 0.0:
        return float(math.log(tail))
    return _log_tail_by_summation(k, n, p0)


def exact_indel_tail(k, n, p0):
    """`P(K >= k | N = n, p0)` as a probability, for reporting.

    Derived from `exact_indel_tail_log`, so it cannot disagree with the decision. It
    underflows to `0.0` in the deep tail, deliberately: SPEC 3.1 says the implementation
    must not promise all reported p-values remain nonzero.
    """
    log_tail = exact_indel_tail_log(k, n, p0)
    if log_tail == 0.0:
        return 1.0
    return math.exp(log_tail)


def tail_below_cutoff(k, n, p0, cutoff):
    """Is the exact tail strictly below `cutoff`? Compared in log space.

    `-inf < log(cutoff)` for every admissible cutoff, so an exactly-zero tail is below
    any of them -- the "valid strong result" SPEC 3.1 asks for, reached without ever
    testing the numeric value zero.
    """
    if not isinstance(cutoff, numbers.Real) or isinstance(cutoff, bool):
        raise ValueError('cutoff must be a real number, got %r' % (cutoff,))
    if math.isnan(cutoff) or not 0.0 < cutoff <= 1.0:
        raise ValueError('cutoff must lie in (0, 1], got %r' % (cutoff,))
    return exact_indel_tail_log(k, n, p0) < math.log(cutoff)
