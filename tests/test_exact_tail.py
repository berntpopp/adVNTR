"""The one-sided exact binomial tail PLAN Task 8 Steps 1-2 specify.

Two of these tests are given verbatim by the plan (`task-8-brief.md` Step 1); the rest
pin the boundaries and the deep tail SPEC 3.1 calls out. Nothing here touches a BAM, an
HMM or a cohort: the tail is a pure function of `(k, N, p0)`.

The pinned SciPy is 1.2.1. Two of its behaviours are the reason this module exists at
all, and both are asserted here against the real library rather than taken on trust:
`binom.sf` silently truncates a fractional `n` (with a RuntimeWarning), and
`binom.logsf` is literally `log(sf)`, so it underflows to `-inf` once the tail drops
below ~1e-308.
"""
import decimal
import math
import unittest
import warnings

from advntr.exact_tail import (exact_indel_tail, exact_indel_tail_log,
                               tail_below_cutoff)


def _reference_log_tail(k, n, p0):
    """`log P(K >= k)` in 60-digit decimal arithmetic, independent of SciPy."""
    with decimal.localcontext() as context:
        context.prec = 60
        probability = decimal.Decimal(p0)
        complement = 1 - probability
        coefficient = decimal.Decimal(1)
        for i in range(k):
            coefficient = coefficient * (n - i) / (i + 1)
        term = coefficient * probability ** k * complement ** (n - k)
        total = decimal.Decimal(0)
        for j in range(k, n + 1):
            total += term
            term = term * decimal.Decimal(n - j) / decimal.Decimal(j + 1) \
                * probability / complement
        return total.ln()


class TestExactTailValidation(unittest.TestCase):
    def test_exact_tail_rejects_a_non_integer_trial_count(self):
        with self.assertRaises(ValueError):
            exact_indel_tail(3, 10.5, 0.001)

    def test_support_cannot_exceed_opportunities(self):
        with self.assertRaises(ValueError):
            exact_indel_tail(11, 10, 0.001)

    def test_the_rejected_fractional_count_never_reaches_scipy(self):
        """SciPy 1.2.1 truncates it and warns; truncation is not a definition of `N`.

        Measured on the pinned build: `binom.sf(2, 10.5, 0.001)` returns exactly
        `binom.sf(2, 10, 0.001)` = 1.1937150990179906e-07 after emitting
        `RuntimeWarning: floating point number truncated to an integer`. Raising before
        the call is what keeps that silent re-definition out of the caller.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self.assertRaises(ValueError, exact_indel_tail, 3, 10.5, 0.001)
        self.assertEqual([], [str(entry.message) for entry in caught])

    def test_a_float_trial_count_is_rejected_even_when_it_is_integral(self):
        """`10.0` needs no truncation, but a float `N` is exactly how the rejected
        `ru_bp_coverage / ru_length` ratio (SPEC 3.1) would arrive. The contract is an
        integer count, so the type is checked, not the value."""
        with self.assertRaises(ValueError):
            exact_indel_tail(3, 10.0, 0.001)

    def test_a_float_support_is_rejected_too(self):
        with self.assertRaises(ValueError):
            exact_indel_tail(3.0, 10, 0.001)

    def test_negative_counts_are_rejected(self):
        self.assertRaises(ValueError, exact_indel_tail, -1, 10, 0.001)
        self.assertRaises(ValueError, exact_indel_tail, 3, -10, 0.001)

    def test_a_probability_outside_the_unit_interval_is_rejected(self):
        self.assertRaises(ValueError, exact_indel_tail, 3, 10, 1.5)
        self.assertRaises(ValueError, exact_indel_tail, 3, 10, -0.001)

    def test_a_nan_background_probability_is_rejected(self):
        with self.assertRaises(ValueError):
            exact_indel_tail(3, 10, float('nan'))

    def test_a_boolean_is_not_an_acceptable_count(self):
        """`True` is an `int` in Python 2, so `N=True` would silently mean `N=1`."""
        self.assertRaises(ValueError, exact_indel_tail, 1, True, 0.001)


class TestExactTailBoundaries(unittest.TestCase):
    def test_zero_support_is_certain(self):
        """`P(K >= 0) = 1` for every `N` and every `p0`, including the degenerate ones."""
        self.assertEqual(exact_indel_tail(0, 10, 0.001), 1.0)
        self.assertEqual(exact_indel_tail_log(0, 10, 0.001), 0.0)
        self.assertEqual(exact_indel_tail(0, 0, 0.0), 1.0)
        self.assertEqual(exact_indel_tail(0, 10, 1.0), 1.0)

    def test_no_opportunities_admits_only_zero_support(self):
        self.assertEqual(exact_indel_tail(0, 0, 0.001), 1.0)
        self.assertRaises(ValueError, exact_indel_tail, 1, 0, 0.001)

    def test_support_equal_to_opportunities_is_p0_to_the_power_n(self):
        self.assertAlmostEqual(exact_indel_tail(3, 3, 0.5), 0.125)
        self.assertAlmostEqual(exact_indel_tail(4, 4, 0.001), 1e-12)

    def test_a_zero_background_makes_any_support_impossible(self):
        """A genuine zero, reported as `-inf` in log space rather than clamped."""
        self.assertEqual(exact_indel_tail_log(1, 10, 0.0), float('-inf'))
        self.assertEqual(exact_indel_tail(1, 10, 0.0), 0.0)
        self.assertTrue(tail_below_cutoff(1, 10, 0.0, 0.001))

    def test_a_certain_background_makes_any_support_unremarkable(self):
        self.assertEqual(exact_indel_tail(4, 10, 1.0), 1.0)
        self.assertEqual(exact_indel_tail_log(4, 10, 1.0), 0.0)
        self.assertFalse(tail_below_cutoff(10, 10, 1.0, 0.001))


class TestExactTailValues(unittest.TestCase):
    def test_the_tail_matches_scipy_where_scipy_is_reliable(self):
        """`P(K >= k) = sf(k - 1)`. Pinned against the shipped library at a depth it
        still represents, so the summation fallback below cannot drift unnoticed."""
        from scipy.stats import binom
        for k, n, p0 in [(3, 10, 0.001), (1, 100, 0.01), (7, 20, 0.3), (60, 100, 0.001)]:
            self.assertAlmostEqual(exact_indel_tail_log(k, n, p0),
                                   math.log(binom.sf(k - 1, n, p0)), places=9)

    def test_the_tail_is_monotone_in_support(self):
        previous = 1.0
        for k in range(0, 11):
            current = exact_indel_tail(k, 10, 0.01)
            self.assertLessEqual(current, previous)
            previous = current

    def test_no_alternative_rate_is_accepted(self):
        """SPEC 3.1: the exact one-sided test has no alternative rate at all. A fourth
        positional argument is the shipped `expected_indel_transitions` trying to get in.
        """
        self.assertRaises(TypeError, exact_indel_tail, 3, 10, 0.001, 0.1)


class TestDeepTail(unittest.TestCase):
    """SPEC 3.1: `binom.logsf` underflows to `-inf` at deep tails in SciPy 1.2.1, and
    that must be a valid strong result, not an error and not a clamp to a tiny positive
    number."""

    def test_scipy_really_does_lose_the_deep_tail(self):
        """Measured on the pinned build: `binom.sf(199, 1000, 1e-4)` is exactly `0.0`
        and `binom.logsf(199, 1000, 1e-4)` is `-inf` with a divide-by-zero
        RuntimeWarning. This is the premise of everything below it."""
        from scipy.stats import binom
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.assertEqual(binom.sf(199, 1000, 1e-4), 0.0)
            self.assertEqual(binom.logsf(199, 1000, 1e-4), float('-inf'))

    def test_the_deep_tail_stays_finite_in_log_space(self):
        """The naive route (`log(sf)`, or `1 - cdf`) returns `-inf` here and throws away
        every distinction between a strong result and an astronomically strong one."""
        log_tail = exact_indel_tail_log(200, 1000, 1e-4)
        self.assertTrue(log_tail < -1000.0)
        self.assertFalse(math.isinf(log_tail))
        # The first term dominates a tail this far out, so the whole sum sits just above
        # `logpmf(k)` -- a bound the summation cannot pass without being wrong.
        from scipy.stats import binom
        anchor = binom.logpmf(200, 1000, 1e-4)
        self.assertLessEqual(anchor, log_tail)
        self.assertLess(log_tail, anchor + 1e-3)

    def test_the_deep_tail_underflows_to_zero_as_a_probability(self):
        """SPEC 3.1 forbids promising all reported p-values remain nonzero. The
        probability form says `0.0` honestly; the log form carries the information."""
        self.assertEqual(exact_indel_tail(200, 1000, 1e-4), 0.0)

    def test_a_deep_tail_is_below_any_cutoff_and_is_not_an_error(self):
        self.assertTrue(tail_below_cutoff(200, 1000, 1e-4, 0.001))
        self.assertTrue(tail_below_cutoff(200, 1000, 1e-4, 1e-300))

    def test_computing_a_deep_tail_emits_no_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            exact_indel_tail_log(200, 1000, 1e-4)
        self.assertEqual([], [str(entry.message) for entry in caught])

    def test_the_deep_tail_agrees_with_an_independent_exact_computation(self):
        """60 decimal digits of `sum_{j>=k} C(n,j) p^j (1-p)^(n-j)`, computed here with
        `decimal` and no SciPy at all, so the summation is checked against arithmetic
        rather than against another floating-point route with the same failure mode."""
        for k, n, p0 in [(200, 1000, 1e-4), (400, 2000, 1e-4)]:
            self.assertAlmostEqual(exact_indel_tail_log(k, n, p0),
                                   float(_reference_log_tail(k, n, p0)), places=9)

    def test_the_summation_agrees_with_scipy_at_a_depth_scipy_still_reaches(self):
        """`sf(59, 100, 0.001)` is 1.3215646717578752e-152 on the pinned build -- deep,
        but still representable, so the two routes must agree there."""
        from advntr import exact_tail
        self.assertAlmostEqual(exact_tail._log_tail_by_summation(60, 100, 0.001),
                               exact_indel_tail_log(60, 100, 0.001), places=9)


class TestDecisionInLogSpace(unittest.TestCase):
    def test_the_decision_is_taken_against_a_log_cutoff(self):
        self.assertTrue(tail_below_cutoff(7, 20, 0.001, 0.001))
        self.assertFalse(tail_below_cutoff(1, 20, 0.001, 0.001))

    def test_a_cutoff_outside_the_unit_interval_is_rejected(self):
        self.assertRaises(ValueError, tail_below_cutoff, 3, 10, 0.001, 0.0)
        self.assertRaises(ValueError, tail_below_cutoff, 3, 10, 0.001, 1.5)

    def test_the_probability_form_never_disagrees_with_the_decision(self):
        """The readable value is derived from the log value, so they cannot diverge."""
        for k, n, p0 in [(3, 10, 0.001), (2, 5, 0.2), (9, 10, 0.5)]:
            probability = exact_indel_tail(k, n, p0)
            self.assertAlmostEqual(math.log(probability), exact_indel_tail_log(k, n, p0),
                                   places=9)


if __name__ == '__main__':
    unittest.main()
