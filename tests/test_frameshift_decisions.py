"""Run-local frameshift policy and shared production decisions."""
import importlib
import pickle
import sys
import unittest

from scipy.stats import binom

from advntr import exact_caller
from advntr import settings
from advntr import background_estimator
from advntr.frameshift_decisions import (
    FrameshiftPolicy,
    exact_call,
    legacy_call,
    passes_support,
    resolve_policy,
)
from advntr.genome_analyzer import GenomeAnalyzer
from advntr.vntr_finder import VNTRFinder


class TestFrameshiftPolicy(unittest.TestCase):

    def test_defaults_equal_the_shipped_caller_constants(self):
        policy = resolve_policy()

        self.assertEqual(settings.INDEL_MUTATION_MIN_PVALUE, policy.cutoff)
        self.assertEqual(settings.MIN_SUPPORTING_READ_COUNT,
                         policy.minimum_read_support)
        self.assertIsInstance(policy.cutoff, float)
        self.assertIsInstance(policy.minimum_read_support, int)

    def test_explicit_policies_are_isolated_and_do_not_mutate_settings(self):
        before = (settings.INDEL_MUTATION_MIN_PVALUE,
                  settings.MIN_SUPPORTING_READ_COUNT)
        recipe_before = dict(background_estimator.HYPERPARAMETERS)

        permissive = resolve_policy(0.01, 1)
        conservative = resolve_policy(0.0001, 7)

        self.assertEqual((0.01, 1), tuple(permissive))
        self.assertEqual((0.0001, 7), tuple(conservative))
        self.assertEqual(before, (settings.INDEL_MUTATION_MIN_PVALUE,
                                  settings.MIN_SUPPORTING_READ_COUNT))
        self.assertEqual(recipe_before, background_estimator.HYPERPARAMETERS)
        self.assertEqual(0.001, recipe_before['floor_target'])
        self.assertEqual(4, recipe_before['kprot'])

    def test_policy_is_immutable(self):
        policy = resolve_policy()

        with self.assertRaises(AttributeError):
            policy.cutoff = 0.5

    def test_policy_pickle_roundtrip_preserves_values_and_decisions(self):
        policy = resolve_policy(0.0002, 5)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            restored = pickle.loads(pickle.dumps(policy, protocol))
            self.assertIsInstance(restored, FrameshiftPolicy)
            self.assertEqual(policy, restored)
            self.assertFalse(passes_support(4, restored))
            self.assertTrue(passes_support(5, restored))
            self.assertFalse(legacy_call(0.0002, restored))
            self.assertTrue(legacy_call(0.0001, restored))

    def test_cutoff_rejects_nonfloat_bool_nonfinite_and_endpoints(self):
        for value in (True, 1, 0.0, 1.0, -0.1, float('nan'),
                      float('inf'), float('-inf')):
            with self.assertRaisesRegexp(ValueError, 'cutoff'):
                resolve_policy(value, 3)

    def test_support_rejects_bool_noninteger_and_values_below_one(self):
        for value in (True, 0, -1, 1.0, float('nan')):
            with self.assertRaisesRegexp(ValueError, 'support'):
                resolve_policy(0.001, value)

    def test_constructor_validates_just_like_the_resolver(self):
        with self.assertRaisesRegexp(ValueError, 'cutoff'):
            FrameshiftPolicy(float('nan'), 3)
        with self.assertRaisesRegexp(ValueError, 'support'):
            FrameshiftPolicy(0.001, False)

    def test_namedtuple_factories_cannot_bypass_validation(self):
        with self.assertRaisesRegexp(ValueError, 'cutoff'):
            FrameshiftPolicy._make((float('nan'), 3))
        with self.assertRaisesRegexp(ValueError, 'support'):
            resolve_policy()._replace(minimum_read_support=0)
        self.assertEqual((0.01, 3), tuple(resolve_policy()._replace(cutoff=0.01)))
        with self.assertRaises(ValueError):
            resolve_policy()._replace(unknown=1)

    def test_direct_analyzer_boundary_rejects_invalid_policy_without_finders(self):
        with self.assertRaisesRegexp(ValueError, 'policy'):
            GenomeAnalyzer([], [], frameshift_policy=(0.001, 3))

    def test_direct_finder_boundary_rejects_invalid_policy(self):
        class Reference(object):
            start_point = 100

            @staticmethod
            def get_genomic_end():
                return 200

        with self.assertRaisesRegexp(ValueError, 'policy'):
            VNTRFinder(Reference(), frameshift_policy=(0.001, 3))

    def test_forged_tuple_is_rejected_at_decision_boundary(self):
        forged = tuple.__new__(FrameshiftPolicy, (0.001, 0))
        with self.assertRaisesRegexp(ValueError, 'support'):
            passes_support(0, forged)

    def test_analyzer_passes_one_policy_object_to_every_finder(self):
        class Reference(object):
            id = 1
            start_point = 100

            @staticmethod
            def get_genomic_end():
                return 200

        policy = resolve_policy(0.0002, 5)

        analyzer = GenomeAnalyzer([Reference()], [1],
                                  frameshift_policy=policy)

        self.assertIs(policy, analyzer.frameshift_policy)
        self.assertIs(policy, analyzer.vntr_finder[1].frameshift_policy)


class TestProductionDecisions(unittest.TestCase):

    def test_support_gate_is_inclusive_at_the_minimum(self):
        policy = resolve_policy(0.001, 3)

        self.assertFalse(passes_support(2, policy))
        self.assertTrue(passes_support(3, policy))
        self.assertTrue(passes_support(4, policy))
        with self.assertRaisesRegexp(ValueError, 'read support'):
            passes_support(True, policy)

    def test_legacy_cutoff_is_strict(self):
        policy = resolve_policy(0.001, 3)

        self.assertTrue(legacy_call(0.000999999, policy))
        self.assertFalse(legacy_call(0.001, policy))
        self.assertFalse(legacy_call(0.001000001, policy))
        with self.assertRaisesRegexp(ValueError, 'p-value'):
            legacy_call(float('inf'), policy)

    def test_legacy_nan_preserves_the_existing_noncall(self):
        pvalue = VNTRFinder.identify_frameshift(float('nan'), 3, 0.495)[2]
        policy = resolve_policy()

        self.assertNotEqual(pvalue, pvalue)
        self.assertFalse(pvalue < policy.cutoff)
        self.assertFalse(legacy_call(pvalue, policy))

    def test_exact_cutoff_is_strict_and_matches_scipy(self):
        tail = float(binom.sf(0, 1, 0.25))
        self.assertEqual(0.25, tail)

        self.assertFalse(exact_call(1, 1, 0.25,
                                    resolve_policy(tail, 1)))
        self.assertTrue(exact_call(1, 1, 0.25,
                                   resolve_policy(tail + 1e-16, 1)))

    def test_exact_decision_survives_probability_underflow(self):
        self.assertEqual(0.0, float(binom.sf(199, 1000, 0.0001)))

        self.assertTrue(exact_call(200, 1000, 0.0001,
                                   resolve_policy(1e-300, 1)))

    def test_exact_caller_validates_policy_before_empty_evidence(self):
        with self.assertRaisesRegexp(ValueError, 'policy'):
            exact_caller.decide({}, 'D1_1', None, policy=(0.001, 3))

    def test_exact_caller_retains_the_scalar_cutoff_interface(self):
        class Background(object):
            @staticmethod
            def probability_for(_state):
                return 0.25

        records = {'D1_1': {
            'opportunities': 1,
            'state_identities': {'D1_1': [('read', 0)]},
        }}

        self.assertEqual((True, 0.25),
                         exact_caller.decide(records, 'D1_1', Background(), 0.3))
        self.assertEqual((False, 0.25),
                         exact_caller.decide(records, 'D1_1', Background(), 0.25))
        with self.assertRaisesRegexp(ValueError, 'cutoff or policy'):
            exact_caller.decide(records, 'D1_1', Background(), 0.3,
                                policy=resolve_policy())


class TestCommandLinePolicy(unittest.TestCase):

    def _parse_genotype(self, extra):
        cli = importlib.import_module('advntr.__main__')
        original_argv = sys.argv
        original_genotype = cli.genotype
        captured = []
        try:
            sys.argv = ['advntr', 'genotype'] + extra
            cli.genotype = lambda args, _parser: captured.append(args)
            cli.main()
        finally:
            cli.genotype = original_genotype
            sys.argv = original_argv
        return captured[0]

    def test_cli_defaults_resolve_to_the_shipped_policy(self):
        args = self._parse_genotype([])

        self.assertIsNone(args.frameshift_pvalue_cutoff)
        self.assertIsNone(args.min_frameshift_read_support)
        self.assertEqual(resolve_policy(), resolve_policy(
            args.frameshift_pvalue_cutoff,
            args.min_frameshift_read_support))

    def test_cli_parses_explicit_float_and_integer_values(self):
        args = self._parse_genotype([
            '--frameshift-pvalue-cutoff', '0.0002',
            '--min-frameshift-read-support', '5',
        ])

        self.assertEqual(0.0002, args.frameshift_pvalue_cutoff)
        self.assertIsInstance(args.frameshift_pvalue_cutoff, float)
        self.assertEqual(5, args.min_frameshift_read_support)
        self.assertIsInstance(args.min_frameshift_read_support, int)


if __name__ == '__main__':
    unittest.main()
