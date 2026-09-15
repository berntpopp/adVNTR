"""Run-local policy primitives use invented values and no read/model assets."""
import copy
import json
import pickle
import unittest

from advntr import capture_policy, settings, utils
from advntr.vntr_finder import VNTRFinder


class TestCapturePolicy(unittest.TestCase):

    def test_defaults_match_actual_fresh_process_and_bound_values(self):
        policy = capture_policy.resolve_capture_policy()
        expected = {
            'minimum_read_length': settings.MIN_READ_LENGTH,
            'minimum_read_match_ratio': settings.MIN_READ_MATCH_RATIO,
            'minimum_relative_ru_coverage': settings.MIN_RELATIVE_RU_COVERAGE,
            'use_reference_alignment': settings.USE_REF_ALIGNMENT,
            'fully_covered_ru_only': settings.USE_ONLY_FULLY_COVERED_RU,
            'filter_adapter_readthrough': settings.FILTER_ADAPTER_READTHROUGH,
            'prune_reverse': settings.PRUNE_REVERSE_DECODE,
            'maximum_error_rate': settings.MAX_ERROR_RATE,
            'legacy_error_rate': VNTRFinder.identify_frameshift.func_defaults[0],
            'mapq_cutoff': utils.MAPQ_CUTOFF,
            'base_quality_cutoff': utils.QUALITY_SCORE_CUTOFF,
            'maximum_low_quality_fraction': utils.LOW_QUALITY_BP_TO_DISCARD_READ,
            'enhanced_hmm': settings.USE_ENHANCED_HMM,
            'trained_hmms': settings.USE_TRAINED_HMMS,
        }
        for name, value in expected.items():
            self.assertEqual(value, getattr(policy, name), name)
        self.assertEqual(1, policy.threads)  # CLI default, not host-dependent CORES.
        self.assertEqual('illumina', policy.platform)
        self.assertEqual('legacy', policy.caller_mode)
        self.assertFalse(policy.frameshift_mode)
        self.assertFalse(policy.is_haploid)

    def test_repeated_resolution_cannot_inherit_prior_options_or_mutable_settings(self):
        baseline = capture_policy.resolve_capture_policy()
        changed = capture_policy.resolve_capture_policy(
            minimum_read_length=50, minimum_read_match_ratio=0.8,
            minimum_relative_ru_coverage=0.5, use_reference_alignment=False,
            fully_covered_ru_only=True, filter_adapter_readthrough=True,
            prune_reverse=True, threads=2, platform='nanopore')
        self.assertNotEqual(baseline, changed)
        previous = settings.MIN_READ_LENGTH, utils.MAPQ_CUTOFF
        try:
            settings.MIN_READ_LENGTH, utils.MAPQ_CUTOFF = 99, 99
            self.assertEqual(baseline, capture_policy.resolve_capture_policy())
        finally:
            settings.MIN_READ_LENGTH, utils.MAPQ_CUTOFF = previous
        self.assertEqual(0.3, changed.maximum_error_rate)
        self.assertEqual(0.05, baseline.maximum_error_rate)

    def test_distinct_caller_and_workflow_modes_are_bound(self):
        exact = capture_policy.resolve_capture_policy(caller_mode='exact', frameshift_mode=True)
        self.assertEqual('exact', exact.caller_mode)
        self.assertTrue(exact.frameshift_mode)
        self.assertEqual(0.3, capture_policy.resolve_capture_policy(platform='pacbio').maximum_error_rate)
        self.assertEqual(0.2, capture_policy.resolve_capture_policy(
            platform='pacbio', maximum_error_rate=0.2).maximum_error_rate)
        for changes in ({'platform': 'other'}, {'caller_mode': 'other'},
                        {'caller_mode': 'exact', 'frameshift_mode': False}):
            with self.assertRaises(ValueError):
                capture_policy.resolve_capture_policy(**changes)

    def test_raw_runtime_domains_are_not_silently_narrowed_to_calibrated_v2(self):
        for ratio, rare in ((None, None), (0.0, 0.0), (1.0, 2.0)):
            policy = capture_policy.resolve_capture_policy(
                minimum_read_match_ratio=ratio, minimum_relative_ru_coverage=rare)
            self.assertEqual(ratio, policy.minimum_read_match_ratio)
            self.assertEqual(rare, policy.minimum_relative_ru_coverage)
            self.assertTrue(capture_policy.calibrated_v2_domain_errors(policy))
        compatible = capture_policy.resolve_capture_policy(
            minimum_read_match_ratio=0.5, minimum_relative_ru_coverage=None)
        self.assertEqual((), capture_policy.calibrated_v2_domain_errors(compatible))
        self.assertEqual((), capture_policy.calibrated_v2_domain_errors(
            compatible._replace(minimum_read_match_ratio=1.0, minimum_relative_ru_coverage=1.0)))
        self.assertEqual(('minimum_read_match_ratio', 'minimum_relative_ru_coverage'),
                         capture_policy.calibrated_v2_domain_errors(
                             compatible._replace(minimum_read_match_ratio=0.0,
                                                 minimum_relative_ru_coverage=0.0)))

    def test_boolean_fields_require_real_booleans(self):
        fields = ('frameshift_mode', 'is_haploid', 'prune_reverse',
                  'filter_adapter_readthrough', 'use_reference_alignment',
                  'fully_covered_ru_only', 'enhanced_hmm', 'trained_hmms')
        for field in fields:
            for value in (0, 1, None, 'false'):
                with self.assertRaises(ValueError):
                    capture_policy.resolve_capture_policy(**{field: value})

    def test_integer_domains_reject_bool_fraction_negative_and_zero_where_required(self):
        for field in ('threads', 'minimum_read_length', 'mapq_cutoff', 'base_quality_cutoff'):
            for value in (True, 1.5, -1):
                with self.assertRaises(ValueError):
                    capture_policy.resolve_capture_policy(**{field: value})
        for field in ('threads', 'minimum_read_length'):
            with self.assertRaises(ValueError):
                capture_policy.resolve_capture_policy(**{field: 0})
        self.assertEqual(0, capture_policy.resolve_capture_policy(base_quality_cutoff=0).base_quality_cutoff)

    def test_float_domains_reject_bool_nonfloat_nonfinite_and_outside_range(self):
        fields = ('minimum_read_match_ratio', 'minimum_relative_ru_coverage',
                  'maximum_error_rate', 'legacy_error_rate', 'maximum_low_quality_fraction')
        for field in fields:
            for value in (True, 1, -0.1, float('nan'), float('inf'), float('-inf')):
                with self.assertRaises(ValueError):
                    capture_policy.resolve_capture_policy(**{field: value})
        for field in fields:
            if field != 'minimum_relative_ru_coverage':
                with self.assertRaises(ValueError):
                    capture_policy.resolve_capture_policy(**{field: 1.1})
        for field in ('legacy_error_rate', 'maximum_low_quality_fraction'):
            with self.assertRaises(ValueError):
                capture_policy.resolve_capture_policy(**{field: None})

    def test_unsupported_backend_and_trained_mode_fail_in_pure_preflight(self):
        for changes in ({'enhanced_hmm': False}, {'trained_hmms': True}):
            with self.assertRaisesRegexp(ValueError, 'unsupported'):
                capture_policy.resolve_capture_policy(**changes)

    def test_closed_immutable_constructor_factories_and_pickle(self):
        policy = capture_policy.resolve_capture_policy(minimum_read_match_ratio=0.8)
        with self.assertRaises(AttributeError):
            policy.threads = 2
        with self.assertRaises((TypeError, ValueError)):
            capture_policy.resolve_capture_policy(unknown=1)
        with self.assertRaises(ValueError):
            policy._replace(unknown=1)
        with self.assertRaises(ValueError):
            policy._replace(threads=0)
        values = list(policy)
        values[policy._fields.index('threads')] = False
        with self.assertRaises(ValueError):
            type(policy)._make(values)
        forged = tuple.__new__(type(policy), values)
        with self.assertRaises(ValueError):
            capture_policy.validate_capture_policy(forged)
        with self.assertRaises(ValueError):
            capture_policy.validate_capture_policy(tuple.__new__(type(policy), ()))
        with self.assertRaises(ValueError):
            capture_policy.validate_capture_policy(tuple(policy))
        with self.assertRaises(TypeError):
            type(policy)(*tuple(policy), unknown=1)
        self.assertEqual(policy, type(policy)._make(policy))
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.assertRaises(ValueError):
                pickle.loads(pickle.dumps(forged, protocol))
        self.assertIs(policy, capture_policy.validate_capture_policy(policy))
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            restored = pickle.loads(pickle.dumps(policy, protocol))
            self.assertIsInstance(restored, type(policy))
            self.assertEqual(policy, restored)
            self.assertIs(restored, capture_policy.validate_capture_policy(restored))

    def test_document_roundtrip_is_closed_complete_and_rejects_version_type_forgery(self):
        policy = capture_policy.resolve_capture_policy()
        document = capture_policy.policy_document(policy)
        self.assertEqual({'schema_version', 'parameters'}, set(document))
        self.assertEqual('advntr-runtime-capture-policy-v1', document['schema_version'])
        self.assertEqual(set(policy._fields), set(document['parameters']))
        self.assertEqual(policy, capture_policy.policy_from_document(json.loads(json.dumps(document))))
        for changes in ({'schema_version': 'unknown'}, {'extra': 1}, {'parameters': []}):
            changed = dict(document, **changes)
            with self.assertRaises(ValueError):
                capture_policy.policy_from_document(changed)
        for parameters in (dict(document['parameters'], extra=1),
                           dict(document['parameters'], threads=True),
                           dict(document['parameters'], maximum_error_rate=None)):
            changed = dict(document, parameters=parameters)
            with self.assertRaises(ValueError):
                capture_policy.policy_from_document(changed)
        missing = copy.deepcopy(document)
        del missing['parameters']['minimum_read_length']
        with self.assertRaises(ValueError):
            capture_policy.policy_from_document(missing)
        for value in (None, [], 'policy'):
            with self.assertRaises(ValueError):
                capture_policy.policy_from_document(value)


if __name__ == '__main__':
    unittest.main()
