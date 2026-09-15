"""Closed replay policy rejects capture changes before any outcomes are decoded."""
import copy
import unittest

from advntr.capture_policy import policy_document, resolve_capture_policy


def replay_policy(mode='legacy', cutoff=0.001, support=3):
    return {'schema_version': 'advntr-frameshift-replay-policy-v1',
            'capture_policy': policy_document(resolve_capture_policy(frameshift_mode=True, caller_mode=mode)),
            'caller_policy': {'schema_version': 'advntr-frameshift-policy-v1', 'mode': mode,
                              'cutoff': cutoff, 'minimum_read_support': support}}


class TestReplayPolicy(unittest.TestCase):
    def test_closed_policy_preserves_runtime_domain_without_coercion(self):
        from advntr.frameshift_replay_policy import decode_replay_policy, caller_policy_document
        for mode in ('legacy', 'exact'):
            raw = replay_policy(mode)
            decoded = decode_replay_policy(raw)
            self.assertEqual(raw['caller_policy'], caller_policy_document(decoded.capture, decoded.frameshift))
            self.assertIsNone(decoded.capture.minimum_read_match_ratio)
            self.assertEqual(3, decoded.frameshift.minimum_read_support)

    def test_ambiguous_modes_invalid_numeric_types_and_unknown_fields_fail(self):
        from advntr.frameshift_replay_policy import decode_replay_policy
        for field, value in (('schema_version', 'old'), ('mode', 'unknown'), ('mode', 'exact'),
                              ('cutoff', 0), ('cutoff', float('nan')), ('cutoff', True),
                              ('minimum_read_support', 3.0), ('extra', None)):
            raw = replay_policy()
            raw['caller_policy'][field] = value
            with self.assertRaises(ValueError):
                decode_replay_policy(raw)
        for raw in (None, {}, dict(replay_policy(), extra=True), dict(replay_policy(), schema_version='old')):
            with self.assertRaises(ValueError):
                decode_replay_policy(raw)

    def test_only_caller_mode_can_change_without_recapture_and_refit(self):
        from advntr.frameshift_replay_policy import decode_replay_policy, require_replay_compatible
        original = decode_replay_policy(replay_policy()).capture
        requested = decode_replay_policy(replay_policy('exact', 0.05, 1)).capture
        self.assertIsNone(require_replay_compatible(original, requested))
        for field, value in (('minimum_relative_ru_coverage', 0.1), ('filter_adapter_readthrough', True),
                              ('minimum_read_match_ratio', 0.6), ('prune_reverse', True), ('threads', 2),
                              ('fully_covered_ru_only', True), ('maximum_error_rate', 0.1)):
            with self.assertRaisesRegexp(ValueError, 'recapture and background refit'):
                require_replay_compatible(original, original._replace(**{field: value}))

    def test_replay_cannot_claim_a_non_frameshift_or_long_read_capture(self):
        from advntr.frameshift_replay_policy import decode_replay_policy
        for changed in ({'frameshift_mode': False}, {'platform': 'pacbio'}):
            raw = replay_policy()
            raw['capture_policy'] = policy_document(resolve_capture_policy(**dict({'frameshift_mode': True}, **changed)))
            with self.assertRaises(ValueError):
                decode_replay_policy(raw)

    def test_explicit_null_caller_values_cannot_request_implicit_defaults(self):
        from advntr.frameshift_replay_policy import decode_replay_policy
        for field in ('cutoff', 'minimum_read_support'):
            raw = replay_policy()
            raw['caller_policy'][field] = None
            with self.assertRaises(ValueError):
                decode_replay_policy(raw)
