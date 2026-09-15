"""Invented HMMs pin probability bytes and explicit per-build error-rate isolation."""
import hashlib
import json
import math
import struct
import unittest

from advntr import hmm_construction, hmm_utils, settings
from hmm.base import State
from advntr.capture_policy import resolve_capture_policy


BUILDERS = (
    ('get_prefix_matcher_hmm', ('ACG',)),
    ('get_suffix_matcher_hmm', ('ACG',)),
    ('get_constant_number_of_repeats_matcher_hmm', (['ACG'], 2, None)),
    ('get_repeat_matcher_enhanced_hmm', ([['ACG']], 2, None)),
    ('get_read_matcher_model_enhanced', ('TGC', 'GCA', ['ACG'], 2, None, True)),
)


def _hex(value):
    return struct.pack('>d', float(value)).encode('hex')


def _fingerprint(model):
    states = []
    for state in model.states:
        emissions = None if state.distribution is None else [
            _hex(state.distribution[index]) for index in range(4)]
        states.append((state.name, emissions))
    transitions = [[_hex(value) for value in row] for row in model.dense_transition_matrix()]
    payload = (states, transitions, _hex(model.dp_score_threshold),
               model.read_length_used_to_build_model)
    return hashlib.sha256(json.dumps(payload, separators=(',', ':'))).hexdigest()


def _transition(model, source, target):
    states = dict((state.name, state) for state in model.states)
    return model.transition_map[states[source]][states[target]]


# Recorded before extraction from five working helpers at maximum error rate0.05.
# The hash covers ordered states, every emission/transition bit and bake thresholds.
DEFAULT_FINGERPRINTS = {
    'get_prefix_matcher_hmm': '1ec1c5ab5bf3fc2b221b20e8d92cd3a0074670b24d8daee7255febbb485a089e',
    'get_suffix_matcher_hmm': '8fd6368e96b75bd64ac25913d0a1b25ad936a4087ce6e5d03b66fcbc5bd3647b',
    'get_constant_number_of_repeats_matcher_hmm': '56888c7d7f8970e2bdf6df4286714d7bd434567f259c57d514c91fa47b262ea9',
    'get_repeat_matcher_enhanced_hmm': '04e0ea6c58fa980d0337af742db3bf9f68f9fd22cfdbee3ed16d83d85abbdc8a',
    'get_read_matcher_model_enhanced': '1bce4507510c4da305c36e3d9542e1253766d062bda7409848a77423bfb8ad84',
}


class TestHMMConstructionPolicy(unittest.TestCase):

    def setUp(self):
        self.old_rate = settings.MAX_ERROR_RATE
        settings.MAX_ERROR_RATE = 0.05

    def tearDown(self):
        settings.MAX_ERROR_RATE = self.old_rate

    def test_default_probabilities_remain_byte_identical_to_pre_extraction(self):
        for name, arguments in BUILDERS:
            model = getattr(hmm_utils, name)(*arguments)
            self.assertEqual(DEFAULT_FINGERPRINTS[name], _fingerprint(model), name)

    def test_explicit_rate_reaches_every_working_builder_without_mutating_settings(self):
        for name, arguments in BUILDERS:
            builder = getattr(hmm_utils, name)
            low = builder(*arguments, maximum_error_rate=0.05)
            high = builder(*arguments, maximum_error_rate=0.3)
            self.assertNotEqual(_fingerprint(low), _fingerprint(high), name)
            self.assertEqual(DEFAULT_FINGERPRINTS[name], _fingerprint(low), name)
            self.assertEqual(0.05, settings.MAX_ERROR_RATE)

    def test_mixed_platform_builds_and_ambient_changes_cannot_influence_explicit_rate(self):
        builder = hmm_utils.get_read_matcher_model_enhanced
        arguments = BUILDERS[-1][1]
        low = resolve_capture_policy(platform='illumina')
        high = resolve_capture_policy(platform='pacbio')
        baseline = _fingerprint(builder(*arguments, maximum_error_rate=low.maximum_error_rate))
        for policy in (high, low, high, low):
            settings.MAX_ERROR_RATE = 0.9
            result = _fingerprint(builder(*arguments, maximum_error_rate=policy.maximum_error_rate))
            self.assertEqual(policy == low, result == baseline)
        self.assertEqual(DEFAULT_FINGERPRINTS[BUILDERS[-1][0]], baseline)

    def test_flank_transition_probabilities_use_explicit_rate_with_inherited_association(self):
        for rate in (0.05, 0.3):
            prefix = hmm_utils.get_prefix_matcher_hmm('ACG', maximum_error_rate=rate)
            suffix = hmm_utils.get_suffix_matcher_hmm('ACG', maximum_error_rate=rate)
            self.assertEqual(rate * 2 / 5, _transition(prefix, 'prefix_start_prefix', 'I0_prefix'))
            self.assertEqual(rate * 1 / 5, _transition(prefix, 'prefix_start_prefix', 'D1_prefix'))
            self.assertEqual((1 - rate * 2 / 5 - rate * 1 / 5) / 3,
                             _transition(suffix, 'suffix_start_suffix', 'M1_suffix'))

    def test_explicit_invalid_rates_are_rejected_before_building_models(self):
        for name, arguments in BUILDERS:
            for rate in (True, 1, -0.1, 1.1, float('nan'), float('inf')):
                with self.assertRaises(ValueError):
                    getattr(hmm_utils, name)(*arguments, maximum_error_rate=rate)


    def test_compatibility_defaults_still_follow_the_explicit_legacy_setting(self):
        for name, arguments in BUILDERS:
            settings.MAX_ERROR_RATE = 0.3
            builder = getattr(hmm_utils, name)
            self.assertEqual(_fingerprint(builder(*arguments, maximum_error_rate=0.3)),
                             _fingerprint(builder(*arguments)))
        settings.MAX_ERROR_RATE = 0.05
        self.assertEqual(DEFAULT_FINGERPRINTS[BUILDERS[-1][0]],
                         _fingerprint(hmm_utils.get_read_matcher_model_enhanced(*BUILDERS[-1][1])))

    def test_repeat_emission_probability_matches_independent_pseudocount_formula(self):
        for rate in (0.05, 0.3):
            model = hmm_utils.get_repeat_matcher_enhanced_hmm([['ACG']], 2, None, maximum_error_rate=rate)
            match = next(state for state in model.states if state.name == 'M1_1')
            pseudocount = rate / 40
            self.assertAlmostEqual((1 + pseudocount) / (1 + 4 * pseudocount),
                                   math.exp(match.distribution[0]), places=14)

    def test_alignment_callback_cannot_change_the_frozen_rate_mid_build(self):
        def alignment(_vpaths):
            settings.MAX_ERROR_RATE = 0.9
            return ['ACG']
        model = hmm_construction.get_read_matcher_model_enhanced(
            'TGC', 'GCA', ['ACG'], 2, ['invented-path'], True, 0.05,
            alignment_from_reads=alignment)
        self.assertEqual(0.9, settings.MAX_ERROR_RATE)
        self.assertEqual(DEFAULT_FINGERPRINTS['get_read_matcher_model_enhanced'], _fingerprint(model))
        constant = hmm_construction.get_constant_number_of_repeats_matcher_hmm(
            ['ACG'], 2, ['invented-path'], 0.05, alignment_from_reads=alignment)
        self.assertEqual(DEFAULT_FINGERPRINTS['get_constant_number_of_repeats_matcher_hmm'], _fingerprint(constant))

    def test_wrapper_reconstructs_alignment_from_invented_read_states(self):
        names = ('start', 'unit_start_1', 'M1_1', 'M2_1', 'M3_1', 'unit_end_1', 'end')
        vpaths = [('ACG', [(index, State(name=name)) for index, name in enumerate(names)])]
        for name, arguments in BUILDERS[2:]:
            arguments = list(arguments)
            arguments[2 if name != 'get_read_matcher_model_enhanced' else 4] = vpaths
            result = getattr(hmm_utils, name)(*arguments, maximum_error_rate=0.05)
            self.assertEqual(DEFAULT_FINGERPRINTS[name], _fingerprint(result), name)

    def test_core_requires_rate_and_an_alignment_callback_for_supplied_paths(self):
        with self.assertRaises(TypeError):
            hmm_construction.get_prefix_matcher_hmm('ACG')
        for function, arguments in (
                (hmm_construction.get_constant_number_of_repeats_matcher_hmm, (['ACG'], 2, ['path'], 0.05)),
                (hmm_construction.get_repeat_matcher_enhanced_hmm, ([['ACG']], 2, ['path'], 0.05))):
            with self.assertRaisesRegexp(ValueError, 'alignment_from_reads'):
                function(*arguments)

    def test_nonframeshift_and_multiple_cluster_builds_keep_explicit_rates(self):
        for rate in (0.05, 0.3, 1.0):
            model = hmm_utils.get_read_matcher_model_enhanced(
                'TGC', 'GCA', ['A'], 2, None, False, maximum_error_rate=rate)
            self.assertEqual(float('-inf'), model.dp_score_threshold)
            self.assertEqual(3, model.read_length_used_to_build_model)
        model = hmm_utils.get_repeat_matcher_enhanced_hmm(
            [['ACG'], ['TGC']], 2, None, maximum_error_rate=0.05)
        self.assertEqual(0.5, _transition(model, model.start.name, 'unit_start_1'))
        self.assertEqual(0.5, _transition(model, model.start.name, 'unit_start_2'))

    def test_zero_error_rate_keeps_the_existing_model_completeness_refusal(self):
        # The enhanced backend refuses -inf emission entries, including declared
        # zero probabilities. This extraction must not silently change that ABI.
        with self.assertRaisesRegexp(ValueError, 'emitting state'):
            hmm_utils.get_repeat_matcher_enhanced_hmm([['ACG']], 2, None, maximum_error_rate=0.0)
        prefix = hmm_utils.get_prefix_matcher_hmm('ACG', maximum_error_rate=0.0)
        self.assertEqual(0.0, _transition(prefix, 'prefix_start_prefix', 'I0_prefix'))

    def test_unsupported_legacy_matrix_builders_remain_unsupported(self):
        for function, arguments in (
                (hmm_utils.get_variable_number_of_repeats_matcher_hmm, (['ACG'], 2, None)),
                (hmm_utils.get_read_matcher_model, ('TGC', 'GCA', ['ACG'], 2, None))):
            with self.assertRaisesRegexp(AttributeError, 'from_matrix'):
                function(*arguments)


if __name__ == '__main__':
    unittest.main()
