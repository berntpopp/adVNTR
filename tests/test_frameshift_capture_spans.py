"""Invented anonymous spans exercise exact identity-set reconstruction."""
import copy
import unittest
from collections import OrderedDict


def source_spans():
    return OrderedDict([
        (('1', 30, 0, True, True), [(1, 'read-example-b', 0), (0, 'read-example-a', 1)]),
        (('1', 14, 4, False, False), [(0, 'read-example-a', 'partial_start')]),
        (('prefix', 6, 0, False, False), [(0, 'read-example-a', 'prefix_flank')]),
        (('suffix', 12, 8, False, False), [(1, 'read-example-b', 'suffix_flank')]),
    ])


def document():
    return {
        'occurrences': [
            {'read_index': 0, 'kind': 'complete', 'ordinal': 1},
            {'read_index': 0, 'kind': 'prefix', 'ordinal': None},
            {'read_index': 0, 'kind': 'partial_start', 'ordinal': None},
            {'read_index': 1, 'kind': 'complete', 'ordinal': 0},
            {'read_index': 1, 'kind': 'suffix', 'ordinal': None},
        ],
        'spans': [
            {'pattern_index': '1', 'reached': '1e', 'inserted': '0', 'saw_start': True,
             'saw_end': True, 'occurrence_ids': [0, 3]},
            {'pattern_index': '1', 'reached': 'e', 'inserted': '4', 'saw_start': False,
             'saw_end': False, 'occurrence_ids': [2]},
            {'pattern_index': 'prefix', 'reached': '6', 'inserted': '0', 'saw_start': False,
             'saw_end': False, 'occurrence_ids': [1]},
            {'pattern_index': 'suffix', 'reached': 'c', 'inserted': '8', 'saw_start': False,
             'saw_end': False, 'occurrence_ids': [4]},
        ],
    }


class TestAnonymousSpanCapture(unittest.TestCase):
    def test_roundtrip_preserves_production_span_order_with_dense_anonymous_ids(self):
        from advntr.frameshift_capture_spans import encode_spans, decode_spans
        source = source_spans()
        result = encode_spans(source, {'1': 4}, 4, 2)
        self.assertEqual(document(), result)
        self.assertNotIn('read-example', repr(result))
        decoded = decode_spans(result, {'1': 4}, 4, 2)
        self.assertEqual(((0, 1), (0, 'prefix_flank'), (0, 'partial_start'),
                          (1, 0), (1, 'suffix_flank')), decoded.occurrences)
        self.assertEqual(tuple(source), decoded.signatures)
        self.assertEqual(((0, 3), (2,), (1,), (4,)), decoded.span_occurrence_ids)
        result['spans'][0]['occurrence_ids'].append(2)
        self.assertEqual((0, 3), decoded.span_occurrence_ids[0])

    def test_source_duplicate_observation_is_deduplicated_without_losing_ownership(self):
        from advntr.frameshift_capture_spans import encode_spans
        source = source_spans()
        first = next(iter(source))
        source[first].append(source[first][0])
        self.assertEqual(document(), encode_spans(source, {'1': 4}, 4, 2))
        source[first].append((1, 'conflicting-name', 0))
        with self.assertRaises(ValueError):
            encode_spans(source, {'1': 4}, 4, 2)

    def test_masks_above_json_safe_integer_range_remain_exact_hex(self):
        from advntr.frameshift_capture_spans import encode_spans, decode_spans
        value = 1 << 64
        source = OrderedDict([(('1', value, value - 1, True, False), [(0, 'synthetic', 0)])])
        encoded = encode_spans(source, {'1': 64}, 4, 1)
        self.assertEqual('10000000000000000', encoded['spans'][0]['reached'])
        self.assertEqual((('1', value, value - 1, True, False),),
                         decode_spans(encoded, {'1': 64}, 4, 1).signatures)

    def test_unknown_fields_and_noncanonical_masks_are_refused(self):
        from advntr.frameshift_capture_spans import decode_spans
        mutations = []
        for value in ('0x1e', '1E', '01e', '-1', '', 30, True, '20'):
            raw = document()
            raw['spans'][0]['reached'] = value
            mutations.append(raw)
        for part in ('root', 'span', 'occurrence'):
            raw = document()
            target = raw if part == 'root' else raw['spans' if part == 'span' else 'occurrences'][0]
            target['extra'] = None
            mutations.append(raw)
        for raw in mutations:
            with self.assertRaises(ValueError):
                decode_spans(raw, {'1': 4}, 4, 2)

    def test_occurrences_must_have_one_span_and_canonical_unique_ids(self):
        from advntr.frameshift_capture_spans import decode_spans
        for ids in ([0, 0, 3], [3, 0], [0, 99], [False, 3], [], [0]):
            raw = document()
            raw['spans'][0]['occurrence_ids'] = ids
            with self.assertRaises(ValueError):
                decode_spans(raw, {'1': 4}, 4, 2)
        raw = document()
        raw['spans'][1]['occurrence_ids'].append(0)
        with self.assertRaises(ValueError):
            decode_spans(raw, {'1': 4}, 4, 2)
        raw = document()
        raw['occurrences'].reverse()
        with self.assertRaises(ValueError):
            decode_spans(raw, {'1': 4}, 4, 2)

    def test_read_occurrence_domains_and_geometry_are_closed(self):
        from advntr.frameshift_capture_spans import decode_spans
        for field, value in (('read_index', -1), ('read_index', 2), ('read_index', True),
                              ('kind', 'other'), ('ordinal', -1), ('ordinal', True), ('ordinal', None)):
            raw = document()
            raw['occurrences'][0][field] = value
            with self.assertRaises(ValueError):
                decode_spans(raw, {'1': 4}, 4, 2)
        for field, value in (('pattern_index', None), ('pattern_index', '2'), ('saw_start', 1),
                              ('pattern_index', 'prefix')):
            raw = document()
            raw['spans'][0][field] = value
            with self.assertRaises(ValueError):
                decode_spans(raw, {'1': 4}, 4, 2)
        for geometry, length, count in (({'01': 4}, 4, 2), ({'1': 0}, 4, 2),
                                         ({'1': 4}, True, 2), ({'1': 4}, 4, False)):
            with self.assertRaises(ValueError):
                decode_spans(document(), geometry, length, count)

    def test_empty_inventory_is_valid_without_manufacturing_occurrences(self):
        from advntr.frameshift_capture_spans import encode_spans, decode_spans
        raw = encode_spans(OrderedDict(), {'1': 4}, 4, 0)
        self.assertEqual({'occurrences': [], 'spans': []}, raw)
        self.assertEqual(((), (), ()), decode_spans(raw, {'1': 4}, 4, 0))

    def test_malformed_shapes_geometry_and_duplicate_signatures_are_refused(self):
        from advntr.frameshift_capture_spans import decode_spans
        for geometry in ({}, [], {'1': False}):
            with self.assertRaises(ValueError):
                decode_spans(document(), geometry, 4, 2)
        for field in ('spans', 'occurrences'):
            raw = document()
            raw[field] = ()
            with self.assertRaises(ValueError):
                decode_spans(raw, {'1': 4}, 4, 2)
        raw = document()
        raw['occurrences'][1]['ordinal'] = 0
        with self.assertRaises(ValueError):
            decode_spans(raw, {'1': 4}, 4, 2)
        raw = document()
        raw['spans'][0]['reached'] = '1000'
        with self.assertRaises(ValueError):
            decode_spans(raw, {'1': 4}, 4, 2)
        raw = document()
        raw['spans'][1] = dict(raw['spans'][0], occurrence_ids=[2])
        with self.assertRaises(ValueError):
            decode_spans(raw, {'1': 4}, 4, 2)
        raw = document()
        raw['spans'][2]['pattern_index'] = '1'
        with self.assertRaises(ValueError):
            decode_spans(raw, {'1': 4}, 4, 2)

    def test_real_synthetic_counter_roundtrip_preserves_each_candidate_trial_set(self):
        from advntr.frameshift_capture_spans import encode_spans, decode_spans
        from advntr.frameshift_opportunities import _signature_supports, parse_components
        from advntr import vntr_finder
        from tests import test_exact_caller as reference
        fixture = reference._ExactCallerTestCase('setUp')
        original = vntr_finder.OpportunityCounter
        captured = []
        def counter(*args, **kwargs):
            result = original(*args, **kwargs)
            captured.append(result)
            return result
        fixture.setUp()
        vntr_finder.OpportunityCounter = counter
        try:
            reads = reference._all_three_sites()
            fixture._run(reads)
            source = captured[0]
            encoded = encode_spans(source._spans, dict(source._hmm_match_count), 12, len(reads))
            decoded = decode_spans(encoded, dict(source._hmm_match_count), 12, len(reads))
            for state, row in fixture.finder.last_frameshift_opportunities.items():
                components = parse_components(state)
                trials = set()
                for signature, ids in zip(decoded.signatures, decoded.span_occurrence_ids):
                    if components is not None and _signature_supports(signature, components):
                        trials.update(decoded.occurrences[index] for index in ids)
                self.assertEqual(row['opportunities'], len(trials), state)
                self.assertTrue(set(row['support_identities']).issubset(trials), state)
            self.assertGreater(len(decoded.occurrences), 0)
        finally:
            vntr_finder.OpportunityCounter = original
            fixture.tearDown()
