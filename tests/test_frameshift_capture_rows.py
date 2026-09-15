"""Invented row evidence tests union/intersection audits independently of counts."""
import copy
import unittest
from collections import OrderedDict

from tests.test_frameshift_capture_spans import document as span_document
from advntr.frameshift_capture_spans import decode_spans


GEOMETRY = {'1': {'sequence': 'ACGT', 'reference_copies': 1, 'ru_bp_coverage': 8}}


def rows():
    return OrderedDict([('D2_1', {
        'candidate': 'D2_1', 'support': 1, 'opportunities': 3,
        'support_identities': ((0, 1),), 'opportunity_spans': ((0, 2), (1, 1)),
        'legacy_support': 4, 'legacy_states': ['D2_1'], 'state_identities': {'D2_1': ((0, 1),)},
        'pattern_index': '1', 'ru_bp_coverage': 8, 'ru_length': 4,
        'ru_bp_coverage_ratio': 2, 'avg_bp_coverage': 1.0,
    })])


def inventory():
    return decode_spans(span_document(), {'1': 4}, 4, 2)


class TestCaptureEvidenceRows(unittest.TestCase):
    def test_dense_roundtrip_preserves_the_native_exact_evidence(self):
        from advntr.frameshift_capture_rows import encode_rows, decode_rows
        original = rows()
        encoded = encode_rows(original, inventory(), GEOMETRY, False)
        self.assertEqual([0], encoded[0]['support_occurrence_ids'])
        self.assertEqual({'D2_1': [0]}, encoded[0]['state_occurrence_ids'])
        self.assertNotIn('opportunity_spans', encoded[0])
        self.assertNotIn('legacy_states', encoded[0])
        decoded, ineligible = decode_rows(encoded, inventory(), GEOMETRY, False)
        self.assertEqual(original, decoded)
        self.assertEqual((), ineligible)
        encoded[0]['state_occurrence_ids']['D2_1'].append(2)
        self.assertEqual(((0, 1),), decoded['D2_1']['state_identities']['D2_1'])

    def test_union_attribution_outside_own_compound_trials_is_ineligible_not_normalized(self):
        from advntr.frameshift_capture_rows import encode_rows, decode_rows
        original = rows()
        compound = 'D2_1&D4_1'
        original['D2_1'].update(support=2, support_identities=((0, 1), (0, 'partial_start')),
                              state_identities={compound: ((0, 'partial_start'),)}, legacy_states=[compound])
        own = dict(original['D2_1'], candidate=compound, support=0, opportunities=2,
                   support_identities=(), opportunity_spans=((0, 2),), state_identities={}, legacy_states=[])
        original[compound] = own
        encoded = encode_rows(original, inventory(), GEOMETRY, False)
        decoded, ineligible = decode_rows(encoded, inventory(), GEOMETRY, False)
        from advntr.exact_caller import aggregate_evidence
        self.assertEqual((1, 2), aggregate_evidence(decoded, compound))
        self.assertEqual((compound,), ineligible)

    def test_counts_ownership_and_geometry_are_independently_recomputed(self):
        from advntr.frameshift_capture_rows import encode_rows, decode_rows
        good = encode_rows(rows(), inventory(), GEOMETRY, False)
        for field, value in (('support', 2), ('support', True), ('opportunities', 2),
                              ('legacy_support', -1), ('ru_length', 3), ('ru_bp_coverage', 9),
                              ('ru_bp_coverage_ratio', 3), ('avg_bp_coverage', float('nan')),
                              ('pattern_index', '2'), ('extra', None),
                              ('support_occurrence_ids', [1]), ('support_occurrence_ids', [0, 0]),
                              ('state_occurrence_ids', {'D2_1': [2]})):
            raw = copy.deepcopy(good)
            raw[0][field] = value
            with self.assertRaises(ValueError):
                decode_rows(raw, inventory(), GEOMETRY, False)

    def test_source_fields_cannot_be_silently_discarded_or_corrected(self):
        from advntr.frameshift_capture_rows import encode_rows
        for field, value in (('extra', None), ('support_identities', ((99, 0),)),
                              ('opportunity_spans', ()), ('legacy_states', ['wrong']), ('candidate', 'D3_1')):
            raw = rows()
            raw['D2_1'][field] = value
            with self.assertRaises(ValueError):
                encode_rows(raw, inventory(), GEOMETRY, False)

    def test_empty_and_flank_records_keep_missing_unit_diagnostics_explicit(self):
        from advntr.frameshift_capture_rows import encode_rows, decode_rows
        self.assertEqual((OrderedDict(), ()), decode_rows([], inventory(), GEOMETRY, False))
        row = dict(rows()['D2_1'], candidate='D1_prefix', support=0, opportunities=1,
                   support_identities=(), opportunity_spans=((2, 1),), legacy_support=0,
                   legacy_states=[], state_identities={}, pattern_index='prefix',
                   ru_bp_coverage=None, ru_length=None, ru_bp_coverage_ratio=None, avg_bp_coverage=None)
        encoded = encode_rows(OrderedDict([('D1_prefix', row)]), inventory(), GEOMETRY, False)
        self.assertIsNone(encoded[0]['avg_bp_coverage'])
        self.assertEqual(row, decode_rows(encoded, inventory(), GEOMETRY, False)[0]['D1_prefix'])

    def test_native_pair_order_survives_dense_id_order_for_both_partial_labels(self):
        from advntr.frameshift_capture_spans import encode_spans
        from advntr.frameshift_capture_rows import encode_rows, decode_rows
        source = OrderedDict([(('1', 30, 0, False, False),
                               [(0, 'invented', 'partial_start'), (0, 'invented', 'partial_end')])])
        spans = decode_spans(encode_spans(source, {'1': 4}, 4, 1), {'1': 4}, 4, 1)
        original = rows()
        original['D2_1'].update(support=2, opportunities=2,
                               support_identities=((0, 'partial_end'), (0, 'partial_start')),
                               state_identities={'D2_1': ((0, 'partial_end'), (0, 'partial_start'))},
                               opportunity_spans=((0, 2),))
        self.assertEqual(original, decode_rows(encode_rows(original, spans, GEOMETRY, False),
                                               spans, GEOMETRY, False)[0])

    def test_untyped_or_unordered_rows_and_attributions_are_refused(self):
        from advntr.frameshift_capture_rows import encode_rows, decode_rows
        good = encode_rows(rows(), inventory(), GEOMETRY, False)
        for field, value in (('support_occurrence_ids', (0,)), ('support_occurrence_ids', [99]),
                              ('candidate', ''), ('candidate', []), ('state_occurrence_ids', []),
                              ('state_occurrence_ids', {'': [0]})):
            raw = copy.deepcopy(good)
            raw[0][field] = value
            with self.assertRaises(ValueError):
                decode_rows(raw, inventory(), GEOMETRY, False)
        for raw, haploid in (((), False), (good, 1), (good + good, False)):
            with self.assertRaises(ValueError):
                decode_rows(raw, inventory(), GEOMETRY, haploid)

    def test_real_synthetic_counter_retains_every_source_field(self):
        from advntr.frameshift_capture_spans import encode_spans
        from advntr.frameshift_capture_rows import encode_rows, decode_rows
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
            geometry = dict((str(index + 1), {'sequence': units[0], 'reference_copies': len(units),
                                              'ru_bp_coverage': fixture.finder.last_frameshift_opportunities[
                                                  'D3_2']['ru_bp_coverage'] if index == 1 else (168 if index == 0 else 48)})
                            for index, units in enumerate(source._pattern_clusters))
            spans = decode_spans(encode_spans(source._spans, dict(source._hmm_match_count), 12, len(reads)),
                                  dict(source._hmm_match_count), 12, len(reads))
            original_rows = fixture.finder.last_frameshift_opportunities
            encoded = encode_rows(original_rows, spans, geometry, False)
            decoded, _ineligible = decode_rows(encoded, spans, geometry, False)
            self.assertEqual(original_rows, decoded)
        finally:
            vntr_finder.OpportunityCounter = original
            fixture.tearDown()

    def test_diagnostic_counts_refuse_integral_float_representations(self):
        from advntr.frameshift_capture_rows import encode_rows, decode_rows
        good = encode_rows(rows(), inventory(), GEOMETRY, False)
        for field in ('ru_bp_coverage', 'ru_length', 'ru_bp_coverage_ratio'):
            raw = copy.deepcopy(good)
            raw[0][field] = float(raw[0][field])
            with self.assertRaises(ValueError):
                decode_rows(raw, inventory(), GEOMETRY, False)
