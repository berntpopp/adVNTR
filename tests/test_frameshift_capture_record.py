"""Complete invented records bind assets, geometry, evidence, traversal and decisions."""
import copy
import hashlib
import unittest

from advntr.capabilities import canonical_bytes
from advntr.frameshift_capture_model import decode_model, geometry_document
from advntr.frameshift_capture_rows import encode_rows
from advntr.frameshift_replay_decisions import evaluate_visits
from advntr.frameshift_replay_policy import decode_replay_policy
from advntr.frameshift_traversal import freeze_traversal
from advntr.frameshift_visit_document import visit_document
from tests.test_frameshift_capture_model import model_document
from tests.test_frameshift_capture_rows import rows, inventory, GEOMETRY
from tests.test_frameshift_capture_spans import document as span_document
from tests.test_frameshift_replay_policy import replay_policy


def capture_document():
    model = model_document()
    model.update(repeat_segments=['ACGT'], ref_end=104, left_flanking_region='TTTT', right_flanking_region='GGGG')
    policy = replay_policy()
    traversal = freeze_traversal({'D2_1': 4}, {})
    visits = evaluate_visits(traversal, ['L', '1', 'R'], (4, 0), GEOMETRY, rows(), decode_replay_policy(policy), None)
    return dict(span_document(), **{
        'schema_version': 'advntr-frameshift-capture-v2', 'completion': 'completed-vntr',
        'producer': {'package_version': '2.4.0', 'build_id': 'a' * 64, 'source_revision': None},
        'assets': {'model_sha256': 'b' * 64, 'model_locus_sha256': hashlib.sha256(canonical_bytes(model)).hexdigest(),
                   'background_sha256': None, 'loaded_background_sha256': None},
        'model_locus': model, 'loaded_background': None,
        'capture_policy': policy['capture_policy'], 'caller_policy': policy['caller_policy'],
        'locus': {'vntr_id': 17, 'read_length': 4, 'is_haploid': False, 'selected_read_count': 4},
        'unit_geometry': geometry_document(decode_model(model), {'1': 8}),
        'reference_order': ['L', '1', 'R'], 'flank_boundaries': {'suffix_min_position': 4, 'prefix_max_position': 0},
        'warnings': [], 'evidence_rows': encode_rows(rows(), inventory(), GEOMETRY, False),
        'candidate_traversal': {'repeat_candidates': [['D2_1', 4]], 'flank_candidates': []},
        'decision_visits': [visit_document(visit) for visit in visits]})


class TestCaptureRecord(unittest.TestCase):
    def test_complete_record_revalidates_native_baseline_calls(self):
        from advntr.frameshift_capture_record import decode_capture
        raw = capture_document()
        decoded = decode_capture(raw)
        self.assertEqual(17, decoded.model.vntr_id)
        self.assertEqual(['called'], [visit.disposition for visit in decoded.visits])
        self.assertEqual((), decoded.ineligible_states)
        self.assertEqual(4, decoded.traversal.repeat_candidates[0][1])
        raw['unit_geometry'][0]['ru_bp_coverage'] = 999
        self.assertEqual(8, decoded.geometry['1']['ru_bp_coverage'])

    def test_receipt_and_binding_tampering_cannot_pass_baseline_parity(self):
        from advntr.frameshift_capture_record import decode_capture
        changes = []
        for field, value in (('schema_version', 'old'), ('completion', 'started'), ('extra', None),
                              ('reference_order', ['L', '2', 'R']), ('warnings', ['ignored'])):
            raw = capture_document()
            raw[field] = value
            changes.append(raw)
        for field, value in (('vntr_id', 18), ('read_length', True), ('is_haploid', 0), ('selected_read_count', -1)):
            raw = capture_document()
            raw['locus'][field] = value
            changes.append(raw)
        raw = capture_document()
        raw['decision_visits'][0]['statistic']['called'] = False
        changes.append(raw)
        raw = capture_document()
        raw['assets']['model_locus_sha256'] = '0' * 64
        changes.append(raw)
        raw = capture_document()
        raw['flank_boundaries']['prefix_max_position'] = True
        changes.append(raw)
        raw = capture_document()
        raw['evidence_rows'] = []
        changes.append(raw)
        for raw in changes:
            with self.assertRaises(ValueError):
                decode_capture(raw)

    def test_exact_baseline_uses_its_bound_semantic_background(self):
        from advntr.frameshift_capture_record import decode_capture, decode_background
        raw = capture_document()
        background = {'schema': 'advntr.frameshift.background', 'version': 1,
                      'default_probability': 0.5, 'states': {}}
        raw['loaded_background'] = background
        raw['assets']['background_sha256'] = 'c' * 64
        raw['assets']['loaded_background_sha256'] = hashlib.sha256(canonical_bytes(background)).hexdigest()
        policy = replay_policy('exact')
        raw['capture_policy'], raw['caller_policy'] = policy['capture_policy'], policy['caller_policy']
        visits = evaluate_visits(freeze_traversal({'D2_1': 4}, {}), ['L', '1', 'R'], (4, 0), GEOMETRY,
                                  rows(), decode_replay_policy(policy), decode_background(background))
        raw['decision_visits'] = [visit_document(visit) for visit in visits]
        decoded = decode_capture(raw)
        self.assertEqual(0.875, decoded.visits[0].statistic.pvalue)
        self.assertFalse(decoded.visits[0].statistic.called)
        raw['loaded_background']['default_probability'] = 0.1
        with self.assertRaises(ValueError):
            decode_capture(raw)

    def test_closed_asset_background_and_traversal_failure_paths(self):
        from advntr.frameshift_capture_record import decode_capture, decode_background
        for changed in ({'schema': 'old'}, {'version': True}, {'states': []}):
            raw = dict({'schema': 'advntr.frameshift.background', 'version': 1,
                        'default_probability': 0.5, 'states': {}}, **changed)
            with self.assertRaises(ValueError):
                decode_background(raw)
        self.assertEqual(0.25, decode_background({'schema': 'advntr.frameshift.background', 'version': 1,
                         'default_probability': 0.5, 'states': {'D2_1': 0.25}}).probability_for('D2_1'))
        changes = []
        for section, field, value in (('producer', 'package_version', ''), ('producer', 'build_id', 'bad'),
                                      ('producer', 'source_revision', 'bad'), ('assets', 'background_sha256', 'a' * 64),
                                      ('candidate_traversal', 'repeat_candidates', ['D2_1'])):
            raw = capture_document()
            raw[section][field] = value
            changes.append(raw)
        raw = capture_document()
        raw['evidence_rows'][0]['legacy_support'] = 3
        changes.append(raw)
        raw = capture_document()
        raw['loaded_background'] = {'schema': 'advntr.frameshift.background', 'version': 1,
                                    'default_probability': 0.5, 'states': {}}
        raw['assets'].update(background_sha256='a' * 64, loaded_background_sha256=hashlib.sha256(
                            canonical_bytes(raw['loaded_background'])).hexdigest())
        changes.append(raw)
        for raw in changes:
            with self.assertRaises(ValueError):
                decode_capture(raw)

    def test_impossible_reached_unit_geometry_fails_as_a_capture_error(self):
        from advntr.frameshift_capture_record import decode_capture
        raw = capture_document()
        raw['candidate_traversal']['repeat_candidates'] = [['D1_9', 4]]
        row = raw['evidence_rows'][0]
        row.update(candidate='D1_9', support=0, opportunities=0, support_occurrence_ids=[], state_occurrence_ids={},
                   pattern_index='9', ru_bp_coverage=None, ru_length=None, ru_bp_coverage_ratio=None, avg_bp_coverage=None)
        with self.assertRaisesRegexp(ValueError, 'cannot complete'):
            decode_capture(raw)
