"""Frozen reference-model geometry is independently reconstructable."""
import copy
import unittest


def model_document():
    return {'vntr_id': 17, 'chromosome': 'chr1', 'start_point': 100, 'ref_end': 112,
            'pattern': 'ACGT', 'repeat_segments': ['ACGT', 'CCGT', 'ACGT'],
            'left_flanking_region': 'TTAA', 'right_flanking_region': 'TTGG', 'scaled_score': -0.5}


class TestCaptureModel(unittest.TestCase):
    def test_loaded_reference_projects_without_paths_or_annotations(self):
        from advntr.frameshift_capture_model import model_document as project, decode_model
        from advntr.reference_vntr import ReferenceVNTR
        raw = model_document()
        reference = ReferenceVNTR(17, 'ACGT', 100, 'chr1', 'GENE', 'coding', scaled_score=-0.5)
        reference.init_from_xml(raw['repeat_segments'], 'TTAA', 'TTGG')
        reference.ref_end = 112
        self.assertEqual(raw, project(reference))
        frozen = decode_model(raw)
        raw['repeat_segments'].append('GGGG')
        self.assertEqual(('ACGT', 'CCGT', 'ACGT'), frozen.repeat_segments)

    def test_geometry_keeps_all_units_and_reference_order(self):
        from advntr.frameshift_capture_model import decode_model, geometry_document, decode_geometry
        model = decode_model(model_document())
        geometry = geometry_document(model, {'1': 0, '2': 16})
        self.assertEqual([{'pattern_index': '1', 'sequence': 'ACGT', 'reference_copies': 2, 'ru_bp_coverage': 0},
                          {'pattern_index': '2', 'sequence': 'CCGT', 'reference_copies': 1, 'ru_bp_coverage': 16}], geometry)
        decoded, order = decode_geometry(geometry, model)
        self.assertEqual(['L', '1', '2', '1', 'R'], order)
        self.assertEqual(0, decoded['1']['ru_bp_coverage'])
        for field, value in (('sequence', 'AAAA'), ('reference_copies', 1), ('ru_bp_coverage', 0.0)):
            raw = copy.deepcopy(geometry)
            raw[0][field] = value
            with self.assertRaises(ValueError):
                decode_geometry(raw, model)
        with self.assertRaises(ValueError):
            decode_geometry(geometry[:1], model)

    def test_model_fields_are_closed_and_finite_with_explicit_nullable_end_and_score(self):
        from advntr.frameshift_capture_model import decode_model
        for field, value in (('vntr_id', True), ('start_point', -1), ('ref_end', 100),
                              ('ref_end', 112.0), ('scaled_score', float('nan')), ('scaled_score', True),
                              ('repeat_segments', []), ('repeat_segments', ['']), ('pattern', ''),
                              ('chromosome', None), ('extra', None)):
            raw = model_document()
            raw[field] = value
            with self.assertRaises(ValueError):
                decode_model(raw)
        raw = model_document()
        raw.update(ref_end=None, scaled_score=None, left_flanking_region='', right_flanking_region='')
        self.assertIsNone(decode_model(raw).scaled_score)
