"""Frozen loaded reference inputs and independently derived unit geometry."""
import math
import numbers
from collections import namedtuple

from advntr.frameshift_capture_spans import _integer, _object
from advntr.repeat_order import get_reference_repeat_order


ModelLocus = namedtuple('ModelLocus', ('vntr_id chromosome start_point ref_end pattern repeat_segments '
                                      'left_flanking_region right_flanking_region scaled_score'))


def _string(value, nonempty=True):
    if not isinstance(value, basestring) or (nonempty and not value):
        raise ValueError('model reference sequence or label must be a string')


def decode_model(document):
    """Keep optional legacy end/score nulls without inventing replacement values."""
    _object(document, set(ModelLocus._fields), 'model locus')
    _integer(document['vntr_id'], 1, 'VNTR identifier')
    _integer(document['start_point'], 0, 'reference start')
    if document['ref_end'] is not None:
        _integer(document['ref_end'], document['start_point'] + 1, 'reference end')
    for name in ('chromosome', 'pattern', 'left_flanking_region', 'right_flanking_region'):
        _string(document[name], nonempty=name in ('chromosome', 'pattern'))
    segments = document['repeat_segments']
    if not isinstance(segments, list) or not segments:
        raise ValueError('model repeat segments must be a nonempty list')
    for segment in segments:
        _string(segment)
    score = document['scaled_score']
    if score is not None and (isinstance(score, bool) or not isinstance(score, numbers.Real)
                              or math.isnan(score) or math.isinf(score)):
        raise ValueError('model recruitment score must be finite or null')
    fields = dict(document, repeat_segments=tuple(segments))
    return ModelLocus(**fields)


def model_document(reference):
    """Project the actual loaded model fields used for recruitment and HMM geometry."""
    document = dict((name, getattr(reference, name)) for name in ModelLocus._fields if name != 'vntr_id')
    document['vntr_id'] = reference.id
    document['repeat_segments'] = list(reference.get_repeat_segments())
    decode_model(document)
    return document


def geometry_document(model, ru_bp_coverage):
    """Retain every model unit, including zero-coverage and zero-candidate units."""
    units = sorted(set(model.repeat_segments))
    return [{'pattern_index': str(index + 1), 'sequence': sequence,
             'reference_copies': model.repeat_segments.count(sequence), 'ru_bp_coverage': ru_bp_coverage.get(str(index + 1), 0)}
            for index, sequence in enumerate(units)]


def decode_geometry(document, model):
    """Check the complete unit inventory against loaded-model sequence and copies."""
    units = sorted(set(model.repeat_segments))
    if not isinstance(document, list) or len(document) != len(units):
        raise ValueError('unit geometry must contain the complete model unit inventory')
    geometry = {}
    for index, (row, sequence) in enumerate(zip(document, units)):
        _object(row, set(('pattern_index', 'sequence', 'reference_copies', 'ru_bp_coverage')), 'unit geometry')
        _integer(row['reference_copies'], 1, 'reference copy count')
        _integer(row['ru_bp_coverage'], 0, 'unit base coverage')
        if (row['pattern_index'] != str(index + 1) or row['sequence'] != sequence
                or row['reference_copies'] != model.repeat_segments.count(sequence)):
            raise ValueError('unit geometry differs from the frozen reference model')
        geometry[row['pattern_index']] = dict((name, row[name]) for name in row if name != 'pattern_index')
    return geometry, get_reference_repeat_order(model.repeat_segments, units)
