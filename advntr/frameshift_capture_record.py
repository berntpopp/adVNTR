"""Closed completed-capture ABI with independently verified native baseline replay.

The record binds model bytes separately from the loaded model fields, preserves
all opportunity inventory and the original candidate traversal, and carries
finite decision receipts. A completion marker is a per-locus assertion; the
controller must still prove process success and its complete target roster.
"""
import hashlib
import re
from collections import namedtuple

from advntr.capabilities import canonical_bytes
from advntr.frameshift_background import BackgroundModel, _validated_probability, _validated_state_keys
from advntr.frameshift_capture_model import decode_model, decode_geometry
from advntr.frameshift_capture_rows import decode_rows
from advntr.frameshift_capture_spans import decode_spans, _integer, _object
from advntr.frameshift_replay_policy import decode_replay_policy
from advntr.frameshift_replay_decisions import evaluate_visits
from advntr.frameshift_traversal import CandidateTraversal
from advntr.frameshift_visit_decisions import flank_boundaries
from advntr.frameshift_visit_document import visit_document, decision_warnings


SCHEMA = 'advntr-frameshift-capture-v2'
_FIELDS = set(('schema_version', 'completion', 'producer', 'assets', 'model_locus', 'loaded_background',
               'capture_policy', 'caller_policy', 'locus', 'unit_geometry', 'reference_order', 'flank_boundaries',
               'warnings', 'occurrences', 'spans', 'evidence_rows', 'candidate_traversal', 'decision_visits'))
Capture = namedtuple('Capture', 'policy model geometry locus inventory records traversal visits background ineligible_states')


def digest(document):
    """Use the same explicit canonical-byte rule as installed asset identity."""
    return hashlib.sha256(canonical_bytes(document)).hexdigest()


def _digest(value, nullable=False):
    if nullable and value is None:
        return
    if not isinstance(value, basestring) or re.match(r'^[0-9a-f]{64}\Z', value) is None:
        raise ValueError('capture asset identity must be a lowercase SHA256 digest')


def _same(actual, expected, label):
    if canonical_bytes(actual) != canonical_bytes(expected):
        raise ValueError('capture %s differs from independently reconstructed content' % label)


def decode_background(document):
    """Decode semantic loaded probabilities using the native model validators."""
    if document is None:
        return None
    _object(document, set(('schema', 'version', 'default_probability', 'states')), 'loaded background')
    if (document['schema'] != 'advntr.frameshift.background' or type(document['version']) is not int
            or document['version'] != 1):
        raise ValueError('unsupported captured background schema/version')
    raw = document['states']
    if not isinstance(raw, dict):
        raise ValueError('captured background states must be an object')
    _validated_state_keys('captured semantic model', raw)
    default = _validated_probability('captured semantic model', 'default', document['default_probability'])
    states = dict((state, _validated_probability('captured semantic model', 'state', value))
                  for state, value in raw.items())
    return BackgroundModel(1, 'captured semantic model', default, states, 'captured-semantic-model')


def _bindings(document):
    producer, assets = document['producer'], document['assets']
    _object(producer, set(('package_version', 'build_id', 'source_revision')), 'producer')
    if not isinstance(producer['package_version'], basestring) or not producer['package_version']:
        raise ValueError('capture package version must be nonempty text')
    _digest(producer['build_id'])
    revision = producer['source_revision']
    if revision is not None and (not isinstance(revision, basestring)
                                 or re.match(r'^(?:[0-9a-f]{40}|[0-9a-f]{64})\Z', revision) is None):
        raise ValueError('capture source revision must be a verified hash or null')
    _object(assets, set(('model_sha256', 'model_locus_sha256', 'background_sha256', 'loaded_background_sha256')), 'assets')
    for name, value in assets.items():
        _digest(value, nullable=name in ('background_sha256', 'loaded_background_sha256'))
    if assets['model_locus_sha256'] != digest(document['model_locus']):
        raise ValueError('capture loaded model digest differs')
    background = decode_background(document['loaded_background'])
    if background is None:
        if assets['background_sha256'] is not None or assets['loaded_background_sha256'] is not None:
            raise ValueError('capture background bindings require loaded content')
    elif (assets['background_sha256'] is None
          or assets['loaded_background_sha256'] != digest(document['loaded_background'])):
        raise ValueError('capture loaded background digest differs')
    return background


def _traversal(document, records):
    _object(document, set(('repeat_candidates', 'flank_candidates')), 'candidate traversal')
    collections = []
    for name in ('repeat_candidates', 'flank_candidates'):
        values = document[name]
        if not isinstance(values, list) or any(not isinstance(pair, list) or len(pair) != 2 for pair in values):
            raise ValueError('candidate traversal must contain ordered state/count pairs')
        collections.append(tuple(tuple(pair) for pair in values))
    traversal = CandidateTraversal(*collections)
    legacy = dict(traversal.repeat_candidates)
    legacy.update(traversal.flank_candidates)
    if set(legacy) - set(records):
        raise ValueError('capture omitted a source candidate evidence row')
    for state, row in records.items():
        if row['legacy_support'] != legacy.get(state, 0):
            raise ValueError('row read support differs from the original candidate traversal')
    return traversal


def decode_capture(document):
    """Validate a complete record and reproduce every original decision receipt.

    Stronger attribution subset failures are returned as calibration ineligibility;
    they do not silently rewrite native cardinality-based genotype decisions.
    """
    _object(document, _FIELDS, 'completed capture')
    if document['schema_version'] != SCHEMA or document['completion'] != 'completed-vntr':
        raise ValueError('capture must declare a supported completed-vntr record')
    background = _bindings(document)
    policy = decode_replay_policy({'schema_version': 'advntr-frameshift-replay-policy-v1',
                                   'capture_policy': document['capture_policy'], 'caller_policy': document['caller_policy']})
    if (policy.capture.caller_mode == 'exact') != (background is not None):
        raise ValueError('captured background presence differs from the original caller mode')
    model = decode_model(document['model_locus'])
    locus = document['locus']
    _object(locus, set(('vntr_id', 'read_length', 'is_haploid', 'selected_read_count')), 'locus')
    _integer(locus['vntr_id'], 1, 'VNTR identifier')
    _integer(locus['read_length'], 1, 'read length')
    _integer(locus['selected_read_count'], 0, 'selected read count')
    if (locus['vntr_id'] != model.vntr_id or type(locus['is_haploid']) is not bool
            or locus['is_haploid'] != policy.capture.is_haploid):
        raise ValueError('capture locus identity/ploidy differs from model or policy')
    geometry, reference_order = decode_geometry(document['unit_geometry'], model)
    _same(document['reference_order'], reference_order, 'reference order')
    boundaries = flank_boundaries(model.repeat_segments, model.left_flanking_region,
                                   model.right_flanking_region, locus['read_length'])
    _same(document['flank_boundaries'], {'suffix_min_position': boundaries[0], 'prefix_max_position': boundaries[1]},
          'flank boundaries')
    inventory = decode_spans({'occurrences': document['occurrences'], 'spans': document['spans']},
                              dict((index, len(row['sequence'])) for index, row in geometry.items()),
                              locus['read_length'], locus['selected_read_count'])
    records, ineligible = decode_rows(document['evidence_rows'], inventory, geometry, locus['is_haploid'])
    traversal = _traversal(document['candidate_traversal'], records)
    try:
        visits = evaluate_visits(traversal, reference_order, boundaries, geometry, records, policy, background)
    except (KeyError, IndexError, ZeroDivisionError) as error:
        raise ValueError('captured source traversal cannot complete with its bound geometry: %s' % type(error).__name__)
    _same(document['decision_visits'], [visit_document(visit) for visit in visits], 'baseline decision receipts')
    _same(document['warnings'], decision_warnings(visits, ineligible), 'decision warnings')
    return Capture(policy, model, geometry, dict(locus), inventory, records, traversal, visits, background, ineligible)
