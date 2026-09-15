"""Adapt verified complete evidence to the unchanged recipe-v1 estimator.

The adapter never repairs a record or pools different opportunity-generating
policies. Diagnostic replay shares production traversal, including candidates
which the original supporting-read cutoff prevented from reaching a statistic.
"""
import hashlib
import json
import os
import stat

from advntr.background_capture import Capture, validate_sink_document
from advntr.background_estimator import FitterError, SINK_SCHEMA, SINK_VERSION
from advntr.capabilities import canonical_bytes
from advntr.frameshift_capture_record import SCHEMA, decode_capture
from advntr.frameshift_replay_policy import ReplayPolicy, decode_caller_policy
from advntr.frameshift_replay_decisions import evaluate_visits
from advntr.frameshift_visit_decisions import flank_boundaries
from advntr.frameshift_visit_document import visit_document
from advntr.repeat_order import get_reference_repeat_order


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def _constant(value):
    raise ValueError('nonfinite JSON value: %s' % value)


def load_fit_capture(sample_id, path):
    """Read a closed v2 sink, retaining the supported legacy v1 loader.

    The v2 estimator is single-locus: state strings contain no locus identity,
    so merging several loci would silently conflate their rates. A controller
    must separately bind a successful process and the expected target roster.
    """
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise FitterError('calibration sink must be a regular file')
        with os.fdopen(os.dup(descriptor), 'rb') as handle:
            raw = handle.read()
            handle.seek(0)
            if handle.read() != raw:
                raise FitterError('calibration sink changed while being read')
        after = os.fstat(descriptor)
        identity = lambda entry: (entry.st_dev, entry.st_ino, entry.st_size, entry.st_mtime, entry.st_ctime)
        if identity(before) != identity(after) or identity(os.lstat(path)) != identity(after):
            raise FitterError('calibration sink changed while being read')
    finally:
        os.close(descriptor)
    try:
        documents = [json.loads(line, object_pairs_hook=_pairs, parse_constant=_constant)
                     for line in raw.splitlines()]
        if not documents or any(not isinstance(item, dict) for item in documents):
            raise ValueError('capture must contain JSON objects')
        versions = set(item.get('schema_version') for item in documents)
        if versions == set((None,)):
            for number, document in enumerate(documents, 1):
                validate_sink_document(document, path, number)
            return Capture(sample_id, path, documents)
        if versions != set((SCHEMA,)) or len(documents) != 1 or not raw.endswith('\n'):
            raise ValueError('v2 fitting requires exactly one complete newline-terminated locus')
        document = documents[0]
        decoded = decode_capture(document)
        if decoded.ineligible_states:
            raise ValueError('capture attribution is outside its own eligible trials')
    except (ValueError, TypeError, KeyError) as error:
        raise FitterError('invalid completed calibration capture: %s' % error)
    projected = dict(decoded.locus, schema=SINK_SCHEMA, version=SINK_VERSION,
                     spans=[list(signature) + [len(ids)] for signature, ids in
                            zip(decoded.inventory.signatures, decoded.inventory.span_occurrence_ids)],
                     candidates=list(decoded.records.values()))
    capture = Capture(sample_id, path, [projected])
    capture.completed = decoded
    identity = dict((key, document[key]) for key in ('producer', 'assets', 'capture_policy', 'caller_policy'))
    identity['locus'] = dict((key, decoded.locus[key]) for key in ('vntr_id', 'read_length', 'is_haploid'))
    capture.fit_identity = identity
    capture.capture_sha256 = hashlib.sha256(raw).hexdigest()
    capture.completed_decisions = decisions_from_visits(decoded.visits, decoded)
    return capture


def require_fit_identity(first, other):
    """Refuse mixed evidence versions or policies instead of averaging their nulls."""
    left, right = getattr(first, 'fit_identity', None), getattr(other, 'fit_identity', None)
    if canonical_bytes(left) != canonical_bytes(right):
        raise FitterError('capture identity differs; recapture and background refit are required')


def decisions_from_visits(visits, capture):
    """Expose the existing diagnostic interface from verified ordered receipts."""
    logged, skipped, tested, called = {}, {}, {}, []
    for visit in visits:
        if visit.plan.disposition == 'outside-boundary':
            continue
        state, count = visit.plan.state, visit.plan.read_support
        logged[state] = count
        if visit.plan.disposition == 'insufficient-read-support':
            skipped[state] = count
        else:
            tested[state] = count
        if visit.statistic is not None and visit.statistic.called:
            called.append(state)
    return {'logged': logged, 'skipped': skipped, 'tested': tested, 'called': called,
            'read_length': capture.locus['read_length'],
            'repeat_units': dict((index, row['sequence']) for index, row in capture.geometry.items())}


def replay_fit_capture(capture, model, caller_document):
    """Use the native full traversal for every fitted-model diagnostic policy."""
    original = capture.completed
    mode, frameshift = decode_caller_policy(caller_document)
    policy = ReplayPolicy(original.policy.capture._replace(caller_mode=mode), frameshift)
    boundaries = flank_boundaries(original.model.repeat_segments, original.model.left_flanking_region,
                                  original.model.right_flanking_region, original.locus['read_length'])
    order = get_reference_repeat_order(original.model.repeat_segments, sorted(set(original.model.repeat_segments)))
    visits = evaluate_visits(original.traversal, order, boundaries, original.geometry,
                            original.records, policy, model if mode == 'exact' else None)
    decisions = decisions_from_visits(visits, original)
    return {'sample_id': capture.sample_id, 'called': bool(decisions['called']),
            'called_states': decisions['called'], 'tested': len(decisions['tested']),
            'k_exceeds_n': sum(visit.disposition == 'support-exceeds-opportunities' for visit in visits),
            'missing_rows': sum(visit.disposition == 'missing-opportunity-row' for visit in visits),
            'details': [visit_document(visit) for visit in visits]}
