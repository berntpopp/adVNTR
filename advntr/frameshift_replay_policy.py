"""Closed replay policies and the conservative recapture/refit boundary."""
from collections import namedtuple

from advntr.capture_policy import policy_from_document, validate_capture_policy
from advntr.frameshift_capture_spans import _object
from advntr.frameshift_decisions import resolve_policy, validate_policy


ReplayPolicy = namedtuple('ReplayPolicy', 'capture frameshift')


def decode_caller_policy(document):
    """Decode only the native statistic and global supporting-read policy."""
    _object(document, set(('schema_version', 'mode', 'cutoff', 'minimum_read_support')), 'caller policy')
    if document['schema_version'] != 'advntr-frameshift-policy-v1' or document['mode'] not in ('legacy', 'exact'):
        raise ValueError('unsupported frameshift policy schema or caller mode')
    if document['cutoff'] is None or document['minimum_read_support'] is None:
        raise ValueError('caller cutoff and minimum read support must be explicit nonnull values')
    return document['mode'], resolve_policy(document['cutoff'], document['minimum_read_support'])


def caller_policy_document(capture, frameshift):
    """Project explicit values; do not turn capture settings into caller defaults."""
    validate_capture_policy(capture)
    validate_policy(frameshift)
    return {'schema_version': 'advntr-frameshift-policy-v1', 'mode': capture.caller_mode,
            'cutoff': frameshift.cutoff, 'minimum_read_support': frameshift.minimum_read_support}


def decode_replay_policy(document):
    """Validate a complete policy, retaining the upstream raw capture domains."""
    _object(document, set(('schema_version', 'capture_policy', 'caller_policy')), 'replay policy')
    if document['schema_version'] != 'advntr-frameshift-replay-policy-v1':
        raise ValueError('unsupported frameshift replay policy schema')
    capture = policy_from_document(document['capture_policy'])
    mode, frameshift = decode_caller_policy(document['caller_policy'])
    if mode != capture.caller_mode:
        raise ValueError('caller mode differs from the capture policy')
    if not capture.frameshift_mode or capture.platform != 'illumina':
        raise ValueError('replay requires supported short-read frameshift capture')
    return ReplayPolicy(capture, frameshift)


def require_replay_compatible(original, requested):
    """Only mode, cutoff and minimum read support have a replay sufficiency proof.

    Every other raw capture parameter, even threads or an inactive ratio, requires
    recapture and refitting the background. This conservative registry can widen
    only with a new production/replay proof, never by ignoring an unhandled field.
    """
    validate_capture_policy(original)
    validate_capture_policy(requested)
    changed = sorted(name for name in original._fields
                     if name != 'caller_mode' and getattr(original, name) != getattr(requested, name))
    if changed:
        raise ValueError('recapture and background refit required for: %s' % ', '.join(changed))
