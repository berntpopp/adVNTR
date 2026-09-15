"""Freeze estimator recipe independently from run-local diagnostic operating points."""
from advntr.background_estimator import FitterError
from advntr.capabilities import read_document
from advntr.frameshift_replay_policy import decode_caller_policy


RECIPE_ID = 'recipe-v1'
DEFAULT_DIAGNOSTIC = {'schema_version': 'advntr-frameshift-policy-v1', 'mode': 'exact',
                      'cutoff': 0.001, 'minimum_read_support': 3}


def resolve_diagnostic_policy(args):
    """A different diagnostic cutoff/support never changes recipe-v1 rate fitting."""
    if getattr(args, 'background_recipe', RECIPE_ID) != RECIPE_ID:
        raise FitterError('unsupported background recipe; a changed estimator requires a new recipe')
    path = getattr(args, 'diagnostic_policy', None)
    try:
        document = dict(DEFAULT_DIAGNOSTIC) if path is None else read_document(path)
        mode, policy = decode_caller_policy(document)
    except (ValueError, TypeError, IOError, OSError) as error:
        raise FitterError('invalid background diagnostic policy: %s' % error)
    return {'schema_version': 'advntr-frameshift-policy-v1', 'mode': mode,
            'cutoff': policy.cutoff, 'minimum_read_support': policy.minimum_read_support}


def require_diagnostic_capture(capture, document):
    """Legacy captures lack candidates hidden by the original support traversal."""
    mode, policy = decode_caller_policy(document)
    if not hasattr(capture, 'completed') and (mode != 'exact' or policy.minimum_read_support != 3):
        raise FitterError('diagnostic mode/support changes require complete v2 captures')
