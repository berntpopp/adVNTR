"""The frozen background model `p0`, supplied as an external artifact and never as code.

PLAN Task 8 Step 3's mechanism, and Step 7's constraint. The exact caller
(`advntr/exact_caller.py`) needs a null probability per candidate. That number is
calibrated outside this repository, on a partition that is not the holdout, and it
enters a run only through a path the operator passes at the command line. **No
cohort-derived constant, path, sample id or finding may enter Git in any form** (PLAN
Global Constraints, the data rule), so this module ships a reader and a validator and no
probability whatsoever.

The temptation this deliberately refuses: SPEC Q-RATE (SPEC line 44) reports 3.0e-4
pooled and 1.7e-4 median indel rates over public candidates, but those are conditional
summaries of candidates already selected at support >= 3, so selection truncation and
state heterogeneity make them **not** plug-in estimates for a production null. (The
1e-3 nearby in the SPEC is `INDEL_MUTATION_MIN_PVALUE`, the calling cutoff at SPEC line
107 -- a different quantity, and not a rate at all.) Planting one as
a default would produce a caller that looks calibrated and is not, which is worse than a
caller that refuses to run. With no artifact supplied the exact caller does not run.

**Schema** -- versioned and self-describing, so a run's log can say exactly what it
scored against. Every number in this example is SYNTHETIC and deliberately absurd for an
indel background; it illustrates the shape, nothing else:

```json
{
  "schema": "advntr.frameshift.background",
  "version": 1,
  "provenance": "SYNTHETIC EXAMPLE -- made-up numbers, calibrated on nothing",
  "default_probability": 0.25,
  "states": {"D3_1": 0.125, "I2_1_T_LEN1": 0.0625}
}
```

- `schema` and `version` are both required and both checked; an artifact from another
  tool, or from a future format, is refused rather than half-read.
- `provenance` is required, free text, and must be non-empty. It is how the operator
  records what the model was calibrated on. It must not name cohort samples -- a
  description of the partition and the date is the intended content -- and this module
  neither parses it nor puts it anywhere except the run log.
- `default_probability` is required: every candidate must get a `p0`, and the artifact,
  not this tree, decides what an unlisted state gets.
- `states` is optional. A single-rate background is a legitimate artifact; the per-state
  table is the refinement -- which is why an unknown top-level field is refused rather
  than ignored: a misspelling would silently downgrade an artifact to that fallback. Keys are the shipped `State` strings exactly as the six-column
  table prints them (SPEC 3.5 keeps those byte-identical, so they are a stable key).

**Validation.** Every probability -- the default and each state -- must be a real number
strictly inside `(0, 1)` and not NaN. The open interval is not fussiness: `p0 = 0` makes
any support infinitely surprising and `p0 = 1` makes none of it surprising, and both are
almost certainly a mis-parsed field rather than an intended model. Refusals name the file
and the problem, because the operator's next action is to fix that file.
"""
import json
import math
import numbers
import os


SCHEMA = 'advntr.frameshift.background'

#: Everything a version 1 artifact may contain. Anything else is refused rather than
#: ignored: a misspelled `"state"` for `"states"` would otherwise load as a perfectly
#: valid single-rate model and score every candidate against the default, which is the
#: silent mis-scoring this module exists to prevent.
KNOWN_FIELDS = ('schema', 'version', 'provenance', 'default_probability', 'states')

#: Only version 1 exists. A reader that silently accepted an unknown version would score
#: against fields it does not understand.
SUPPORTED_VERSIONS = (1,)


class BackgroundModelError(ValueError):
    """A background artifact is absent, unreadable, or does not validate."""


class BackgroundModel(object):
    """A validated frozen background. Immutable by convention; built only by the loader."""

    def __init__(self, version, provenance, default_probability, states, path):
        self.version = version
        self.provenance = provenance
        self.default_probability = default_probability
        self.states = states
        self.path = path

    def probability_for(self, state):
        """`p0` for one emitted `State` string, falling back to the declared default."""
        return self.states.get(state, self.default_probability)

    def describe(self):
        """One line for the run log: what was loaded, and what it says it is."""
        return ('frameshift background v%s from %s (%d state-specific rates): %s'
                % (self.version, self.path, len(self.states), self.provenance))


def _refuse(path, problem):
    raise BackgroundModelError('background model %s: %s' % (path, problem))


def _validated_probability(path, field, value):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        _refuse(path, '%s is not a number (%r)' % (field, value))
    if math.isnan(value):
        _refuse(path, '%s is NaN' % field)
    if not 0.0 < value < 1.0:
        _refuse(path, '%s is %r, which is outside the open interval (0, 1)'
                % (field, float(value)))
    return float(value)


def _document(path):
    if not os.path.isfile(path):
        _refuse(path, 'file not found')
    try:
        with open(path) as handle:
            document = json.load(handle)
    except ValueError as error:
        _refuse(path, 'file is not valid JSON (%s)' % error)
    except IOError as error:
        _refuse(path, 'file could not be read (%s)' % error)
    if not isinstance(document, dict):
        _refuse(path, 'top level must be a JSON object, got %s'
                % type(document).__name__)
    return document


def load_background_model(path):
    """Read, validate and return the frozen background at `path`.

    Raises `BackgroundModelError` naming the file and the problem for every rejection.
    """
    document = _document(path)
    if document.get('schema') != SCHEMA:
        _refuse(path, 'schema is %r, expected %r' % (document.get('schema'), SCHEMA))
    if 'version' not in document:
        _refuse(path, 'no version field')
    # The type first: `True in (1,)` is True in Python 2, so a boolean version would
    # otherwise pass the membership test and then log as `vTrue`.
    if isinstance(document['version'], bool) or not isinstance(document['version'], int):
        _refuse(path, 'version must be an integer, got %r' % (document['version'],))
    if document['version'] not in SUPPORTED_VERSIONS:
        _refuse(path, 'declares version %r; supported versions are %r'
                % (document['version'], list(SUPPORTED_VERSIONS)))
    unknown = sorted(set(document) - set(KNOWN_FIELDS))
    if unknown:
        _refuse(path, 'unknown field(s) %s; known fields are %s'
                % (', '.join(repr(field) for field in unknown),
                   ', '.join(KNOWN_FIELDS)))
    provenance = document.get('provenance')
    if not isinstance(provenance, basestring) or not provenance.strip():  # noqa: F821
        _refuse(path, 'no provenance: the artifact must record what it was calibrated '
                      'on, in free text that names no sample')
    if 'default_probability' not in document:
        _refuse(path, 'no default_probability: every candidate must get a p0, and the '
                      'artifact decides what an unlisted state gets')
    default = _validated_probability(path, 'default_probability',
                                     document['default_probability'])
    raw_states = document.get('states', {})
    if not isinstance(raw_states, dict):
        _refuse(path, 'states must be a JSON object, got %s' % type(raw_states).__name__)
    states = {}
    for state in sorted(raw_states):
        states[state] = _validated_probability(path, 'probability for state %s' % state,
                                               raw_states[state])
    return BackgroundModel(document['version'], provenance.strip(), default, states, path)
