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

**Key validation (Task 8i).** `probability_for` (`:probability_for` below) is
`self.states.get(state, default)` -- a byte-exact dict lookup with no normalisation. A
key that is not byte-identical to an emitted `State` string therefore never matches
anything and silently scores that state against `default_probability`: no exception, no
log line, nothing an operator would notice short of re-deriving the whole table by hand.
That was tolerable while no artifact existed; it stops being tolerable once the
calibration freezes an artifact with roughly 21,000 keys, where one systematic key
defect would be silent across the entire table and look, from the outside, like a
background that simply did not help. `_validated_state_keys` therefore refuses (rather
than warns on) a key that is not a string, that differs from its own `.strip()`, that is
empty, or that collides with another key once stripped -- see that function's docstring
for why each is a separate rule and for the public-corpus evidence behind also refusing
a key the shipped grammar cannot produce.
"""
import json
import math
import numbers
import os

from advntr.frameshift_opportunities import parse_components


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


def _validated_state_keys(path, raw_states):
    """Refuse any `states` key that a byte-exact lookup can never match.

    Task 8i. `BackgroundModel.probability_for` (`:102-104`) does `self.states.get(state,
    default)` with no normalisation, so this function's job is to make every surviving
    key a plausible byte-exact `State` before it ever reaches that dict. Five rules, all
    refusing rather than warning, because a silent partial load is exactly the failure
    this function exists to close -- and this docstring, not scattered comments, is
    where the rationale for each lives, per this repository's evidence-citing style.

    1. **Not a string.** A JSON object can only ever have string keys -- `json.load`
       raises before `load_background_model` calls this function if the document text
       tries to write anything else -- so this branch is unreachable through the public
       `load_background_model(path)` path. It stays in because `raw_states` is handed
       to this function as a plain `dict` with no guarantee its caller came through
       JSON at all; `tests/test_frameshift_background.py` demonstrates it by calling
       this function directly with a non-string key, the same way this module's
       existing `test_no_probability_literal_lives_in_the_module_code` reaches past the
       public API to inspect something JSON alone cannot exercise.
    2. **Collides with another key once stripped.** Checked before rule 3 on purpose.
       JSON itself does not catch this: two keys that differ at the raw-text level (say
       `"D3_1"` and `"D3_1 "`) both survive `json.load` as separate dict entries -- only
       a literally repeated key string collapses, and that collapse happens silently
       inside `json.load`, before this function or anything else in this module ever
       sees the dict (out of scope here; it would need `object_pairs_hook` at
       `_document`'s `json.load` call). An artifact carrying both is unambiguously
       broken -- the calibration meant one state, not two rates for it -- and running
       this check first means the refusal names both keys and what they collide to,
       rather than reporting only the dirtier of the pair as if the clean one were not
       implicated. Every colliding group in the document is collected and named in ONE
       refusal, not just the first group found: a 21,000-key artifact should cost one
       edit-and-rerun cycle for its whole set of collisions, not one per group.
    3. **Empty.** `''.strip() == ''`, so rule 4 below cannot catch it, and an empty
       string is even less like a `State` than a whitespace-padded one.
    4. **Differs from its own `.strip()` -- checked for the whole key AND, separately,
       for each of its `&`-joined components.** The defect this task exists to close.
       Fix round 1 shipped only the whole-key half of this rule, which is incomplete in
       exactly the class of bug it exists to close: `"D3_1 &D4_1"` carries no LEADING or
       TRAILING whitespace on the *whole* key, so it passed that check, and it passed
       rule 5's grammar check too -- `parse_components` never validates a component's
       pattern-index field's *contents*, only that one is present (rule 5's `fields[1]`
       below) -- so the whole key loaded silently and scored against
       `default_probability`: the identical failure this task exists to close, moved one
       character to the right of a `&`. No shipped compound `State` ever carries
       whitespace in any component either (rule 5's structural argument covers every
       component the same way it covers a lone key), and this is not an edge case for
       the calibration artifact: 2,642 of the 5,500 real states collected as rule 5's
       evidence (below) are compound `A&B&...` forms. The component check iterates
       `key.split('&')` in the key's own left-to-right order, not sorted -- a `&`-joined
       key's component order is already the caller's canonical order, so nothing here
       needs re-deriving it.
    5. **Not a form the shipped grammar can produce.** The load-bearing argument is
       structural closure, not corpus corroboration -- the run below corroborates it, it
       is not the whole basis for it. Every `I`/`D` HMM state name this fork can ever
       produce is built by `advntr/hmm_utils.py` as `'%s%s_%s' % (kind, index,
       hmm_name)` for `kind` in `('I', 'D')` (also `'M'`, addressed below) --
       `advntr/hmm_utils.py:368,373,376` (the prefix flank matcher) and `:435,440,443`
       (the suffix flank matcher), where `hmm_name` is the hardcoded literal `'prefix'`
       or `'suffix'`; `:642,646,649` (`get_repeat_matcher_enhanced_hmm`, the one
       `advntr/frameshift_opportunities.py`'s own docstring cites as what production
       decodes against), where `hmm_name` is `str(pattern_count)`, an incrementing
       integer counter -- so `index` is always an integer and `hmm_name` is always
       exactly `'prefix'`, `'suffix'`, or an integer, at every construction site, for
       any VNTR. `advntr/vntr_finder.py:336-337`
       (`if not current_state.startswith('I') and not current_state.startswith('D'):
       continue`) admits only `I`/`D` states into any candidate at all, so an `M` state
       -- the one kind `parse_components` itself refuses -- structurally never reaches a
       `states` key either; that branch has no legitimate case to refuse. Past that,
       `advntr/mutation_keys.py`'s `legacy_mutation_candidates` only ever APPENDS: one
       emitted base character (`get_emitted_basepair_from_visited_states`,
       `advntr/hmm_utils.py:125-132`, sliced straight out of the read's own
       already-uppercased sequence -- never whitespace, by construction of a DNA
       sequence string) and a literal `_LEN<int>` suffix, joining multiple components
       with `&`. So every key this fork can ever generate, for any cohort, matches
       `[ID]<int>_(<int>|prefix|suffix)(_<base>)?(_LEN<int>)?(&...)*` -- digits,
       underscores, `&`, and the constant substrings `prefix`/`suffix`/`LEN`, nothing
       else, ever -- and `parse_components` accepts that whole family unconditionally
       (`kind in ('I', 'D')`, `len(fields) >= 2`,
       `advntr/frameshift_opportunities.py:236-250`). The rule cannot refuse a
       legitimate key for any cohort, not only the public one.

       Corroboration: a run over the real caller (`advntr_harness.capture.build_finder`
       + `select_illumina_reads` + `find_frameshift_from_selected_reads`,
       `finder.last_frameshift_evidence.keys()`) across all eight public `example_*`
       BAMs produced 5,500 distinct candidate/State strings -- including both
       repeat-unit and prefix/suffix flank forms (`I0_prefix_LEN1`, and `D148_suffix`,
       the undecorated deletion flank form with no `_LEN` suffix) and 2,642 compound
       `A&B&...` forms -- and `parse_components` accepted every single one: 0
       rejections. This is not proof for all 21,000 future keys, only the strongest
       evidence available before that artifact exists; if a calibrated artifact is later
       refused here, that is new evidence to act on, not a reason to have skipped the
       check -- the structural argument above is what actually carries the "for any
       cohort" claim.

    Rules 2-5 iterate `sorted(raw_states)` (or, for rule 2, sorted collision groups) so
    which key gets named in a refusal is deterministic across runs. Rule 1 must run over
    every key, unsorted-safe, before any of the others -- `.strip()` on a non-string key
    raises `AttributeError`, not a validation error the operator can act on, so rules 2-5
    may not run until rule 1 has confirmed every key is a string.
    """
    for key in raw_states:
        if not isinstance(key, basestring):  # noqa: F821
            _refuse(path, 'a states key is not a string (%r)' % (key,))

    stripped_to_originals = {}
    for key in raw_states:
        stripped_to_originals.setdefault(key.strip(), []).append(key)
    collisions = [(stripped, sorted(originals))
                  for stripped, originals in sorted(stripped_to_originals.items())
                  if len(originals) > 1]
    if collisions:
        _refuse(path, 'states keys collide once stripped: %s -- the artifact names '
                      'the same state more than once'
                % '; '.join('%s all collide to %r'
                            % (', '.join(repr(key) for key in originals), stripped)
                            for stripped, originals in collisions))

    for key in sorted(raw_states):
        if key == '':
            _refuse(path, 'a states key is the empty string')
        if key != key.strip():
            _refuse(path, 'states key %r has leading or trailing whitespace, so it can '
                          'never byte-match an emitted State and would silently score '
                          'against default_probability instead' % (key,))
        for component in key.split('&'):
            if component != component.strip():
                _refuse(path, 'states key %r has whitespace around its %r component, '
                              'so it can never byte-match an emitted State and would '
                              'silently score against default_probability instead'
                        % (key, component))
        if parse_components(key) is None:
            _refuse(path, 'states key %r is not a form the shipped grammar can produce '
                          '(advntr/frameshift_opportunities.py:parse_components '
                          'rejected it)' % (key,))


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
    _validated_state_keys(path, raw_states)
    states = {}
    for state in sorted(raw_states):
        states[state] = _validated_probability(path, 'probability for state %s' % state,
                                               raw_states[state])
    return BackgroundModel(document['version'], provenance.strip(), default, states, path)
