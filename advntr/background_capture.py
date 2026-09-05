"""Task 8c: Background capture representations and sink / log parsers.

Represents one sample's sink (Capture), its (k, N) evidence table, and parsers for
calibration sinks and decision logs.
"""
import json
import os
import re

from advntr.background_estimator import FitterError, SINK_SCHEMA, SINK_VERSION
from advntr.exact_caller import aggregate_evidence
from advntr.frameshift_opportunities import parse_components, _signature_supports

def read_sink(path):
    """Every line of one sink, parsed. Abort on the first line that does not parse.

    Contract 1 inherited from the sink's review (`advntr/frameshift_calibration.py`'s
    module docstring): a torn line is confined to itself by the writer, but skipping it
    silently drops that sample's denominators -- including the zero-support states it
    alone witnessed -- and biases the null by an amount nothing downstream can bound.
    """
    if not os.path.isfile(path):
        raise FitterError('sink %s: file not found' % path)
    documents = []
    with open(path) as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                raise FitterError(
                    'sink %s:%d: blank line. Every line of every sink must parse; '
                    'skipping one would drop a sample\'s denominators' % (path, number))
            try:
                document = json.loads(line)
            except ValueError as error:
                raise FitterError(
                    'sink %s:%d: line does not parse as JSON (%s). A run killed '
                    'mid-write leaves a torn line; the fit stops here rather than '
                    'silently dropping this sample\'s denominators'
                    % (path, number, error))
            if not isinstance(document, dict):
                raise FitterError('sink %s:%d: line is not a JSON object' % (path, number))
            if document.get('schema') != SINK_SCHEMA:
                raise FitterError('sink %s:%d: schema is %r, expected %r'
                                  % (path, number, document.get('schema'), SINK_SCHEMA))
            if document.get('version') != SINK_VERSION:
                raise FitterError('sink %s:%d: version is %r, expected %r'
                                  % (path, number, document.get('version'),
                                     SINK_VERSION))
            for field in ('vntr_id', 'read_length', 'is_haploid', 'spans', 'candidates'):
                if field not in document:
                    raise FitterError('sink %s:%d: no %r field' % (path, number, field))
            documents.append(document)
    if not documents:
        raise FitterError('sink %s: no lines at all' % path)
    return documents


def _identities(raw):
    """JSON gives identities as lists; the shipped union puts them in a set."""
    return tuple(tuple(pair) for pair in raw)


class Capture(object):
    """One sample's sink: the span inventory, the rows, and the two aggregations.

    Sample identity comes from the sink's PATH and the controller's manifest, never from
    a line -- the line deliberately carries no sample id
    (`advntr/frameshift_calibration.py`, "A line carries no sample identifier").
    """

    def __init__(self, sample_id, path, documents):
        self.sample_id = sample_id
        self.path = path
        self.vntr_ids = []
        self.read_length = None
        self.is_haploid = None
        self.rows = {}
        self._spans_by_pattern = {}
        self._span_count = 0
        self._n_cache = {}
        self._k_cache = {}
        seen_vntr = set()
        for index, document in enumerate(documents):
            vntr_id = document['vntr_id']
            if vntr_id in seen_vntr:
                raise FitterError(
                    'sink %s: line %d repeats vntr_id %r. A resumed run can leave a '
                    'duplicate line, and consuming both would double that sample\'s '
                    'denominators; remove the duplicate offline first'
                    % (path, index + 1, vntr_id))
            seen_vntr.add(vntr_id)
            self.vntr_ids.append(vntr_id)
            if self.read_length is None:
                self.read_length = document['read_length']
                self.is_haploid = document['is_haploid']
            for entry in document['spans']:
                signature = tuple(entry[:5])
                count = entry[5]
                pattern = signature[0]
                self._spans_by_pattern.setdefault(pattern, []).append((signature, count))
                self._span_count += 1
            for raw in document['candidates']:
                row = dict(raw)
                row['state_identities'] = dict(
                    (state, _identities(pairs))
                    for state, pairs in raw['state_identities'].items())
                row['support_identities'] = _identities(raw['support_identities'])
                candidate = row['candidate']
                if candidate in self.rows:
                    raise FitterError('sink %s: two rows for candidate %r'
                                      % (path, candidate))
                self.rows[candidate] = row
        self._observed = set(self.rows)
        for row in self.rows.values():
            self._observed.update(row['state_identities'])
            self._observed.update(row['legacy_states'])

    # -- inventory ------------------------------------------------------------

    def span_count(self):
        return self._span_count

    def observed_states(self):
        """Every `State` that appears as a `candidate` or in `legacy_states`."""
        return set(self._observed)

    def patterns(self):
        return set(self._spans_by_pattern)

    # -- the two aggregations -------------------------------------------------

    def support_for(self, state):
        """`k(s)`: the union over ALL rows of the identities attributed to `s`.

        Ruling 1. Never a sum of supports: an identity belongs to the `State` that its
        own read's whole-read fusion produced, and summing sibling rows' supports credits
        a state with occurrences that produced something else
        (`advntr/exact_caller.py`'s module docstring, measured at 347 against 2).
        """
        if state not in self._observed:
            # No row of this sample named the state, so no identity was attributed to
            # it here: k = 0 by definition. Returned without caching, because the
            # enumerated state space is ~21,000 wide and ~20,000 of those are zero in
            # any one sample -- caching them would cost more memory than the sink.
            return 0
        if state in self._k_cache:
            return self._k_cache[state]
        identities = set()
        for row in self.rows.values():
            identities.update(row['state_identities'].get(state, ()))
        self._k_cache[state] = len(identities)
        return self._k_cache[state]

    def opportunities_for(self, state):
        """`N(s)`: the state's OWN row's `opportunities`, else recomputed from `spans`.

        The recompute is the ESTIMATION path -- a control that never produced this state
        has no row and still contributes a denominator, which is the whole point of the
        untruncated inventory. It is not the caller's path; see ruling 6 and
        `replay_sample`.
        """
        own = self.rows.get(state)
        if own is not None:
            return own['opportunities']
        return self.opportunities_for_components(parse_components(state))

    def opportunities_for_components(self, components):
        """`sum(count)` over spans whose signature satisfies EVERY component.

        Memoised on the component tuple, which is what makes the untruncated inventory
        affordable: `I3_2_A_LEN1` and `I3_2_T_LEN8` parse to the same single component,
        so the ~21,000 enumerated states collapse to a few thousand distinct denominators
        per sample. The test itself is always the shipped `_signature_supports`.
        """
        if components is None:
            return 0
        key = tuple(components)
        cached = self._n_cache.get(key)
        if cached is not None:
            return cached
        pattern = components[0][2]
        total = 0
        for signature, count in self._spans_by_pattern.get(pattern, ()):
            if _signature_supports(signature, components):
                total += count
        self._n_cache[key] = total
        return total

    def evidence(self, state):
        """`(k, N)` for the estimation population."""
        return self.support_for(state), self.opportunities_for(state)

    # -- the assertions -------------------------------------------------------

    def round_trip_failures(self):
        """Brief assertion 1: recompute every row's `opportunities` from `spans`.

        A mismatch means the sink's span table and its rows disagree, and the fit must
        stop -- the run's own arithmetic is the only thing that says the offline `N` is
        the same quantity the caller would compute.
        """
        failures = []
        for candidate in sorted(self.rows):
            row = self.rows[candidate]
            components = parse_components(candidate)
            if components is None:
                failures.append({'candidate': candidate, 'stored': row['opportunities'],
                                 'recomputed': None,
                                 'problem': 'parse_components refused the candidate'})
                continue
            total = 0
            for signatures in self._spans_by_pattern.values():
                for signature, count in signatures:
                    if _signature_supports(signature, components):
                        total += count
            if total != row['opportunities']:
                failures.append({'candidate': candidate, 'stored': row['opportunities'],
                                 'recomputed': total, 'problem': 'recompute disagrees'})
        return failures

    def shipped_aggregation_disagreements(self):
        """Compare this module's `(k, N)` against the shipped `aggregate_evidence`."""
        disagreements = []
        for candidate in sorted(self.rows):
            shipped = aggregate_evidence(self.rows, candidate)
            mine = (self.support_for(candidate), self.opportunities_for(candidate))
            if shipped != mine:
                disagreements.append({'candidate': candidate, 'shipped': shipped,
                                      'fitter': mine})
        return disagreements

    def cardinality_violations(self, states):
        """States where `k > N`: refused by the caller's guard, so a sensitivity cost.

        This is the ONLY half of the subset property an offline consumer can check. The
        sink exports a COUNT per span signature, not the identities behind it, so set
        membership is unanswerable here and is not claimed.
        """
        violations = []
        for state in states:
            support, opportunities = self.evidence(state)
            if support > opportunities:
                violations.append({'state': state, 'k': support, 'N': opportunities})
        return violations

    def compact(self):
        """Drop the identity payloads once `k` is cached. Keeps 200 captures in memory."""
        for state in list(self._observed):
            self.support_for(state)
        for row in self.rows.values():
            row['state_identities'] = {}
            row['support_identities'] = ()
        return self


def load_capture(sample_id, path):
    return Capture(sample_id, path, read_sink(path))

_LOG_LOGGED = re.compile(r'INFO:Frameshift Candidate and Occurrence (\S+): (\d+)\s*$')
_LOG_SKIPPED = re.compile(
    r'INFO:Skipped due to too small number of occurrence (\S+): (\d+)\s*$')
_LOG_CALLED = re.compile(r'INFO:(?:VID|ID):\d+, There is a mutation at (\S+)\s*$')
_LOG_RU = re.compile(r'INFO:RU(\d+) ([ACGTN]+)\s*$')
_LOG_READ_LENGTH = re.compile(r'INFO:Using read length (\d+)\s*$')


def parse_decision_log(path):
    """Which states reached a decision site, and which were called.

    Established by reading `advntr/vntr_finder.py` rather than by assumption. There are
    THREE decision sites and they share one body, `decide_and_record` (`:435-470`):

    - `:472-481`, the repeat-unit candidates. Every member of `sorted_mutations` is
      logged `Frameshift Candidate and Occurrence <State>: <count>` at `:476`,
      unconditionally; `:477-480` then logs `Skipped due to too small number of
      occurrence` and `continue`s when `count < settings.MIN_SUPPORTING_READ_COUNT`;
      otherwise `:481` decides, with log prefix `'VID'`.
    - `:511-521`, the SUFFIX candidates. The boundary gate is `:514`
      (`mutation_position >= suffix_mutation_check_boundary`) and it sits ABOVE the log
      line at `:516`, so a suffix candidate that fails the boundary is never logged at
      all. Prefix `'VID'` (`:521`).
    - `:523-531`, the PREFIX candidates. Gate `:524`
      (`mutation_position <= prefix_mutation_check_boundary`), again above the log line
      at `:526`. Prefix `'ID'` (`:531`) -- the third site is the only one that logs
      `ID:` rather than `VID:`, which is why both are read here.

    So the log line at `:476`/`:516`/`:526` defines the set of candidates that reached a
    decision site, and it already has the flank boundary gates applied: the gates are
    upstream of the line, not downstream. The TESTED set is that set minus the states
    that also carry the `Skipped` line -- equivalently, those with `count >=
    MIN_SUPPORTING_READ_COUNT`. Both are returned so a caller can check the identity
    rather than assume it.
    """
    if not os.path.isfile(path):
        raise FitterError('decision log %s: file not found' % path)
    logged = {}
    skipped = {}
    called = []
    repeat_units = {}
    read_length = None
    with open(path) as handle:
        for number, line in enumerate(handle, 1):
            match = _LOG_LOGGED.search(line)
            if match:
                state, count = match.group(1), int(match.group(2))
                if state in logged and logged[state] != count:
                    raise FitterError(
                        'decision log %s:%d: %s is logged twice with different counts '
                        '(%d and %d); a multi-VNTR run cannot be scored as one sample'
                        % (path, number, state, logged[state], count))
                logged[state] = count
                continue
            match = _LOG_SKIPPED.search(line)
            if match:
                state, count = match.group(1), int(match.group(2))
                if state not in logged:
                    raise FitterError(
                        'decision log %s:%d: %s is skipped without ever being logged as '
                        'a candidate; the log does not describe the code at '
                        'advntr/vntr_finder.py:476-480' % (path, number, state))
                skipped[state] = count
                continue
            match = _LOG_CALLED.search(line)
            if match:
                called.append(match.group(1))
                continue
            match = _LOG_RU.search(line)
            if match:
                repeat_units[match.group(1)] = match.group(2)
                continue
            match = _LOG_READ_LENGTH.search(line)
            if match:
                read_length = int(match.group(1))
    tested = dict((state, count) for state, count in logged.items()
                  if state not in skipped)
    return {'logged': logged, 'skipped': skipped, 'tested': tested,
            'called': called, 'repeat_units': repeat_units,
            'read_length': read_length}

class Observation(object):
    """One sample's `(k, N)` table. `evidence` is the only thing the estimator uses."""

    def __init__(self, sample_id, table):
        self.sample_id = sample_id
        self._table = table

    def evidence(self, state):
        return self._table.get(state, (0, 0))


class CaptureObservation(Observation):
    """An `Observation` backed by a real `Capture`, evaluated lazily and memoised."""

    def __init__(self, capture):
        Observation.__init__(self, capture.sample_id, None)
        self.capture = capture

    def evidence(self, state):
        return self.capture.evidence(state)

