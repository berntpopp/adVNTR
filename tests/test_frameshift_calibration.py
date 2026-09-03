"""Task 8h: the default-off calibration sink, and the offline recompute it must support.

Every test drives the real `VNTRFinder.find_frameshift_from_selected_reads` from
synthetic vpaths -- no BAM, no HMM, no decoder -- so the sink is exercised through the
real `OpportunityCounter.finalise` hook rather than through hand-built records. The
builders are duplicated from `tests/test_frameshift_opportunities.py:74-118` rather than
imported, following that file's own note: neither module should constrain the other's
fixtures.

The load-bearing test here is `TestOfflineRecompute`: it recomputes every row's
`opportunities` from the sink's span table alone, with the shipped `parse_components` and
`_signature_supports`, and asserts it equals the integer the sink stored. That is what
makes "the sink stores primitives and derives nothing"
(`advntr/frameshift_calibration.py`) checkable rather than asserted, and it is the check
an external fitter runs on every captured line.
"""
import json
import os
import shutil
import tempfile
import unittest

from advntr import frameshift_calibration
from advntr import frameshift_opportunities
from advntr import settings
from advntr.reference_vntr import ReferenceVNTR
import advntr.vntr_finder as vntr_finder_module
from advntr.vntr_finder import SelectedRead, VNTRFinder


REFERENCE_UNIT = 'ACGTACGTACGT'
UNIT_LENGTH = len(REFERENCE_UNIT)
READ_LENGTH = 61


class _FakeState(object):
    def __init__(self, name):
        self.name = name


class _FakeHMM(object):
    read_length_used_to_build_model = READ_LENGTH


class _CallingFinder(VNTRFinder):
    """Always call, so no candidate is hidden behind the p-value cutoff."""

    def identify_frameshift(self, *_args, **_kwargs):
        return 0.0, 1.0, 0.0


def _vpath(state_names):
    return ([(0, _FakeState('start'))] +
            [(i + 1, _FakeState(name)) for i, name in enumerate(state_names)] +
            [(len(state_names) + 1, _FakeState('end'))])


def _unit(events=None, pattern='1', first=1, last=UNIT_LENGTH, close=True):
    """One repeat occurrence. `events` maps a reference position to ordered I/D ops."""
    events = events or {}
    states = ['unit_start_%s' % pattern]
    sequence = []
    for position in range(first, last + 1):
        operations = events.get(position, [])
        if any(kind == 'D' for kind, _payload in operations):
            states.append('D%d_%s' % (position, pattern))
        else:
            states.append('M%d_%s' % (position, pattern))
            sequence.append(REFERENCE_UNIT[position - 1])
        for kind, payload in operations:
            if kind == 'I':
                for inserted in payload:
                    states.append('I%d_%s' % (position, pattern))
                    sequence.append(inserted)
    if close:
        states.append('unit_end_%s' % pattern)
    return states, ''.join(sequence)


def _read(units, query_name):
    states = []
    sequence = []
    for unit_states, unit_sequence in units:
        states.extend(unit_states)
        sequence.append(unit_sequence)
    return SelectedRead(''.join(sequence), -1.0, _vpath(states), query_name=query_name)


def _clean_read(query_name):
    return _read([_unit(), _unit()], query_name)


def _deletion_read(position, query_name):
    return _read([_unit({position: [('D', '')]}), _unit()], query_name)


def _insertion_read(inserted_sequence, query_name, position=2):
    return _read([_unit({position: [('I', inserted_sequence)]}), _unit()], query_name)


def _deletions_in_two_occurrences(first, second, query_name):
    """One read whose whole-read fusion names a compound `State` its rows do not.

    `legacy_mutation_candidates` fuses adjacent deletions across occurrences
    (`advntr/mutation_keys.py:189`), so this read produces occurrence-scoped rows
    `D%d_1` and `D%d_1` whose `state_identities` name the fused compound -- the shape
    the subset obligation below is about.
    """
    return _read([_unit({first: [('D', '')]}), _unit({second: [('D', '')]})], query_name)


#: Read names are deliberately conspicuous: the anonymity test greps the written line for
#: them, mirroring `tests/test_frameshift_context.py:199`.
READS = [
    _clean_read('CONSPICUOUS-clean-a'),
    _clean_read('CONSPICUOUS-clean-b'),
    _deletion_read(3, 'CONSPICUOUS-deleted'),
    _insertion_read('TC', 'CONSPICUOUS-inserted'),
    _deletions_in_two_occurrences(6, 7, 'CONSPICUOUS-fused'),
]


def denominator_from_spans(spans, state):
    """`N` for any `State`, from the sink's span table alone.

    This is the whole point of the sink and it is written the way an external fitter has
    to write it: nothing but the shipped `parse_components` and `_signature_supports`,
    the six-element span entries, and no access to the run. A `State` that produced no
    row in this sample still gets its denominator here.
    """
    components = frameshift_opportunities.parse_components(state)
    if components is None:
        return 0
    total = 0
    for entry in spans:
        if frameshift_opportunities._signature_supports(tuple(entry[:5]), components):
            total += entry[5]
    return total


class _SinkFixture(unittest.TestCase):
    """Drive the real finder, keep the counter, and give every test a private sink path."""

    def setUp(self):
        self.original_ref_alignment = settings.USE_REF_ALIGNMENT
        self.original_min_support = settings.MIN_SUPPORTING_READ_COUNT
        self.original_full_ru = settings.USE_ONLY_FULLY_COVERED_RU
        self.original_sink = settings.FRAMESHIFT_CALIBRATION_OUT
        self.original_get_pattern_clusters = vntr_finder_module.get_pattern_clusters
        self.original_counter = vntr_finder_module.OpportunityCounter
        settings.USE_REF_ALIGNMENT = False
        settings.USE_ONLY_FULLY_COVERED_RU = False
        settings.MIN_SUPPORTING_READ_COUNT = 1
        settings.FRAMESHIFT_CALIBRATION_OUT = None
        vntr_finder_module.get_pattern_clusters = lambda patterns: [[patterns[0], patterns[1]]]

        self.counters = []
        fixture = self

        class _RecordingCounter(self.original_counter):
            def __init__(self, *args, **kwargs):
                fixture.original_counter.__init__(self, *args, **kwargs)
                fixture.counters.append(self)

        vntr_finder_module.OpportunityCounter = _RecordingCounter
        self.directory = tempfile.mkdtemp(prefix='advntr-calibration-')
        self.path = os.path.join(self.directory, 'sink.jsonl')

    def tearDown(self):
        vntr_finder_module.OpportunityCounter = self.original_counter
        vntr_finder_module.get_pattern_clusters = self.original_get_pattern_clusters
        settings.FRAMESHIFT_CALIBRATION_OUT = self.original_sink
        settings.MIN_SUPPORTING_READ_COUNT = self.original_min_support
        settings.USE_ONLY_FULLY_COVERED_RU = self.original_full_ru
        settings.USE_REF_ALIGNMENT = self.original_ref_alignment
        shutil.rmtree(self.directory, ignore_errors=True)

    def _finder(self, vntr_id=1):
        reference = ReferenceVNTR(vntr_id, REFERENCE_UNIT, 100, 'chr1', None, None)
        reference.init_from_xml([REFERENCE_UNIT, REFERENCE_UNIT], 'TTTTTTTT', 'GGGGGGGG')
        finder = _CallingFinder(reference)
        finder.hmm = _FakeHMM()
        return finder

    def _run(self, reads=None, vntr_id=1):
        finder = self._finder(vntr_id)
        finder.find_frameshift_from_selected_reads(READS if reads is None else reads)
        return finder.last_frameshift_opportunities

    def _lines(self):
        with open(self.path) as handle:
            return handle.read().splitlines()

    def _capture(self, reads=None, vntr_id=1):
        """One flag-on invocation; returns the decoded single line."""
        settings.FRAMESHIFT_CALIBRATION_OUT = self.path
        self._run(reads, vntr_id)
        lines = self._lines()
        self.assertEqual(len(lines), 1)
        return json.loads(lines[0])


class TestDefaultOff(_SinkFixture):
    def test_nothing_is_written_when_the_flag_is_unset(self):
        """Tier B's first requirement: with the setting unset the sink does not exist."""
        self._run()

        self.assertFalse(os.path.exists(self.path))

    def test_the_records_finalise_returns_are_identical_with_the_sink_on(self):
        """The sink observes; it must not move a single field of what `finalise` returns,
        or a calibration capture would not be measuring the run it claims to measure."""
        off = self._run()
        settings.FRAMESHIFT_CALIBRATION_OUT = self.path
        on = self._run()

        self.assertEqual(off, on)
        self.assertTrue(os.path.exists(self.path))

    def test_the_writer_itself_refuses_to_write_when_no_path_is_configured(self):
        """The guard lives in one place, so a future second call site cannot lose it, and
        it runs before anything is read off the finder -- `None` here proves that."""
        self.assertIsNone(
            frameshift_calibration.write_if_configured(None, False, [], {}))


class TestLineShape(_SinkFixture):
    def test_the_line_identifies_itself(self):
        document = self._capture()

        self.assertEqual(document['schema'], 'advntr.frameshift.calibration')
        self.assertEqual(document['version'], 1)
        self.assertEqual(document['vntr_id'], 1)
        self.assertEqual(document['read_length'], READ_LENGTH)
        self.assertEqual(document['is_haploid'], False)
        self.assertEqual([str(key) for key in sorted(document)],
                         ['candidates', 'is_haploid', 'read_length', 'schema', 'spans',
                          'version', 'vntr_id'])

    def test_a_candidate_carries_every_record_field_except_the_span_list(self):
        """A field added to `_record` must reach the fitter or be excluded on purpose;
        this fails on the next field added, which is the point."""
        document = self._capture()
        records = self.counters[-1].finalise({}, {}, {})

        expected = set(records['D3_1']) - set(['opportunity_spans'])
        for candidate in document['candidates']:
            self.assertEqual(set(candidate), expected)

    def test_opportunity_spans_is_the_only_field_dropped(self):
        """2,644,839 bytes against the 59,082-byte span table that regenerates it, on one
        real example_66bf capture -- see this sink's module docstring."""
        self.assertEqual(frameshift_calibration.EXCLUDED_FIELDS, ('opportunity_spans',))

    def test_a_span_entry_is_a_signature_and_a_count(self):
        document = self._capture()

        self.assertTrue(document['spans'])
        for entry in document['spans']:
            self.assertEqual(len(entry), 6)
            pattern_index, reached, inserted, saw_start, saw_end, count = entry
            self.assertTrue(pattern_index is None
                            or isinstance(pattern_index, basestring))
            self.assertIsInstance(reached, int)
            self.assertIsInstance(inserted, int)
            self.assertIsInstance(saw_start, bool)
            self.assertIsInstance(saw_end, bool)
            self.assertGreaterEqual(count, 1)

    def test_the_span_table_counts_distinct_identities(self):
        """`count` is `len(set(identities))`, matching what `finalise` puts in its own
        span ids (`advntr/frameshift_opportunities.py:550-552`); anything else would make
        the offline `N` disagree with the run's."""
        document = self._capture()
        counter = self.counters[-1]

        self.assertEqual([entry[5] for entry in document['spans']],
                         [len(set(identities))
                          for identities in counter._spans.values()])


class TestOfflineRecompute(_SinkFixture):
    """The load-bearing check: the sink's primitives must regenerate the run's integers."""

    def test_every_rows_opportunities_is_recomputed_from_the_span_table(self):
        document = self._capture()

        checked = 0
        for candidate in document['candidates']:
            self.assertEqual(
                denominator_from_spans(document['spans'], candidate['candidate']),
                candidate['opportunities'],
                'offline N disagrees with the run for %s' % candidate['candidate'])
            checked += candidate['opportunities'] > 0
        self.assertGreater(checked, 0, 'no row had a denominator to check')

    def test_the_recompute_fails_when_the_span_table_and_the_rows_disagree(self):
        """A round trip that cannot fail proves nothing. Drop one span's count and the
        rows it fed must stop matching."""
        document = self._capture()
        document['spans'][0][5] += 1

        mismatches = [candidate['candidate'] for candidate in document['candidates']
                      if denominator_from_spans(document['spans'],
                                                candidate['candidate'])
                      != candidate['opportunities']]

        self.assertTrue(mismatches)

    def test_a_state_with_no_row_at_all_still_has_a_denominator(self):
        """The entire reason this sink exists. `D9_1` fired in no read, so `finalise`
        emits no row for it (`advntr/frameshift_opportunities.py:554` iterates
        `set(legacy_support) | set(self._support)`) and the run log carries no `N` for
        it. The span table still does."""
        document = self._capture()

        self.assertNotIn('D9_1', [row['candidate'] for row in document['candidates']])
        self.assertGreater(denominator_from_spans(document['spans'], 'D9_1'), 0)

    def test_the_denominator_of_an_unobserved_state_matches_an_observed_sibling(self):
        """`D9_1` and `D3_1` are the same shape at different positions and every span
        here spans the whole unit, so the state that fired and the state that did not
        must get the same denominator -- the property the fitter relies on."""
        document = self._capture()
        observed = [row for row in document['candidates']
                    if row['candidate'] == 'D3_1'][0]

        self.assertEqual(denominator_from_spans(document['spans'], 'D9_1'),
                         observed['opportunities'])


class TestSubsetObligation(_SinkFixture):
    """`advntr/exact_caller.py:163-185` states the obligation and cannot meet it.

    `decide`'s `support > opportunities` guard compares two integers, so an identity
    attributed to a `State` whose own spans never contained it is invisible at runtime
    with `N` in the tens of thousands. The consumer that holds the span inventory has to
    assert the set property itself. This is the first place in the tree that can.
    """

    @staticmethod
    def _leaked(counter, records, state):
        """Identities credited to `state` that no span supporting `state` recorded."""
        components = frameshift_opportunities.parse_components(state)
        trials = set()
        for signature, identities in counter._spans.items():
            if components and frameshift_opportunities._signature_supports(signature,
                                                                          components):
                trials.update(frameshift_opportunities.anonymous_identities(identities))
        credited = set()
        for row in records.values():
            credited.update(row['state_identities'].get(state, ()))
        return credited - trials

    def test_the_identities_behind_k_are_among_the_trials_n_counts(self):
        settings.FRAMESHIFT_CALIBRATION_OUT = self.path
        records = self._run()
        counter = self.counters[-1]

        states = set()
        for row in records.values():
            states.update(row['legacy_states'])
        self.assertTrue(states)
        for state in sorted(states):
            self.assertEqual(self._leaked(counter, records, state), set(),
                             'identities credited to %s were not trials for it' % state)

    def test_the_check_catches_an_identity_that_was_never_a_trial(self):
        """The fixture the measurement needs: a leak the runtime guard would not see."""
        settings.FRAMESHIFT_CALIBRATION_OUT = self.path
        records = self._run()
        counter = self.counters[-1]
        records['D3_1']['state_identities']['D3_1'] = ((999, 'invented'),)

        self.assertEqual(self._leaked(counter, records, 'D3_1'),
                         set([(999, 'invented')]))


class TestDeterminismAndAnonymity(_SinkFixture):
    def test_two_writes_of_the_same_invocation_are_byte_identical(self):
        settings.FRAMESHIFT_CALIBRATION_OUT = self.path
        self._run()
        self._run()

        lines = self._lines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0], lines[1])

    def test_no_read_name_reaches_the_sink(self):
        """Mirrors `tests/test_frameshift_context.py:199`. `anonymous_identities` drops
        `query_name` and the sink must not reintroduce it."""
        settings.FRAMESHIFT_CALIBRATION_OUT = self.path
        self._run()

        line = self._lines()[0]
        self.assertIn('CONSPICUOUS', ' '.join(read.query_name for read in READS))
        self.assertNotIn('CONSPICUOUS', line)
        self.assertNotIn('query_name', line)

    def test_the_line_is_compact_and_has_no_trailing_whitespace(self):
        settings.FRAMESHIFT_CALIBRATION_OUT = self.path
        self._run()

        line = self._lines()[0]
        self.assertEqual(line, line.rstrip())
        self.assertNotIn(', ', line)
        self.assertNotIn('": ', line)


class TestAppendSemantics(_SinkFixture):
    def test_two_invocations_append_two_self_identifying_lines(self):
        """`genotype -vid` takes a comma-separated list and
        `advntr/genome_analyzer.py:215-216` loops over it, so one file holds several
        invocations and each has to say which VNTR it scored."""
        settings.FRAMESHIFT_CALIBRATION_OUT = self.path
        self._run(vntr_id=25561)
        self._run(vntr_id=915594)

        ids = [json.loads(line)['vntr_id'] for line in self._lines()]
        self.assertEqual(ids, [25561, 915594])

    def test_an_existing_file_is_appended_to_and_not_truncated(self):
        with open(self.path, 'w') as handle:
            handle.write('{"schema":"advntr.frameshift.calibration","resumed":true}\n')
        settings.FRAMESHIFT_CALIBRATION_OUT = self.path
        self._run()

        lines = self._lines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(json.loads(lines[0])['resumed'])


if __name__ == '__main__':
    unittest.main()
