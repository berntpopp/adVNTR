"""Task 8h: the default-off calibration sink, and the offline recompute it must support.

Every test drives the real `VNTRFinder.find_frameshift_from_selected_reads` from
synthetic vpaths -- no BAM, no HMM, no decoder -- so the sink is exercised through the
real `finalise` hook rather than through hand-built records. The builders are duplicated
from `tests/test_frameshift_opportunities.py:74-118` rather than imported, per that file's
own note: neither module should constrain the other's fixtures.

`TestOfflineRecompute` is the load-bearing one. It recomputes every row's `opportunities`
from the span table alone, with the shipped `parse_components` and `_signature_supports`,
and asserts it equals the integer the sink stored -- which is what makes "stores
primitives, derives nothing" checkable rather than asserted, and is the check an external
fitter runs on every captured line.
"""
import json
import os
import shutil
import tempfile
import time
import unittest
from collections import OrderedDict

from advntr import advntr_commands
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


class _StubReference(object):
    def __init__(self, vntr_id):
        self.id = vntr_id


class _StubFinder(object):
    """Only the two attributes the sink reads, for tests about the append mechanics."""

    def __init__(self, vntr_id=1):
        self.reference_vntr = _StubReference(vntr_id)
        self.hmm = _FakeHMM()


class _SilentParser(object):
    """Stands in for the `genotype` subparser, as `tests/test_exact_caller.py:477-482`
    does: `print_error` calls `print_help` and exits, and a real one would print help."""

    def print_help(self):
        pass


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
    """One read whose whole-read fusion names a compound `State` its own rows do not:
    `legacy_mutation_candidates` fuses adjacent deletions across occurrences
    (`advntr/mutation_keys.py:189`). The shape the subset obligation is about.
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


def spin_until(predicate, what, timeout=60.0):
    """Busy-wait on a file barrier, with a deadline so a broken race cannot hang the gate.
    No `sleep`: one cheap enough to use is coarse enough to stagger the writers."""
    deadline = time.time() + timeout
    while not predicate():
        if time.time() > deadline:
            raise RuntimeError('barrier timed out waiting for %s' % what)


def denominator_from_spans(spans, state):
    """`N` for any `State`, from the span table alone -- as an external fitter must write
    it: the shipped `parse_components` and `_signature_supports`, the six-element span
    entries, no access to the run. A `State` with no row still gets its denominator."""
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
        """The sink observes: it must not move a field of what `finalise` returns."""
        off = self._run()
        settings.FRAMESHIFT_CALIBRATION_OUT = self.path
        on = self._run()

        self.assertEqual(off, on)
        self.assertTrue(os.path.exists(self.path))

    def test_the_writer_itself_refuses_to_write_when_no_path_is_configured(self):
        """One guard, and it runs before anything is read off the finder."""
        self.assertIsNone(
            frameshift_calibration.write_if_configured(None, False, [], {}))

    def test_a_counter_without_its_finder_refuses_rather_than_naming_no_vntr(self):
        settings.FRAMESHIFT_CALIBRATION_OUT = self.path
        self.assertRaises(ValueError, frameshift_calibration.write_if_configured,
                          None, False, [], {})
        self.assertFalse(os.path.exists(self.path))


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
        """A field added to `_record` must reach the fitter or be excluded on purpose."""
        document = self._capture()
        records = self.counters[-1].finalise({}, {}, {})

        expected = set(records['D3_1']) - set(['opportunity_spans'])
        for candidate in document['candidates']:
            self.assertEqual(set(candidate), expected)

    def test_opportunity_spans_is_the_only_field_dropped(self):
        """2,644,839 bytes against the 59,082-byte span table that regenerates it."""
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
        """`count` is `len(set(identities))`, as `finalise` computes it
        (`advntr/frameshift_opportunities.py:550-552`); anything else desynchronises N."""
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
        """A round trip that cannot fail proves nothing: perturb one span count."""
        document = self._capture()
        document['spans'][0][5] += 1

        mismatches = [candidate['candidate'] for candidate in document['candidates']
                      if denominator_from_spans(document['spans'],
                                                candidate['candidate'])
                      != candidate['opportunities']]

        self.assertTrue(mismatches)

    def test_a_state_with_no_row_at_all_still_has_a_denominator(self):
        """The entire reason this sink exists. `D9_1` fired in no read, so `finalise`
        emits no row for it (`advntr/frameshift_opportunities.py:554`). Spans still do."""
        document = self._capture()

        self.assertNotIn('D9_1', [row['candidate'] for row in document['candidates']])
        self.assertGreater(denominator_from_spans(document['spans'], 'D9_1'), 0)

    def test_the_denominator_of_an_unobserved_state_matches_an_observed_sibling(self):
        """Same shape, different position, and every span here covers the whole unit, so
        the state that fired and the one that did not must get the same denominator."""
        document = self._capture()
        observed = [row for row in document['candidates']
                    if row['candidate'] == 'D3_1'][0]

        self.assertEqual(denominator_from_spans(document['spans'], 'D9_1'),
                         observed['opportunities'])


class TestSubsetObligation(_SinkFixture):
    """`advntr/exact_caller.py:163-185` states the obligation and cannot meet it: `decide`
    compares two integers, so an identity credited to a `State` whose own spans never held
    it is invisible at runtime with `N` in the tens of thousands. The consumer holding the
    span inventory must assert the set property, and this is the first place that can.
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
        """Mirrors `tests/test_frameshift_context.py:199`: no name may be reintroduced."""
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
        """`advntr/genome_analyzer.py:215-216` loops over `-vid`, so one file holds
        several invocations and each has to say which VNTR it scored."""
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


class TestTheAppendIsNotCorruptibleByAccident(_SinkFixture):
    """One line is 443 KB on a real capture, far above the size at which an append is
    atomic: stdio splits it into many `write()` calls and two writers interleave. A review
    drove eight barrier-synchronised appends into one file and got 0 of 8 lines back.

    **Forking real processes is not incidental.** The thread version caught a deleted lock
    in 14 of 36 runs, 39%, green every time WITH it -- honest, but not a guard. Threads
    share one file table and one GIL, so the window with two writers inside `write()` is
    far narrower than between processes, and forked children take separate open file
    descriptions, which is also what makes `flock` contend. After the rework: 20 of 20
    caught with the lock deleted, 12 of 12 green with it. Each child pre-builds its line
    BEFORE the barrier and calls `_append_line`, so the only thing racing is the critical
    section and not milliseconds of `json.dumps`.
    """

    #: Big enough that stdio splits the write into many `write()` calls, which is what
    #: makes interleaving possible at all. ~794 KB, larger than the real 443 KB line.
    ROWS = 3000

    #: Rounds of the race per run. One forked round is already lethal; three costs
    #: little and removes the last of the flakiness in the other direction.
    ROUNDS = 3
    WRITERS = 8

    def _bulky_records(self, tag):
        records = OrderedDict()
        for index in range(self.ROWS):
            name = 'D%d_1' % index
            records[name] = {'candidate': name, 'support': index, 'ru_length': 1,
                             'opportunities': index + 1, 'support_identities': (),
                             'opportunity_spans': (), 'legacy_support': 0,
                             'legacy_states': [tag * 20], 'state_identities': {},
                             'pattern_index': '1', 'ru_bp_coverage': 1,
                             'ru_bp_coverage_ratio': 1, 'avg_bp_coverage': 1.0}
        return records

    def _race(self, round_index, lines):
        """Fork one writer per line, release them all at once, return their exit codes."""
        go = os.path.join(self.directory, 'go-%d' % round_index)
        ready = [os.path.join(self.directory, 'ready-%d-%d' % (round_index, index))
                 for index in range(len(lines))]
        children = []
        for index, line in enumerate(lines):
            pid = os.fork()
            if pid:
                children.append(pid)
                continue
            status = 1
            try:
                open(ready[index], 'w').close()
                spin_until(lambda: os.path.exists(go), 'the barrier')
                frameshift_calibration._append_line(self.path, line)
                status = 0
            finally:
                # Never unwind into the parent's test run, and never flush its buffers.
                os._exit(status)
        spin_until(lambda: all(os.path.exists(item) for item in ready), 'every writer')
        open(go, 'w').close()
        return [os.waitpid(pid, 0)[1] for pid in children]

    def test_eight_forked_writers_leave_eight_parseable_lines_every_round(self):
        lines = [frameshift_calibration.sink_line(
            index, READ_LENGTH, False, [],
            self._bulky_records(chr(ord('a') + index)))
            for index in range(self.WRITERS)]
        self.assertGreater(len(lines[0]), 700000, 'must span several write() calls')

        for round_index in range(self.ROUNDS):
            self.assertEqual(set(self._race(round_index, lines)), set([0]))

            written = self._lines()
            self.assertEqual(len(written), self.WRITERS * (round_index + 1))
            recovered = sorted(json.loads(line)['vntr_id'] for line in written)
            self.assertEqual(recovered,
                             sorted(list(range(self.WRITERS)) * (round_index + 1)))

    def test_the_public_writer_goes_through_the_locked_append(self):
        """The race targets `_append_line`; this stops a future write around it."""
        calls = []
        original = frameshift_calibration._append_line
        frameshift_calibration._append_line = lambda path, line: calls.append((path, line))
        try:
            settings.FRAMESHIFT_CALIBRATION_OUT = self.path
            returned = self._run()
        finally:
            frameshift_calibration._append_line = original

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], self.path)
        self.assertIn('"vntr_id":1', calls[0][1])
        self.assertTrue(returned)


class TestATornLineCostsOnlyItself(_SinkFixture):
    def test_an_unterminated_final_line_is_not_welded_to_the_next_one(self):
        """Appending onto a killed writer's fragment destroys the next record too."""
        with open(self.path, 'w') as handle:
            handle.write('{"schema":"advntr.frameshift.calibration","torn":tru')
        settings.FRAMESHIFT_CALIBRATION_OUT = self.path
        self._run()

        lines = self._lines()
        self.assertEqual(len(lines), 2)
        self.assertRaises(ValueError, json.loads, lines[0])
        self.assertEqual(json.loads(lines[1])['schema'],
                         'advntr.frameshift.calibration')

    def test_a_properly_terminated_file_gains_no_blank_line(self):
        with open(self.path, 'w') as handle:
            handle.write('{"resumed":true}\n')
        settings.FRAMESHIFT_CALIBRATION_OUT = self.path
        self._run()

        with open(self.path) as handle:
            self.assertNotIn('\n\n', handle.read())


class TestKeysAreSortedEverywhere(_SinkFixture):
    """`sort_keys=True` is the whole determinism guarantee, and until this test nothing
    died when it was removed -- a probe that set `sort_keys=False` survived the suite."""

    def _assert_sorted(self, node, where):
        if isinstance(node, OrderedDict):
            keys = list(node)
            self.assertEqual(keys, sorted(keys), 'unsorted keys at %s' % where)
            for key, value in node.items():
                self._assert_sorted(value, '%s.%s' % (where, key))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                self._assert_sorted(value, '%s[%d]' % (where, index))

    def test_every_object_in_the_line_has_its_keys_in_sorted_order(self):
        settings.FRAMESHIFT_CALIBRATION_OUT = self.path
        self._run()

        document = json.loads(self._lines()[0],
                              object_pairs_hook=OrderedDict)
        self._assert_sorted(document, 'line')
        self.assertGreater(len(document), 1, 'a single-key object would prove nothing')


class TestTheStartupPreflight(_SinkFixture):
    """The only other check is inside `finalise`, after every read has been decoded."""

    class _Args(object):
        alignment_file = 'reads.txt'
        fasta = None
        nanopore = False
        pacbio = False
        threads = 1
        prune_reverse = False
        exact_frameshift_caller = False
        frameshift_background = None
        frameshift_calibration_out = None
        expansion = False
        coverage = None

    def _genotype_exit(self, sink):
        args = self._Args()
        args.frameshift_calibration_out = sink
        with self.assertRaises(SystemExit) as caught:
            advntr_commands.genotype(args, _SilentParser())
        return str(caught.exception)

    def test_an_unwritable_path_is_refused_at_startup(self):
        unwritable = os.path.join(self.directory, 'no-such-directory', 'sink.jsonl')

        self.assertIn('not writable', self._genotype_exit(unwritable))

    def test_a_writable_but_unreadable_path_is_refused_too(self):
        """A preflight opening `a` would pass a write-only path and fail inside
        `finalise` instead -- after the decode this check exists to save."""
        if os.geteuid() == 0:
            raise unittest.SkipTest('root ignores the read bit, so there is no such path')
        write_only = os.path.join(self.directory, 'write-only.jsonl')
        open(write_only, 'w').close()
        os.chmod(write_only, 0222)

        self.assertIn('not writable', self._genotype_exit(write_only))

    def test_a_writable_path_is_accepted_and_created(self):
        """`reads.txt` is not a BAM, so an accepted run stops at the input format --
        how this tells "accepted" from "refused" (`tests/test_exact_caller.py:492-494`)."""
        message = self._genotype_exit(self.path)

        self.assertIn('file format is not supported', message)
        self.assertTrue(os.path.exists(self.path))


if __name__ == '__main__':
    unittest.main()
