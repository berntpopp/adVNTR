"""The three frameshift decision sites, and the default-off exact caller behind them.

Two things are pinned here, both by driving the real
`VNTRFinder.find_frameshift_from_selected_reads` over synthetic vpaths -- no BAM, no
HMM, no decoder, in the idiom of `tests/test_frameshift_opportunities.py:17-107`.

1. **The collapsed decision helper is behaviourally the three inlined blocks.** The
   fixture makes all three sites fire on one run, each reading a *different* repeat-unit
   index -- the repeat candidate's own `'2'`, `reference_repeat_order[1]` = `'1'` for
   the suffix site and `[-2]` = `'3'` for the prefix site -- with a different coverage
   and copy count behind each, so borrowing another site's index moves a number the
   assertions read. Two sites take the `observed > coverage` short circuit
   (`advntr/vntr_finder.py:187-188`) and one goes through the shipped SciPy statistic,
   so both branches are live. `EXPECTED_LOG` was captured from the three inlined blocks
   before they were collapsed, and is compared line for line -- which is also what pins
   the third site's `ID:` prefix against the other two sites' `VID:`.

2. **The exact caller is default-off and cannot run without a frozen background.** Every
   probability in this file is synthetic (0.125, 0.0625 -- powers of two, absurd for an
   indel background) and calibrated on nothing; PLAN Global Constraints forbid a
   cohort-derived number entering the tree in any form.
"""
import ast
import json
import logging
import os
import shutil
import tempfile
import unittest

from advntr import advntr_commands
from advntr import exact_caller
from advntr import settings
from advntr.exact_tail import exact_indel_tail
from advntr.frameshift_background import BackgroundModelError
from advntr.reference_vntr import ReferenceVNTR
from advntr.vntr_finder import SelectedRead, VNTRFinder


#: Three repeat units differing only in their first base, so `sorted(set(...))` numbers
#: them 1, 2, 3 in the order the reference visits them.
UNITS = {'1': 'ACGTACGTACGT', '2': 'CCGTACGTACGT', '3': 'GCGTACGTACGT'}
UNIT_LENGTH = 12

#: `[1, 2, 2, 2, 3]` makes `reference_repeat_order` `['L', '1', '2', '2', '2', '3', 'R']`,
#: so the suffix site reads `'1'` (one copy), the repeat site `'2'` (THREE copies) and the
#: prefix site `'3'` (one copy): three distinct indices, three distinct denominators. The
#: three copies are load bearing -- with a 12 bp unit, `x / 12 / 2 / c` and
#: `x / (12 * 2 * c)` are bit-identical for every `c` that is a power of two, so only an
#: odd copy count can catch a re-associated division.
SEGMENTS = [UNITS['1'], UNITS['2'], UNITS['2'], UNITS['2'], UNITS['3']]

#: Four drivers put 44 emitted bases behind the repeat index (4 occurrences x 11 after the
#: deletion), and 44/12/2/3 is one of the operand sets where the two associations differ.
DRIVER_READS = 4
COVERAGE_ONLY_READS = 10

SYNTHETIC_BACKGROUND = {
    'schema': 'advntr.frameshift.background',
    'version': 1,
    'provenance': 'SYNTHETIC FIXTURE -- calibrated on nothing, numbers are made up',
    'default_probability': 0.125,
    'states': {'D3_2': 0.0625},
}

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _FakeState(object):
    def __init__(self, name):
        self.name = name


class _FakeHMM(object):
    read_length_used_to_build_model = UNIT_LENGTH


class _ListHandler(logging.Handler):
    def __init__(self):
        logging.Handler.__init__(self)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def _vpath(state_names):
    return ([(0, _FakeState('start'))] +
            [(i + 1, _FakeState(name)) for i, name in enumerate(state_names)] +
            [(len(state_names) + 1, _FakeState('end'))])


def _unit(pattern, deleted=None):
    """One complete occurrence of `pattern`, optionally deleting one position."""
    states = ['unit_start_%s' % pattern]
    sequence = []
    for position in range(1, UNIT_LENGTH + 1):
        if position == deleted:
            states.append('D%d_%s' % (position, pattern))
            continue
        states.append('M%d_%s' % (position, pattern))
        sequence.append(UNITS[pattern][position - 1])
    states.append('unit_end_%s' % pattern)
    return states, ''.join(sequence)


def _read(units, prefix_states=(), suffix_states=(), flank_sequence='', name='read'):
    """Left flank (the `suffix` matcher), the units, then the right flank (`prefix`)."""
    states = list(suffix_states)
    sequence = [flank_sequence[:len(suffix_states)]]
    for unit_states, unit_sequence in units:
        states.extend(unit_states)
        sequence.append(unit_sequence)
    states.extend(prefix_states)
    sequence.append(flank_sequence[len(suffix_states):])
    return SelectedRead(''.join(sequence), -1.0, _vpath(states), query_name=name)


def _all_three_sites():
    """Reads that reach every decision site, plus reads that only add coverage.

    A flank indel is counted only when the same read also carries a repeat-unit mutation
    (`advntr/vntr_finder.py:402-403` short-circuits first), so one read has to carry all
    three. The coverage-only reads lift the suffix site's denominator above its support,
    which is what keeps that site off the `observed > coverage` short circuit.
    """
    drivers = [_read([_unit('1'), _unit('2', deleted=3), _unit('3')],
                     prefix_states=['I0_prefix'], suffix_states=['I12_suffix'],
                     flank_sequence='AA', name='driver-%d' % index)
               for index in range(DRIVER_READS)]
    return drivers + [_read([_unit('1')], name='coverage-%d' % index)
                      for index in range(COVERAGE_ONLY_READS)]


def _capture_log(level, action):
    """Run `action` with the root logger collected into a list, then restore it.

    Restoring matters twice over: these tests assert on the messages, and a stray
    handler would print a deliberate WARNING into an otherwise clean test run.
    """
    root = logging.getLogger()
    saved_level, saved_handlers = root.level, root.handlers[:]
    handler = _ListHandler()
    root.handlers = [handler]
    root.setLevel(level)
    try:
        result = action()
    finally:
        root.handlers = saved_handlers
        root.setLevel(saved_level)
    return result, handler.messages


def _add_argument_keywords(path, flag):
    """The keyword arguments of the `add_argument(flag, ...)` call in `path`."""
    with open(path) as handle:
        tree = ast.parse(handle.read())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, 'attr', None) != 'add_argument' or not node.args:
            continue
        if getattr(node.args[0], 's', None) == flag:
            return dict((keyword.arg, keyword.value) for keyword in node.keywords)
    return None


class _ExactCallerTestCase(unittest.TestCase):
    def setUp(self):
        self.saved = (settings.USE_REF_ALIGNMENT, settings.MIN_SUPPORTING_READ_COUNT,
                      settings.USE_ONLY_FULLY_COVERED_RU,
                      settings.EXACT_FRAMESHIFT_CALLER,
                      settings.FRAMESHIFT_BACKGROUND_FILE)
        settings.USE_REF_ALIGNMENT = False
        settings.USE_ONLY_FULLY_COVERED_RU = False
        settings.EXACT_FRAMESHIFT_CALLER = False
        settings.FRAMESHIFT_BACKGROUND_FILE = None
        self.tempdir = tempfile.mkdtemp(prefix='advntr-exact-caller-test-')

        reference = ReferenceVNTR(1, UNITS['1'], 100, 'chr1', None, None)
        reference.init_from_xml(SEGMENTS, 'TTTTTTTT', 'GGGGGGGG')
        self.finder = VNTRFinder(reference, is_frameshift_mode=True)
        self.finder.hmm = _FakeHMM()

    def tearDown(self):
        shutil.rmtree(self.tempdir, ignore_errors=True)
        (settings.USE_REF_ALIGNMENT, settings.MIN_SUPPORTING_READ_COUNT,
         settings.USE_ONLY_FULLY_COVERED_RU, settings.EXACT_FRAMESHIFT_CALLER,
         settings.FRAMESHIFT_BACKGROUND_FILE) = self.saved

    def _write_background(self, document=None, name='background.json'):
        """A distinct `name` per artifact: `exact_caller` caches on the path for the
        life of the process (see its `_LOADED`), so rewriting one path mid-test would
        not be picked up -- deliberately, since that is the mixed-basis hazard."""
        path = os.path.join(self.tempdir, name)
        with open(path, 'w') as handle:
            json.dump(SYNTHETIC_BACKGROUND if document is None else document, handle)
        return path

    def _run(self, reads=None):
        return self.finder.find_frameshift_from_selected_reads(
            _all_three_sites() if reads is None else reads)

    def _run_capturing_info(self, reads=None):
        """The INFO log, with the shadow counters' one diagnostic line filtered out."""
        results, messages = _capture_log(logging.INFO, lambda: self._run(reads))
        return results, [message for message in messages
                         if not message.startswith('frameshift opportunity counters')]

    @staticmethod
    def _by_state(results):
        return dict((state, (count, coverage, pvalue))
                    for state, count, coverage, pvalue in results or ())


#: Captured from the three inlined blocks before they were collapsed. Coverage is
#: `float(total_bps) / ru_length / 2 / copies` associated left to right, exactly as
#: `advntr/vntr_finder.py:447` wrote it -- `x / (a * b * c)` is equal in exact arithmetic
#: but not guaranteed bit-identical, and these strings would show the difference.
EXPECTED_LOG = [
    'Frameshift Candidate and Occurrence D3_2: 4',
    'Observed repeating base pairs in RU: 44',
    'Average coverage for each base pair in RU: 0.611111111111',
    'Sequencing error prob: 0',
    'Frame-shift prob: 1.0',
    'P-value: 0',
    'VID:1, There is a mutation at D3_2',
    'Frameshift Candidate and Occurrence I12_suffix_LEN1: 4',
    'Observed repeating base pairs in RU: 168',
    'Average coverage for each base pair in RU: 7.0',
    'Sequencing error prob: 3.396046500000007e-07',
    'Frame-shift prob: 0.27062192218332315',
    'P-value: 1.8566359291277707e-07',
    'VID:1, There is a mutation at I12_suffix_LEN1',
    'Frameshift Candidate and Occurrence I0_prefix_LEN1: 4',
    'Observed repeating base pairs in RU: 48',
    'Average coverage for each base pair in RU: 2.0',
    'Sequencing error prob: 0',
    'Frame-shift prob: 1.0',
    'P-value: 0',
    'ID:1, There is a mutation at I0_prefix_LEN1',
]


class TestTheThreeDecisionSites(_ExactCallerTestCase):
    """One helper, three call sites, every difference between them preserved."""

    def test_the_three_sites_log_exactly_what_the_inlined_blocks_logged(self):
        _results, messages = self._run_capturing_info()

        self.assertEqual(messages, EXPECTED_LOG)

    def test_each_site_reads_its_own_repeat_unit_index(self):
        """Coverage is `total_bps / ru_length / ploidy / copies`, and the three indices
        carry (44, 3), (168, 1) and (48, 1), so borrowing another site's index changes
        the reported MeanCoverage."""
        results = self._by_state(self._run())

        self.assertEqual(results['D3_2'][1], float(44) / 12 / 2 / 3)
        self.assertEqual(results['I12_suffix_LEN1'][1], float(168) / 12 / 2 / 1)
        self.assertEqual(results['I0_prefix_LEN1'][1], float(48) / 12 / 2 / 1)

    def test_the_third_site_still_logs_ID_where_the_others_log_VID(self):
        """`advntr/vntr_finder.py:551` against `:459` and `:519`. Almost certainly an
        upstream typo, but a log string is output: changing it is a separate decision,
        not something a refactor may quietly make."""
        _results, messages = self._run_capturing_info()
        mutations = [line for line in messages if 'There is a mutation at' in line]

        self.assertEqual([line.split(':')[0] for line in mutations],
                         ['VID', 'VID', 'ID'])

    def test_the_coverage_division_stays_left_associated(self):
        """`float(x) / a / b / c` is not guaranteed bit-identical to `x / (a * b * c)`.

        The fixture is built so the repeat site lands on operands where the two forms
        really do differ -- 44 emitted bases, a 12 bp unit, 3 copies -- because with a
        power-of-two copy count every divisor is exact and the difference vanishes.
        `'%s' % value` prints 12 significant digits, so `EXPECTED_LOG` shows
        `0.611111111111` either way: only an exact comparison can catch this."""
        results = self._by_state(self._run())

        self.assertNotEqual(float(44) / 12 / 2 / 3, float(44) / (12 * 2 * 3))
        self.assertEqual(results['D3_2'][1], float(44) / 12 / 2 / 3)

    def test_the_haploid_branch_drops_the_ploidy_divisor(self):
        self.finder.is_haploid = True
        results = self._by_state(self._run())

        self.assertEqual(results['D3_2'][1], float(44) / 12 / 3)

    def test_a_candidate_below_the_support_floor_never_reaches_a_site(self):
        settings.MIN_SUPPORTING_READ_COUNT = DRIVER_READS + 1
        results, messages = self._run_capturing_info()

        self.assertIsNone(results)
        self.assertEqual([line for line in messages if line.startswith('Observed')], [])
        self.assertEqual(len([line for line in messages if 'Skipped due to' in line]), 3)


class TestTheFlagIsWiredLikePruneReverse(_ExactCallerTestCase):
    """`--prune-reverse` is the one Tier B flag shipped so far (AGENTS.md); this one
    copies its shape: declared default-off on the command line, written into `settings`
    from `args` in `genotype`, and read nowhere else."""

    def test_the_setting_ships_off(self):
        source = os.path.join(REPO, 'advntr', 'settings.py')
        namespace = {}
        exec(compile(open(source).read(), source, 'exec'), namespace)

        self.assertIs(namespace['EXACT_FRAMESHIFT_CALLER'], False)
        self.assertIsNone(namespace['FRAMESHIFT_BACKGROUND_FILE'])

    def test_the_command_line_declares_the_flag_default_off(self):
        keywords = _add_argument_keywords(os.path.join(REPO, 'advntr', '__main__.py'),
                                          '--exact-frameshift-caller')

        self.assertIsNotNone(keywords)
        self.assertEqual(ast.literal_eval(keywords['action']), 'store_true')
        self.assertIs(ast.literal_eval(keywords['default']), False)

    def test_the_command_line_declares_the_artifact_path_absent_by_default(self):
        keywords = _add_argument_keywords(os.path.join(REPO, 'advntr', '__main__.py'),
                                          '--frameshift-background')

        self.assertIsNotNone(keywords)
        self.assertIsNone(ast.literal_eval(keywords['default']))

    def test_genotype_writes_both_settings_from_args(self):
        """The same one-line shape as `advntr/advntr_commands.py:75`."""
        with open(advntr_commands.__file__.rstrip('c')) as handle:
            tree = ast.parse(handle.read())
        assignments = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Attribute):
                target = node.targets[0]
                if isinstance(target, ast.Attribute):
                    assignments.add((target.attr, node.value.attr))

        self.assertIn(('EXACT_FRAMESHIFT_CALLER', 'exact_frameshift_caller'), assignments)
        self.assertIn(('FRAMESHIFT_BACKGROUND_FILE', 'frameshift_background'), assignments)

    def test_with_the_flag_off_no_background_is_loaded(self):
        settings.FRAMESHIFT_BACKGROUND_FILE = self._write_background()

        self.assertIsNone(exact_caller.configured_background())

    def test_with_the_flag_off_the_shipped_statistic_still_decides(self):
        settings.FRAMESHIFT_BACKGROUND_FILE = self._write_background()
        _results, messages = self._run_capturing_info()

        self.assertEqual(messages, EXPECTED_LOG)


class TestTheFlagOnPathNeedsAnArtifact(_ExactCallerTestCase):
    def test_turning_the_flag_on_without_an_artifact_fails_fast(self):
        """Falling back to the shipped statistic would make the flag a lie, and SPEC
        Q-RATE rules out a plug-in default to fall back to instead."""
        settings.EXACT_FRAMESHIFT_CALLER = True

        with self.assertRaises(BackgroundModelError) as caught:
            exact_caller.configured_background()
        self.assertIn('no background model', str(caught.exception))

    def test_the_failure_reaches_the_caller_rather_than_being_swallowed(self):
        settings.EXACT_FRAMESHIFT_CALLER = True

        self.assertRaises(BackgroundModelError, self._run)

    def test_a_malformed_artifact_fails_the_same_way(self):
        settings.EXACT_FRAMESHIFT_CALLER = True
        settings.FRAMESHIFT_BACKGROUND_FILE = self._write_background(
            dict(SYNTHETIC_BACKGROUND, default_probability=2.0))

        self.assertRaises(BackgroundModelError, self._run)


class TestTheExactCallerWithASyntheticBackground(_ExactCallerTestCase):
    def setUp(self):
        _ExactCallerTestCase.setUp(self)
        settings.EXACT_FRAMESHIFT_CALLER = True
        settings.FRAMESHIFT_BACKGROUND_FILE = self._write_background()

    def test_the_p_value_is_the_exact_tail_over_task_sevens_integer_pair(self):
        results = self._by_state(self._run())
        records = self.finder.last_frameshift_opportunities

        self.assertEqual((records['D3_2']['support'],
                          records['D3_2']['opportunities']), (DRIVER_READS, DRIVER_READS))
        expected = exact_indel_tail(DRIVER_READS, DRIVER_READS, 0.0625)
        # Relative, not `assertAlmostEqual`'s 7 absolute decimal places: this p-value is
        # 1.5e-05 and the default would accept a clamped 0.0 for it. The `Pvalue` column
        # is user-visible output. See `tests/test_exact_tail.py`'s `_TailTestCase`.
        self.assertAlmostEqual(results['D3_2'][2], expected, delta=expected * 1e-9)
        self.assertGreater(results['D3_2'][2], 0.0)

    def test_an_unlisted_state_scores_against_the_artifacts_default(self):
        results = self._by_state(self._run())

        expected = exact_indel_tail(DRIVER_READS, DRIVER_READS, 0.125)
        self.assertAlmostEqual(results['I12_suffix_LEN1'][2], expected,
                               delta=expected * 1e-9)
        self.assertGreater(results['I12_suffix_LEN1'][2], 0.0)

    def test_the_state_column_and_the_coverage_column_are_untouched(self):
        """SPEC 3.5: `State` stays byte-identical and the table stays six columns. Only
        the decision moves, so MeanCoverage is still the legacy quantity."""
        results = self._by_state(self._run())

        self.assertEqual(sorted(results), ['D3_2', 'I0_prefix_LEN1', 'I12_suffix_LEN1'])
        self.assertEqual(results['D3_2'][1], float(44) / 12 / 2 / 3)

    def test_the_run_log_records_which_artifact_scored_the_run(self):
        _results, messages = self._run_capturing_info()

        self.assertTrue([line for line in messages if 'SYNTHETIC FIXTURE' in line
                         and self.tempdir in line])

    def test_a_background_that_makes_the_observation_unremarkable_calls_nothing(self):
        settings.FRAMESHIFT_BACKGROUND_FILE = self._write_background(
            dict(SYNTHETIC_BACKGROUND, default_probability=0.5, states={'D3_2': 0.5}),
            name='unremarkable.json')

        self.assertIsNone(self._run())

    def test_the_shipped_statistic_is_not_consulted_at_all(self):
        """`identify_frameshift` is the malformed statistic (Q-DENOM, Q-STAT, Q-UNDER).
        With the flag on it must not contribute, not even a log line."""
        calls = []
        # The class attribute, not `VNTRFinder.identify_frameshift`: on an old-style
        # class that read unwraps the `staticmethod`, so assigning it back would leave a
        # plain function that swallows `self` as its first argument.
        original = VNTRFinder.__dict__['identify_frameshift']
        VNTRFinder.identify_frameshift = staticmethod(
            lambda *args, **kwargs: calls.append(args) or (0, 1.0, 0.0))
        try:
            _results, messages = self._run_capturing_info()
        finally:
            VNTRFinder.identify_frameshift = original

        self.assertEqual(calls, [])
        self.assertEqual([line for line in messages if line.startswith('Frame-shift prob')],
                         [])


class TestTheArtifactIsReadOncePerRun(_ExactCallerTestCase):
    def test_a_second_vntr_in_one_run_scores_against_the_same_basis(self):
        """`advntr/genome_analyzer.py:215-216` loops over `target_vntr_ids`, so
        `configured_background` runs once per VNTR. Re-reading there would let an
        operator's edit land between two VNTRs of one run."""
        settings.EXACT_FRAMESHIFT_CALLER = True
        path = self._write_background(name='once.json')
        settings.FRAMESHIFT_BACKGROUND_FILE = path

        first = exact_caller.configured_background()
        with open(path, 'w') as handle:
            json.dump(dict(SYNTHETIC_BACKGROUND, default_probability=0.5), handle)
        second = exact_caller.configured_background()

        self.assertIs(first, second)
        self.assertEqual(second.probability_for('anything'), 0.125)

    def test_the_artifact_is_described_in_the_log_once_not_once_per_vntr(self):
        settings.EXACT_FRAMESHIFT_CALLER = True
        settings.FRAMESHIFT_BACKGROUND_FILE = self._write_background(name='logged.json')

        _first, messages = _capture_log(logging.INFO,
                                        exact_caller.configured_background)
        _second, again = _capture_log(logging.INFO,
                                      exact_caller.configured_background)

        self.assertEqual(len([line for line in messages
                              if 'SYNTHETIC FIXTURE' in line]), 1)
        self.assertEqual([line for line in again if 'SYNTHETIC FIXTURE' in line], [])


class _SilentParser(object):
    """Stands in for the `genotype` subparser: `print_error` calls `print_help` on it and
    then `sys.exit`s, and a real parser would write its help to the test output."""

    def print_help(self):
        pass


class TestTheStartupCheckDoesNotWaitForReadSelection(_ExactCallerTestCase):
    """`find_frameshift_from_selected_reads` runs after `select_illumina_reads` has
    decoded every read (`advntr/vntr_finder.py:977-978`), so the library-level raise is
    a backstop. `genotype` is where "fails fast" can actually be true, and it has to
    validate the artifact rather than merely notice that a path was given."""

    class _Args(object):
        #: Not a BAM, so a run that gets PAST the background check stops at the input
        #: format -- which is how the last test tells "accepted" from "refused".
        alignment_file = 'reads.txt'
        fasta = None
        nanopore = False
        pacbio = False
        threads = 1
        prune_reverse = False
        exact_frameshift_caller = True
        frameshift_background = None
        #: The calibration capture is independent of this flag and off here, so these
        #: startup tests exercise the background refusal alone.
        frameshift_calibration_out = None
        expansion = False
        coverage = None

    def setUp(self):
        _ExactCallerTestCase.setUp(self)
        self.saved_globals = (settings.CORES, settings.MAX_ERROR_RATE,
                              settings.PRUNE_REVERSE_DECODE)

    def tearDown(self):
        (settings.CORES, settings.MAX_ERROR_RATE,
         settings.PRUNE_REVERSE_DECODE) = self.saved_globals
        _ExactCallerTestCase.tearDown(self)

    def _genotype_exit(self, background):
        args = self._Args()
        args.frameshift_background = background
        with self.assertRaises(SystemExit) as caught:
            advntr_commands.genotype(args, _SilentParser())
        return str(caught.exception)

    def test_a_missing_path_is_refused_at_startup(self):
        self.assertIn('needs a frozen background model', self._genotype_exit(None))

    def test_a_path_that_does_not_exist_is_refused_at_startup(self):
        """The earlier check tested only for `None`, so a mistyped path reached
        `find_frameshift_from_selected_reads` -- after every read had been decoded."""
        missing = os.path.join(self.tempdir, 'no-such-background.json')
        message = self._genotype_exit(missing)

        self.assertIn(missing, message)
        self.assertIn('not found', message)

    def test_an_artifact_that_does_not_validate_is_refused_at_startup(self):
        broken = self._write_background(dict(SYNTHETIC_BACKGROUND,
                                             default_probability=2.0),
                                        name='broken.json')
        message = self._genotype_exit(broken)

        self.assertIn(broken, message)
        self.assertIn('outside the open interval', message)

    def test_a_valid_artifact_gets_past_the_check(self):
        """It must fail later, on the input file, not on the background."""
        valid = self._write_background(name='valid-at-startup.json')
        message = self._genotype_exit(valid)

        self.assertNotIn(valid, message)
        self.assertIn('file format is not supported', message)


if __name__ == '__main__':
    unittest.main()
