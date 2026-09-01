"""Workload characterization for the Viterbi DP -- how much of its work is wasted.

`hmm/_viterbi_fill_core.pxi` (included by both hmm/hmm.pyx and hmm/hmm_instrumented.pyx,
see Task 3 fix round 1 / task-3-report.md) pushes a state onto the DP work queue once
per improving relaxation into it, but only reads its cell at POP time. A state relaxed
twice within one column (by two different sources, before either push is popped) is
therefore popped twice: the first pop does real work, the second re-reads the exact
same value the first pop already finalised, recomputes identical scores for every
neighbour, and writes nothing. That is what the pop-time duplicate skip (the
`last_col`/`last_val` check immediately after `current = dynamic_table[row, col]`)
exists to short-circuit.

This module measures that waste through `decode_with_counters`
(advntr_harness/workload.py), which drives `hmm.hmm_instrumented.decode_instrumented`
-- a SEPARATE compiled extension from production `hmm.hmm`, built from the identical
`_viterbi_fill_core.pxi` source with `DEF INSTRUMENTED = True` instead of False, so this
module's counting and its `skip_enabled` toggle cost the production build nothing (not
a guard, not a branch -- see hmm/_viterbi_fill_core.pxi's docstring for why a runtime
guard, measured, was rejected). Production's `Model.viterbi(sequence)` takes no
counters/dp_tables/skip_enabled arguments at all.
"""
import gzip
import os
import subprocess
import sys
import unittest

import numpy as np

from advntr_harness.capture import _ModelCache
from advntr_harness.workload import decode_instrumented, decode_with_counters
from hmm.base import DiscreteDistribution, State
from hmm.hmm import Model

GOLDEN = os.path.join(os.path.dirname(__file__), 'golden')
MODELS = os.path.join(GOLDEN, 'models')
READS = os.path.join(GOLDEN, 'tier1_reads.txt.gz')

# Fix round 1 (Finding 1): the naive-prototype failure this guards against is a
# HANG (AGENTS.md Traps), so the production half of the test below runs in a
# `timeout`-wrapped subprocess, not in-process -- see _early_break_worker.py.
_EARLY_BREAK_THRESHOLD = -23.0
_EARLY_BREAK_TIMEOUT_S = 30
_EARLY_BREAK_WORKER = os.path.join(os.path.dirname(__file__), '_early_break_worker.py')

has_fixtures = unittest.skipUnless(
    os.path.isfile(READS),
    'Tier 1 fixtures not captured (python -m advntr_harness.capture --tier 1 --out tests/golden)')


def _first_fixture(model_key):
    """The first Tier 1 fixture sequence captured under `model_key`.

    hg19@151 is what example_7a61/example_b178 derive (AGENTS.md Traps), and is the
    model this fork's per-attempt timing figures (61.1 ms/attempt pristine, the
    audit's 605,110 writes/attempt) were measured against.
    """
    with gzip.open(READS) as handle:
        content = handle.read()
    for line in content.split('\n'):
        if not line:
            continue
        key, sequence = line.split('\t', 1)
        if key == model_key:
            return sequence
    raise AssertionError('no %s fixture in %s' % (model_key, READS))


@has_fixtures
class TestDecoderWorkload(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cache = _ModelCache(MODELS)
        cls.model, _fingerprint, _score = cache.get('hg19', 151)
        cls.read = _first_fixture('hg19@151')

    def test_no_pop_reprocesses_an_unchanged_cell(self):
        """A state is pushed once per successful relaxation, but the cell value is read
        at POP time, so every pop after the first re-reads an already-final value,
        recomputes identical scores and writes nothing. Measured: 41-45% of pops.

        This is the brief's Step 1 assertion, verbatim, and it now passes for real: with
        the skip_enabled toggle from Task 3 fix round 1 (Finding 1), `noop_pops` is
        redefined to count pops that REACH the edge loop despite an unchanged cell --
        the work the skip eliminates -- rather than every pop matching the redundant
        condition regardless of outcome. Against skip_enabled=True (the default, and
        exactly what production does), such a pop always `continue`s before that
        counting site is reached, so this is 0 by construction, not by a runtime check
        that happens to always be false. See test_disabling_the_skip_reproduces_the_
        measured_noop_fraction below for the skip-disabled measurement this task's
        first round reported instead of shipping this assertion; that round's own
        report (task-3-report.md) explains why the ORIGINAL `noop_pops` definition
        could never satisfy this literal check.
        """
        counters = decode_with_counters(self.model, self.read)
        self.assertEqual(counters['noop_pops'], 0)

    def test_disabling_the_skip_reproduces_the_measured_noop_fraction(self):
        """With the skip's `continue` disabled (skip_enabled=False), every pop that
        matches the redundant condition falls through into the edge loop instead of
        short-circuiting -- reproducing the ORIGINAL, pre-fix decoder's behaviour
        exactly, and with it the 41-45% figure the brief's docstring names."""
        counters = decode_with_counters(self.model, self.read, skip_enabled=False)
        self.assertGreater(counters['pops'], 0)
        fraction = counters['noop_pops'] / float(counters['pops'])
        self.assertGreaterEqual(fraction, 0.30, 'noop_pops fraction %.3f' % fraction)
        self.assertLessEqual(fraction, 0.55, 'noop_pops fraction %.3f' % fraction)

    def test_write_count_is_unchanged_with_the_skip_disabled(self):
        """Step 4's "independently assert the successful-write count is unchanged",
        as a committed regression test rather than a one-time manual edit (Task 3 fix
        round 1, Finding 2): the same read, through the same compiled fill, with only
        `skip_enabled` toggled. `pops` and `successful_writes` must be identical either
        way -- proof the skip changes only which pops do wasted work, never a push or a
        write -- while the wasted work itself (`edge_relaxations`, and `noop_pops`
        specifically) must be strictly smaller with the skip enabled.
        """
        skip_on = decode_with_counters(self.model, self.read, skip_enabled=True)
        skip_off = decode_with_counters(self.model, self.read, skip_enabled=False)

        self.assertEqual(skip_on['pops'], skip_off['pops'])
        self.assertEqual(skip_on['successful_writes'], skip_off['successful_writes'])

        self.assertEqual(skip_on['noop_pops'], 0)
        self.assertGreater(skip_off['noop_pops'], 0)
        self.assertLess(skip_on['edge_relaxations'], skip_off['edge_relaxations'])

    def test_write_count_invariant_holds(self):
        """`pops <= successful_writes + 1`: every push comes from one successful write
        and every push is popped at most once (plus the one seed push), so pops can
        never exceed that bound. It is not equality -- AGENTS.md's Traps section notes
        the main loop drains only `col in range(sequence_length)`, so states pushed
        into the final column (`col == sequence_length`) are written (and counted as
        successful_writes) but never popped inside `_viterbi_fill`. Measured on this
        read: 605110 writes, 602080 pops, a gap of 3031 -- exactly the undrained
        final-column states, not a counting bug. Holds regardless of skip_enabled."""
        for skip_enabled in (True, False):
            counters = decode_with_counters(self.model, self.read, skip_enabled=skip_enabled)
            self.assertGreater(counters['pops'], 0)
            self.assertLessEqual(counters['pops'], counters['successful_writes'] + 1)

    def test_edge_relaxations_exceed_successful_writes(self):
        """Every successful write came from an edge relaxation that passed the
        threshold, but most relaxations fail it -- so edges scanned must exceed
        writes made, on any read that visits more than a handful of states."""
        counters = decode_with_counters(self.model, self.read)
        self.assertGreater(counters['edge_relaxations'], counters['successful_writes'])

    def test_early_break_leaves_the_final_column_at_negative_infinity(self):
        """Task 5's Step 1: synthesises the early-break case the real corpus never
        hits (0 of 800 attempts empty the queue before the final column). Fix
        round 1 folded in two review findings, both about THIS test rather than
        the DP it exercises (task-5-report.md Fix round 1).

        Half 1 (Minor 5) -- IN-PROCESS: proves `dp_score_threshold=-23.0` still
        causes a GENUINE early break with real work behind it, rather than
        trusting that claim to prose. Without this, a future change to the score
        landscape could make -23.0 degenerate into trivial immediate rejection
        (break at column 0, nothing written) and this test would keep passing
        while silently no longer exercising the crux -- the corpus itself never
        does (0 of 800 attempts), so this synthetic case is the ONLY guard for
        it. Safe to run directly: `decode_instrumented`'s table is always
        full-size (Task 5 only rolls PRODUCTION's), so this half cannot hang.

        Half 2 (Finding 1) -- SUBPROCESS, `timeout`-bounded: the naive
        rolled-table prototype (task-5-report.md Sec 2) did not merely return a
        wrong `logp` and stop -- the wrong finite value sent `Model.viterbi`'s
        traceback into an unbounded walk over `vpath_table_row` cells nothing
        legitimately wrote. That is a tight `nogil`-compiled C loop holding the
        GIL: it never reaches the bytecode dispatch point that delivers a
        pending signal, so `signal.alarm` cannot stop it (AGENTS.md Traps). A
        bare in-process assertion here would therefore hang `make test`/CI for
        GitHub Actions' 360-minute default instead of failing fast on a
        regression -- so the production call runs in `_early_break_worker.py`
        under `timeout`, and a killed subprocess FAILS this test loudly.
        """
        original = self.model.dp_score_threshold
        self.model.dp_score_threshold = _EARLY_BREAK_THRESHOLD
        try:
            tables = {}
            counters = decode_instrumented(self.model, self.read, dp_tables=tables)
        finally:
            self.model.dp_score_threshold = original

        dynamic_table = tables['dynamic_table']
        written = np.where(np.any(dynamic_table != -np.inf, axis=0))[0]
        last_written_column = int(written.max())
        final_column = dynamic_table.shape[1] - 1
        self.assertGreater(last_written_column, 0,
                            'break happened at column 0 -- no real work behind it')
        self.assertLess(
            last_written_column, final_column,
            'fill reached column %d (the final one) -- not an early break; '
            '-23.0 no longer isolates this case' % final_column)
        self.assertGreater(
            int(counters[3]), 1000,
            'only %d successful writes -- too small to trust as genuine work, '
            'not a threshold so extreme nothing gets written' % counters[3])

        proc = subprocess.Popen(
            ['timeout', str(_EARLY_BREAK_TIMEOUT_S), sys.executable, _EARLY_BREAK_WORKER,
             MODELS, READS, 'hg19@151', repr(_EARLY_BREAK_THRESHOLD)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout, _stderr = proc.communicate()
        self.assertEqual(
            proc.returncode, 0,
            'production Model.viterbi did not return within %ds (returncode=%r) '
            '-- this is the naive-prototype HANG this test exists to catch, not '
            'a clean assertion failure; see AGENTS.md Traps. subprocess output:\n%s'
            % (_EARLY_BREAK_TIMEOUT_S, proc.returncode, stdout))
        self.assertEqual(stdout.strip(), '-inf')

    def test_predecessor_column_matches_the_relaxation_that_wrote_each_cell(self):
        """Task 4 deleted `vpath_table_col`: a silent source always relaxes
        in-column and an emitting source always into the next column, so a written
        cell (r, c)'s predecessor column is fully determined by its predecessor row
        alone -- `c if silent[vpath_row[r, c]] else c - 1`. A one-time check against
        the OLD stored column, run before it was deleted, found 0 violations across
        1,041,573 written cells spanning 3 decodes and both Tier 1 model contexts
        (task-4-report.md) -- the licence to delete it. With the table gone there is
        nothing left to compare the derived column against directly, so this
        committed regression checks the DP's own recurrence instead: for every
        written cell, dynamic_table[r, c] must equal dynamic_table at the DERIVED
        predecessor cell plus the CSR edge weight (plus the emission, if that
        predecessor is not silent) -- proof the derived column names the cell whose
        relaxation actually produced (r, c), not merely a structurally plausible
        one."""
        tables = {}
        decode_instrumented(self.model, self.read, dp_tables=tables)
        dynamic_table = tables['dynamic_table']
        vpath_row = tables['vpath_row']
        silent = tables['silent']

        indptr = np.asarray(self.model.nbr_indptr)
        indices = np.asarray(self.model.nbr_indices)
        weights = np.asarray(self.model.nbr_logp)
        emissions = np.asarray(self.model.emissions)
        encoded = np.asarray(self.model.get_encoded_sequence(self.read))
        start_index = self.model.state_to_index[self.model.start]

        # One CSR pass, not a per-cell search: (from_row, to_row) -> edge weight.
        edge_weight = {}
        for from_row in range(len(indptr) - 1):
            for k in range(indptr[from_row], indptr[from_row + 1]):
                edge_weight[(from_row, int(indices[k]))] = weights[k]

        n_states, n_cols = dynamic_table.shape
        checked = 0
        for col in range(n_cols):
            for row in np.where(dynamic_table[:, col] != -np.inf)[0]:
                row = int(row)
                if row == start_index and col == 0:
                    continue  # the seed, not a relaxation target
                pred_row = int(vpath_row[row, col])
                pred_col = col if silent[pred_row] else col - 1
                self.assertGreaterEqual(pred_col, 0)
                self.assertNotEqual(dynamic_table[pred_row, pred_col], float('-inf'))

                expected = dynamic_table[pred_row, pred_col] + edge_weight[(pred_row, row)]
                if not silent[pred_row]:
                    expected += emissions[pred_row, encoded[pred_col]]
                self.assertAlmostEqual(dynamic_table[row, col], expected, places=9)
                checked += 1

        self.assertGreater(checked, 300000, 'checked %d cells' % checked)

    def test_production_call_is_unaffected(self):
        """`model.viterbi(sequence)` -- no counters, no dp_tables, no skip_enabled --
        is exactly what every production call site uses, and must keep returning what
        it always has. Production's Model.viterbi does not even ACCEPT those keywords
        any more (Task 3 fix round 1): passing them is a TypeError, not a silent no-op,
        which is the point -- there is nothing left in the production build for a stray
        instrumentation call to silently do nothing to."""
        logp, vpath = self.model.viterbi(self.read)
        self.assertIsInstance(logp, float)
        self.assertGreater(len(vpath), 0)

        self.assertRaises(TypeError, self.model.viterbi, self.read, counters=None)


class TestBakeRejectsANonSilentFinalRelaxationSource(unittest.TestCase):
    """`viterbi()`'s hardcoded final relaxation (states[n_states-2] into its
    neighbours) derives -- does not store -- its write's predecessor column, sound
    only because that source relaxes in-column, i.e. is silent (Task 4). Needs no
    Tier 1 fixtures: builds the smallest topology that puts an EMITTING state at
    states[n_states-2] directly, so this guards the inference against topology
    drift rather than against anything a real MUC1 model happens to look like
    today."""

    def test_bake_raises_when_the_second_to_last_state_emits(self):
        model = Model(name='t4-topology-guard')
        mid = State(DiscreteDistribution({'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25}),
                    name='mid')
        model.add_state(mid)
        model.set_transition(model.start, mid, 1.0)
        model.set_transition(mid, model.end, 1.0)

        self.assertRaises(ValueError, model.bake, sort_by_name=True)

    def test_a_silent_second_to_last_state_bakes_cleanly(self):
        model = Model(name='t4-topology-guard-silent')
        mid = State(None, name='mid')
        model.add_state(mid)
        model.set_transition(model.start, mid, 1.0)
        model.set_transition(mid, model.end, 1.0)

        model.bake(sort_by_name=True)
        self.assertTrue(model.is_baked)


class TestSourceNoLongerUsesThePerAttemptGilBoundPatterns(unittest.TestCase):
    """Task 6: per-attempt score-table allocation, dict-based sequence encoding, and
    the O(n^2) `vpath.insert(0, ...)` traceback all held the GIL every decode attempt.
    Reading the SOURCE (not just behaviour) is deliberate: a new path added beside a
    still-present old one would pass any behavioural test while leaving the GIL-bound
    cost exactly where it was. Written and run BEFORE the implementation
    (task-6-report.md Steps 1-2): all three assertions below FAIL against the
    pre-Task-6 source, each for the reason its own message names, not by accident.
    """

    @classmethod
    def setUpClass(cls):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'hmm', 'hmm.pyx')
        with open(path) as handle:
            cls.source = handle.read()
        cls.viterbi_body = cls.source[cls.source.index('cpdef tuple viterbi(self, sequence):'):]

    def test_sequence_encoding_is_not_a_dict_lookup(self):
        self.assertNotIn("{'A':0", self.source,
                         'get_encoded_sequence still keys a dict by base letter')

    def test_traceback_does_not_insert_at_position_zero(self):
        self.assertNotIn('insert(0,', self.viterbi_body,
                         'viterbi() still rebuilds vpath with O(n^2) insert(0, ...)')

    def test_viterbi_does_not_allocate_a_fresh_score_table_per_call(self):
        self.assertNotIn('np.full((self.n_states, 2)', self.viterbi_body,
                         'viterbi() still allocates dynamic_table fresh every attempt '
                         'instead of reusing per-thread scratch')


class TestEncodedSequenceStillRaisesOnAnUndeclaredSymbol(unittest.TestCase):
    """get_encoded_sequence's KeyError on an undeclared symbol is load-bearing: bake()
    already refuses a model whose emitting states do not declare all of A/C/G/T,
    precisely so an undeclared symbol stays a loud failure and not a silent 0
    (hmm.pyx's bake() comment). The 256-entry LUT (Task 6) must keep raising it."""

    def test_an_n_raises_key_error(self):
        model = Model(name='t6-lut-guard')
        self.assertRaises(KeyError, model.get_encoded_sequence, 'N')


class TestBakeAssertsTheRowIdentity(unittest.TestCase):
    """viterbi()'s traceback (Task 6) uses a row it already has as its own index
    instead of re-deriving it through state_to_index[states[row]] -- sound only
    because bake() builds state_to_index as the plain positional inverse of states.
    A state object appended TWICE breaks that: dict construction keeps only the
    LATER pair for a repeated key, so the EARLIER position's row no longer matches --
    proving the new guard has teeth, not merely that it exists."""

    def test_bake_raises_when_a_state_is_duplicated_in_the_list(self):
        # All-silent chain start->a->b->dup->end, so this trips ONLY the new
        # identity check -- not the pre-existing "states[n_states-2] must be
        # silent" one (n_states-2 lands on dup, itself silent) or the self-loop
        # one (no state neighbours itself).
        model = Model(name='t6-row-identity-guard')
        a = State(None, name='a')
        b = State(None, name='b')
        dup = State(None, name='dup')
        model.add_state(a)
        model.add_state(dup)
        model.add_state(b)
        model.add_state(dup)  # same State object, appended a second time
        model.set_transition(model.start, a, 1.0)
        model.set_transition(a, b, 1.0)
        model.set_transition(b, dup, 1.0)
        model.set_transition(dup, model.end, 1.0)

        self.assertRaises(ValueError, model.bake, sort_by_name=True)


@has_fixtures
class TestPerThreadScratchAcrossDifferentModelShapesOnOneThread(unittest.TestCase):
    """_thread_scratch (Task 6) must grow or re-key, never alias, when the SAME
    thread decodes two differently shaped models -- different n_states AND a
    different max sequence length. Tier 3 always decodes one model repeatedly, so it
    cannot see this failure mode; this test drives both shapes on one thread,
    interleaved, and checks each against its own reference from before any
    interleaving happened.

    A one-subModel model is deliberate, not merely minimal: it SEGFAULTS on
    pre-Task-6 viterbi() (`self.subModels[1]` under file-wide boundscheck=False,
    unchecked list indexing on a length-1 list), because repeat_start_index/
    repeat_end_index are computed from it and never used again -- confirmed dead by
    grepping the method body -- which Task 6 deletes along with the rest of the
    per-attempt work it does not need (task-6-report.md). Do not run this test
    against pre-Task-6 hmm.pyx: the crash takes the whole test process down, the
    same class of danger as the hang task-5-report.md documents.
    """

    def test_interleaving_two_model_shapes_matches_each_ones_own_reference(self):
        big_model, _fp, _score = _ModelCache(MODELS).get('hg19', 151)
        big_read = _first_fixture('hg19@151')

        # Emitting 'a' then silent 'z': n_states-2 lands on 'z', satisfying bake()'s
        # existing "final relaxation source must be silent" check.
        small_model = Model(name='t6-scratch-shape-guard')
        dist = DiscreteDistribution({'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25})
        a, z = State(dist, name='a'), State(None, name='z')
        small_model.add_state(a)
        small_model.add_state(z)
        small_model.set_transition(small_model.start, a, 1.0)
        small_model.set_transition(a, z, 1.0)
        small_model.set_transition(z, small_model.end, 1.0)
        small_model.bake(sort_by_name=True)
        small_read = 'A'

        big_reference = big_model.viterbi(big_read)
        small_reference = small_model.viterbi(small_read)

        for _ in range(5):
            self.assertEqual(small_model.viterbi(small_read), small_reference)
            self.assertEqual(big_model.viterbi(big_read), big_reference)


if __name__ == '__main__':
    unittest.main()
