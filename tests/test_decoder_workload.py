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
import unittest

import numpy as np

from advntr_harness.capture import _ModelCache
from advntr_harness.workload import decode_instrumented, decode_with_counters
from hmm.base import DiscreteDistribution, State
from hmm.hmm import Model

GOLDEN = os.path.join(os.path.dirname(__file__), 'golden')
MODELS = os.path.join(GOLDEN, 'models')
READS = os.path.join(GOLDEN, 'tier1_reads.txt.gz')

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


if __name__ == '__main__':
    unittest.main()
