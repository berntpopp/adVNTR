"""Structural invariants of the flat tables bake() builds for the Viterbi DP.

These are the properties that make the nogil rewrite equivalent to the dict-and-object
version it replaced. If any of them breaks, the decoder is still fast and quietly wrong.
"""
import os
import struct
import unittest

import numpy as np

from advntr_harness.capture import _ModelCache

MODELS = os.path.join(os.path.dirname(__file__), 'golden', 'models')

class TestBakeInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Lazy, not a module-level skipUnless: that would bake a model (~0.4 s) during
        # test collection, on every run, even when the whole class is skipped.
        cls.model = _ModelCache(MODELS).get('hg19', 151)[0]
        if not hasattr(cls.model, 'nbr_indptr'):
            raise unittest.SkipTest('CSR tables not built yet (pre-rewrite kernel)')
        cls.indptr = np.asarray(cls.model.nbr_indptr)
        cls.indices = np.asarray(cls.model.nbr_indices)
        cls.weights = np.asarray(cls.model.nbr_logp)
        cls.silent = np.asarray(cls.model.silent)
        cls.emissions = np.asarray(cls.model.emissions)

    def test_indptr_is_monotonic_and_correctly_sized(self):
        self.assertEqual(len(self.indptr), self.model.n_states + 1)
        self.assertTrue(np.all(np.diff(self.indptr) >= 0))
        self.assertEqual(self.indptr[0], 0)
        self.assertEqual(self.indptr[-1], len(self.indices))

    def test_weights_are_parallel_to_indices(self):
        self.assertEqual(len(self.weights), len(self.indices))

    def test_csr_reproduces_the_neighbour_dict_including_order(self):
        """Order is semantic: the relaxation guard is `> 1e-10`, not `> 0`."""
        index_of = dict((id(state), position)
                        for position, state in enumerate(self.model.states))
        checked = 0
        for state, neighbours in self.model.neighbors.items():
            position = index_of.get(id(state))
            if position is None:
                continue
            actual = list(self.indices[self.indptr[position]:self.indptr[position + 1]])
            self.assertEqual(actual, list(neighbours),
                             'state %d CSR row differs from neighbors[]' % position)
            checked += 1
        self.assertGreater(checked, 1000, 'invariant checked on too few states')

    def test_packed_weights_are_bit_identical_to_the_dense_matrix(self):
        """COPIED, never recomputed. libm's log is not correctly rounded, so
        re-deriving these could differ in the last ulp and break equivalence."""
        dense = self.model.transition_matrix_view()
        mismatches = 0
        for row in range(self.model.n_states):
            for k in range(self.indptr[row], self.indptr[row + 1]):
                if struct.pack('>d', self.weights[k]) != struct.pack('>d',
                                                                     dense[row, self.indices[k]]):
                    mismatches += 1
        self.assertEqual(mismatches, 0)

    def test_the_dense_view_refuses_assignment(self):
        """The docstring promises "read-only"; np.asarray on the cdef memoryview ALIASES
        the model's storage, so without the flag the promise is false.

        It is not cosmetic. The two decoder stages read different copies of the same
        edges: the main DP reads the CSR copy (hmm.pyx:923/928) while the hardcoded
        final relaxation reads the dense matrix (hmm.pyx:946). Measured on this model,
        writing the final edge through a writable view moved a 151-base sequence's score
        from -335.85084206362586 to -336.85084206362586 while nbr_logp[9015] stayed 0.0
        -- i.e. a test could silently desynchronise the two stages of the decoder it is
        supposed to be protecting.
        """
        dense = self.model.transition_matrix_view()
        self.assertFalse(dense.flags.writeable)
        self.assertRaises(ValueError, dense.__setitem__, (0, 0), 1.0)

    def test_silence_flags_match_distribution_is_none(self):
        for position, state in enumerate(self.model.states):
            self.assertEqual(bool(self.silent[position]), state.distribution is None,
                             'silence flag wrong for state %d' % position)

    def test_silent_states_have_no_finite_emission(self):
        for position in range(self.model.n_states):
            if self.silent[position]:
                self.assertTrue(np.all(np.isneginf(self.emissions[position])))

    def test_emitting_states_have_all_four_bases_finite(self):
        """A -inf here would mean the table invented a value for a missing symbol."""
        for position in range(self.model.n_states):
            if not self.silent[position]:
                self.assertFalse(np.any(np.isneginf(self.emissions[position])),
                                 'emitting state %d has a -inf emission' % position)

    def test_emissions_match_the_distribution_bit_for_bit(self):
        for position, state in enumerate(self.model.states):
            if state.distribution is None:
                continue
            for base, logp in state.distribution.log_emission.items():
                self.assertEqual(struct.pack('>d', self.emissions[position][base]),
                                 struct.pack('>d', logp))

    def test_no_silent_state_is_its_own_neighbour(self):
        """bake() raises on this; assert the shipped model actually satisfies it."""
        for row in range(self.model.n_states):
            if not self.silent[row]:
                continue
            neighbours = self.indices[self.indptr[row]:self.indptr[row + 1]]
            self.assertNotIn(row, list(neighbours))


if __name__ == '__main__':
    unittest.main()
