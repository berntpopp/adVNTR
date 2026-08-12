"""Pinned decoder invariants and quirks.

These are NOT bugs to fix. They are current behaviour that a rewrite must not silently
change, and in two cases they are the preconditions that make the rewrite sound at all.
Changing any of them is a Tier B decision, not a refactor.
"""
import os
import unittest

from advntr_harness.capture import _ModelCache

MODELS = os.path.join(os.path.dirname(__file__), 'golden', 'models')
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _ModelFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cache = _ModelCache(MODELS)
        cls.model, cls.fingerprint, cls.recruitment_score = cache.get('hg19', 151)


class TestStateIndexing(_ModelFixture):
    def test_position_in_states_is_the_state_index(self):
        """The DP addresses states by list position; CSR and vpath both rely on it.

        `state_to_index` is a private cdef attribute and cannot be read from Python, so
        this asserts the property that matters rather than the mapping itself: the
        neighbour lists bake() stores are indices into `states`, and every one is in range.
        """
        n_states = len(self.model.states)
        self.assertEqual(n_states, self.model.n_states)
        for state, neighbours in self.model.neighbors.items():
            for index in neighbours:
                self.assertGreaterEqual(index, 0)
                self.assertLess(index, n_states)


class TestNeighbourOrder(_ModelFixture):
    def test_neighbour_lists_are_sorted_ascending(self):
        """bake() stores sorted() neighbours and the DP relaxes in that order.

        Because the relaxation guard is `> 1e-10` rather than `> 0`, the DP is not an
        order-independent fixpoint: a chain of sub-epsilon improvements is truncated
        differently under a different order. Any rewrite that permutes these lists can
        change the reported path without changing any score.
        """
        for state, neighbours in self.model.neighbors.items():
            self.assertEqual(list(neighbours), sorted(neighbours))


class TestNoSilentSelfLoops(_ModelFixture):
    def test_no_silent_state_relaxes_itself(self):
        """The precondition for hoisting the source cell out of the neighbour loop.

        A silent state writes into its own column, so if one could relax itself the
        hoisted value would go stale mid-loop and the rewrite would diverge from the
        original, which re-read the cell on every neighbour.
        """
        index_of = dict((id(state), position)
                        for position, state in enumerate(self.model.states))
        offenders = []
        for state, neighbours in self.model.neighbors.items():
            if state.distribution is not None:
                continue
            position = index_of.get(id(state))
            if position is not None and position in list(neighbours):
                offenders.append(position)
        self.assertEqual(offenders, [], 'silent self-loops at %r' % offenders)

    def test_the_model_really_does_contain_silent_states(self):
        """Otherwise the test above passes vacuously."""
        silent = [s for s in self.model.states if s.distribution is None]
        self.assertGreater(len(silent), 100)


class TestRelaxationGuard(unittest.TestCase):
    def test_guard_is_an_epsilon_not_a_strict_improvement(self):
        """Source-level tripwire, not a proof.

        A behavioural test would need a read whose optimal path turns on a sub-1e-10
        margin, which is not constructible against the real model. This at least fails
        loudly if someone "cleans up" the epsilon into `> 0`, which would change which
        of several near-equal paths is reported.
        """
        with open(os.path.join(REPO, 'hmm', 'hmm.pyx')) as handle:
            source = handle.read()
        self.assertIn('> 1e-10', source)
        occurrences = source.count('> 1e-10')
        self.assertGreaterEqual(occurrences, 2,
                                'expected the guard in both the silent and emitting '
                                'branches; found %d' % occurrences)


class TestEmissionAlphabet(_ModelFixture):
    def test_every_emitting_state_declares_all_four_bases(self):
        """A flat emission cache defaults missing symbols to something.

        The original DiscreteDistribution.__getitem__ raised KeyError on a missing base.
        A zero-filled cache would instead return log(1.0) -- certainty -- turning a loud
        failure into a silent wrong answer. This pins that the question never arises.
        """
        for position, state in enumerate(self.model.states):
            if state.distribution is None:
                continue
            keys = set(state.distribution.log_emission.keys())
            self.assertEqual(keys, set([0, 1, 2, 3]),
                             'state %d declares %r' % (position, sorted(keys)))


if __name__ == '__main__':
    unittest.main()
