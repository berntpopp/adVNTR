"""The threaded decode helper.

Uses a stub model so the decoding contract is tested without a 3-minute HMM build. The
real end-to-end equivalence lives in tests/test_tier3_occurrence.py.
"""
import threading
import unittest

from advntr.read_selection import (PendingRead, _decode_one, decode_in_threads,
                                   decode_pending, decode_serially, resolve_thread_count)


class _StubModel(object):
    """Records which sequences it was asked to decode, and from which thread."""

    def __init__(self, fail_on=None, delay=0.0):
        self.seen = []
        self.threads = set()
        self._lock = threading.Lock()
        self._fail_on = fail_on
        self._delay = delay

    def viterbi(self, sequence):
        if self._delay:
            import time
            time.sleep(self._delay)
        with self._lock:
            self.seen.append(sequence)
            self.threads.add(threading.current_thread().name)
        if self._fail_on is not None and sequence == self._fail_on:
            raise ValueError('boom: %s' % sequence)
        return (-float(len(sequence)), [(0, None), (1, None)])


def _pending(n):
    return [PendingRead(sequence='A' * (i + 1), reverse='T' * (i + 1),
                        query_name='r%d' % i) for i in range(n)]


class TestResolveThreadCount(unittest.TestCase):
    def test_none_and_garbage_fall_back_to_serial(self):
        for value in (None, 'x', '', object()):
            self.assertEqual(resolve_thread_count(value), 1)

    def test_zero_and_negative_are_serial(self):
        self.assertEqual(resolve_thread_count(0), 1)
        self.assertEqual(resolve_thread_count(-4), 1)

    def test_a_real_count_is_honoured(self):
        self.assertEqual(resolve_thread_count(8), 8)

    def test_a_numeric_string_is_honoured(self):
        """settings.CORES is written from argparse and has been a string before."""
        self.assertEqual(resolve_thread_count('8'), 8)


class TestDecoding(unittest.TestCase):
    def test_serial_decodes_every_read_twice(self):
        model, pending = _StubModel(), _pending(5)
        decode_serially(model, pending)
        self.assertEqual(len(model.seen), 10)
        for item in pending:
            self.assertIsNotNone(item.logp)
            self.assertIsNotNone(item.rev_logp)

    def test_threaded_decodes_every_read_exactly_once_per_orientation(self):
        model, pending = _StubModel(), _pending(50)
        decode_in_threads(model, pending, 8)
        self.assertEqual(len(model.seen), 100)
        self.assertEqual(sorted(model.seen), sorted([p.sequence for p in pending] +
                                                    [p.reverse for p in pending]))

    def test_threaded_and_serial_agree(self):
        serial_model, serial = _StubModel(), _pending(30)
        decode_serially(serial_model, serial)
        threaded_model, threaded = _StubModel(), _pending(30)
        decode_in_threads(threaded_model, threaded, 8)
        self.assertEqual([(p.logp, p.rev_logp) for p in serial],
                         [(p.logp, p.rev_logp) for p in threaded])

    def test_threads_are_actually_used(self):
        """Otherwise every scaling claim is untested."""
        model, pending = _StubModel(delay=0.002), _pending(40)
        decode_in_threads(model, pending, 4)
        self.assertGreater(len(model.threads), 1)

    def test_reads_rejected_before_decoding_are_never_decoded(self):
        pending = _pending(4)
        pending[1].reject_before_decode = 'short'
        pending[3].reject_before_decode = 'duplicate'
        model = _StubModel()
        decode_in_threads(model, pending, 4)
        self.assertEqual(len(model.seen), 4)
        self.assertIsNone(pending[1].logp)
        self.assertIsNone(pending[3].logp)

    def test_a_worker_exception_is_re_raised_on_the_caller(self):
        """A silently dead worker would leave logp as None and fail confusingly later."""
        model, pending = _StubModel(fail_on='AAA'), _pending(10)
        with self.assertRaises(ValueError):
            decode_in_threads(model, pending, 4)

    def test_decode_pending_runs_serially_at_one_thread(self):
        """`-t 1` must run no threading machinery at all."""
        model, pending = _StubModel(), _pending(3)
        decode_pending(model, pending, 1)
        self.assertEqual(model.threads, set([threading.current_thread().name]))

    def test_decode_pending_threads_above_one(self):
        model, pending = _StubModel(delay=0.002), _pending(30)
        decode_pending(model, pending, 4)
        self.assertGreater(len(model.threads), 1)

    def test_empty_work_list_is_a_no_op(self):
        model = _StubModel()
        decode_in_threads(model, [], 8)
        self.assertEqual(model.seen, [])

    def test_all_rejected_is_a_no_op(self):
        pending = _pending(3)
        for item in pending:
            item.reject_before_decode = 'x'
        model = _StubModel()
        decode_in_threads(model, pending, 8)
        self.assertEqual(model.seen, [])

    def test_more_threads_than_reads_is_safe(self):
        model, pending = _StubModel(), _pending(2)
        decode_in_threads(model, pending, 16)
        self.assertEqual(len(model.seen), 4)




class _OrientedModel(object):
    """Returns a chosen log-probability and a distinguishable path per sequence."""

    def __init__(self, scores):
        self._scores = scores

    def viterbi(self, sequence):
        return self._scores[sequence], [(0, sequence)]


class TestOnlyTheWinningTracebackIsRetained(unittest.TestCase):
    """Peak memory scaled with locus coverage because BOTH tracebacks were kept.

    Every eligible read held two complete `(int, State)` lists -- ~156 entries each --
    from the moment it finished decoding until `select_illumina_reads` returned. Measured
    at -t 16: 16,678,600 retained entries and 1.46 GB on the 50,619-read BAM.

    Pristine was already O(coverage): it retains the winning vpath of every SELECTED read
    for the life of the return value. What the restructure added was the losing
    orientation, and the rejected reads, on top of that. Dropping the loser removes the
    first of those two, not both -- one traceback per ELIGIBLE read is still more than
    pristine's one per selected read (1,617 against 1,047 on example_7a61).

    The traceback that survives must be the one phase 3 picks (vntr_finder.py:1140-1142),
    whose test is `logp < rev_logp` -- so forward wins a tie.
    """

    def _decoded(self, forward_score, reverse_score):
        pending = PendingRead(sequence='FWD', reverse='REV', query_name='r0')
        model = _OrientedModel({'FWD': forward_score, 'REV': reverse_score})
        decode_serially(model, [pending])
        return pending

    def test_the_losing_reverse_traceback_is_dropped(self):
        pending = self._decoded(forward_score=-1.0, reverse_score=-2.0)

        self.assertEqual(pending.vpath, [(0, 'FWD')])
        self.assertIsNone(pending.rev_vpath)

    def test_the_losing_forward_traceback_is_dropped(self):
        pending = self._decoded(forward_score=-2.0, reverse_score=-1.0)

        self.assertEqual(pending.rev_vpath, [(0, 'REV')])
        self.assertIsNone(pending.vpath)

    def test_a_tie_keeps_the_forward_traceback(self):
        """`logp < rev_logp` is phase 3's test, so equal scores keep forward. Dropping the
        wrong one here silently swaps the sequence recorded for the read."""
        pending = self._decoded(forward_score=-1.0, reverse_score=-1.0)

        self.assertEqual(pending.vpath, [(0, 'FWD')])
        self.assertIsNone(pending.rev_vpath)

    def test_both_log_probabilities_survive_for_phase_three(self):
        """Phase 3 still makes the orientation decision, and makes it from these two."""
        pending = self._decoded(forward_score=-2.0, reverse_score=-1.0)

        self.assertEqual(pending.logp, -2.0)
        self.assertEqual(pending.rev_logp, -1.0)

    def test_the_survivor_is_what_phase_three_would_select(self):
        """Replays vntr_finder.py:1140-1142 verbatim against both orientations."""
        for forward, reverse, expected_sequence in ((-1.0, -2.0, 'FWD'),
                                                    (-2.0, -1.0, 'REV'),
                                                    (-1.0, -1.0, 'FWD')):
            pending = self._decoded(forward, reverse)

            sequence, logp, vpath = pending.sequence, pending.logp, pending.vpath
            if logp < pending.rev_logp:
                sequence, logp, vpath = pending.reverse, pending.rev_logp, pending.rev_vpath

            self.assertEqual(sequence, expected_sequence)
            self.assertEqual(vpath, [(0, expected_sequence)])

    def test_threaded_decoding_drops_the_loser_too(self):
        model, pending = _StubModel(), _pending(4)
        decode_in_threads(model, pending, 3)

        for item in pending:
            retained = [path for path in (item.vpath, item.rev_vpath) if path is not None]
            self.assertEqual(len(retained), 1)


class _PruneAwareModel(object):
    """Records every `viterbi()` call as (sequence, min_threshold) and returns a
    caller-chosen score per (sequence, whether a threshold was passed).

    `min_threshold=None` on the signature mirrors the real `hmm.hmm.Model.viterbi` --
    a stub without it would raise TypeError the moment `_decode_one` passes the kwarg,
    which is the point: this stub only accepts what production accepts.
    """

    def __init__(self, forward_score, pruned_reverse_score, unpruned_reverse_score):
        self.calls = []
        self._forward_score = forward_score
        self._pruned_reverse_score = pruned_reverse_score
        self._unpruned_reverse_score = unpruned_reverse_score

    def viterbi(self, sequence, min_threshold=None):
        self.calls.append((sequence, min_threshold))
        if sequence == 'FWD':
            return self._forward_score, [(0, 'FWD')]
        score = self._unpruned_reverse_score if min_threshold is None else self._pruned_reverse_score
        return score, [(0, 'REV')]


class TestPruneReverseFlag(unittest.TestCase):
    """`_decode_one`'s `prune_reverse` path (Task 8, `--prune-reverse`).

    hmm.pyx's real DP is what actually prunes; these tests are about the CONTROL FLOW
    around it -- that the forward logp becomes the reverse call's floor, that a plain
    `viterbi(sequence)` call is untouched when the flag is off, and that the safety
    valve re-runs unpruned exactly when the pruned result could plausibly matter. The
    real relaxation-count and byte-identity evidence lives in
    tests/test_prune_reverse.py, against the compiled kernel.
    """

    def test_flag_off_never_passes_a_threshold(self):
        """Byte-identical-by-construction: with the flag off, both viterbi() calls are
        exactly what pre-Task-8 `_decode_one` made -- one positional argument, nothing
        else -- so every existing caller (prune_reverse defaulting to False) is
        untouched."""
        model = _PruneAwareModel(forward_score=-10.0, pruned_reverse_score=-50.0,
                                 unpruned_reverse_score=-50.0)
        pending = PendingRead(sequence='FWD', reverse='REV', query_name='r0')

        _decode_one(model, pending, prune_reverse=False)

        self.assertEqual(model.calls, [('FWD', None), ('REV', None)])

    def test_flag_on_floors_the_reverse_call_at_the_forward_logp(self):
        """The reverse decode's threshold is `pending.logp` -- max()'d with the model's
        own dp_score_threshold inside hmm.hmm.Model.viterbi itself (hmm.pyx), not here;
        `_decode_one` only ever needs to pass the forward score."""
        model = _PruneAwareModel(forward_score=-10.0, pruned_reverse_score=-50.0,
                                 unpruned_reverse_score=-999.0)
        pending = PendingRead(sequence='FWD', reverse='REV', query_name='r0')

        _decode_one(model, pending, prune_reverse=True)

        self.assertEqual(model.calls, [('FWD', None), ('REV', -10.0)])
        self.assertEqual(pending.rev_logp, -50.0)

    def test_the_safety_valve_reruns_unpruned_when_the_pruned_score_is_close(self):
        """A pruned result within _SAFETY_VALVE_MARGIN of the forward score might
        actually win or tie -- exactly the case pruning cannot be trusted to get
        byte-identical, so the reverse decode is re-run with no threshold at all, and
        that unpruned call's result is what phase 3 must see."""
        model = _PruneAwareModel(forward_score=-10.0, pruned_reverse_score=-10.0,
                                 unpruned_reverse_score=-9.5)
        pending = PendingRead(sequence='FWD', reverse='REV', query_name='r0')

        _decode_one(model, pending, prune_reverse=True)

        self.assertEqual(model.calls, [('FWD', None), ('REV', -10.0), ('REV', None)])
        self.assertEqual(pending.rev_logp, -9.5)
        self.assertIsNone(pending.vpath)  # reverse (-9.5) beats forward (-10.0)

    def test_the_safety_valve_does_not_fire_on_a_decisive_forward_win(self):
        """The common case (AGENTS.md: reverse wins 77 of 29,998 corpus reads) must NOT
        pay for a third decode."""
        model = _PruneAwareModel(forward_score=-10.0, pruned_reverse_score=-500.0,
                                 unpruned_reverse_score=-500.0)
        pending = PendingRead(sequence='FWD', reverse='REV', query_name='r0')

        _decode_one(model, pending, prune_reverse=True)

        self.assertEqual(len(model.calls), 2)

    def test_the_safety_valve_margin_is_exclusive(self):
        """`pruned_rev_logp > fwd_logp - 1e-6` -- exactly AT the margin does not fire."""
        model = _PruneAwareModel(forward_score=-10.0, pruned_reverse_score=-10.0 - 1e-6,
                                 unpruned_reverse_score=-999.0)
        pending = PendingRead(sequence='FWD', reverse='REV', query_name='r0')

        _decode_one(model, pending, prune_reverse=True)

        self.assertEqual(len(model.calls), 2, 'valve fired exactly at the margin')

        model = _PruneAwareModel(forward_score=-10.0, pruned_reverse_score=-10.0 - 5e-7,
                                 unpruned_reverse_score=-999.0)
        pending = PendingRead(sequence='FWD', reverse='REV', query_name='r0')

        _decode_one(model, pending, prune_reverse=True)

        self.assertEqual(len(model.calls), 3, 'valve did not fire just inside the margin')

    def test_threaded_decoding_honours_the_flag_too(self):
        """`decode_pending`/`decode_in_threads` must forward `prune_reverse` to every
        worker, not just `decode_serially`."""
        pending_reads = [PendingRead(sequence='FWD', reverse='REV', query_name='r%d' % i)
                         for i in range(6)]
        model = _PruneAwareModel(forward_score=-10.0, pruned_reverse_score=-50.0,
                                 unpruned_reverse_score=-999.0)

        decode_pending(model, pending_reads, n_threads=3, prune_reverse=True)

        for call in model.calls:
            if call[0] == 'REV':
                self.assertEqual(call[1], -10.0)
        for pending in pending_reads:
            self.assertEqual(pending.rev_logp, -50.0)


if __name__ == '__main__':
    unittest.main()
