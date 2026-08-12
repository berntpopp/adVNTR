"""The threaded decode helper.

Uses a stub model so the decoding contract is tested without a 3-minute HMM build. The
real end-to-end equivalence lives in tests/test_tier3_occurrence.py.
"""
import threading
import unittest

from advntr.read_selection import (PendingRead, decode_in_threads, decode_pending,
                                   decode_serially, resolve_thread_count)


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


if __name__ == '__main__':
    unittest.main()
