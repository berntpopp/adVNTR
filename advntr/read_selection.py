"""Threaded read decoding for `select_illumina_reads`.

Lives outside vntr_finder.py because that file is on the shrink-only list, and because
the threading concerns are worth reading on their own.

The shape is three phases:

1. **Serial, pysam.** Walk `fetch`, apply every filter, and materialise a plain record
   per read. The pysam handle is never touched again after this.
2. **Parallel, decoder only.** Two `viterbi` calls per read. Safe because `Model.viterbi`
   assigns nothing to `self` -- verified -- so one baked model is shared read-only, and
   every DP buffer is a per-call local. The DP releases the GIL, which is what makes
   threads worth anything here at all.
3. **Serial, in original fetch order.** Apply the rejection rules, emit every log line,
   and build the `SelectedRead` list.

Phase 3 is ordered for two reasons that are easy to miss. `hmm_alignment.generate_aln`
parses an unframed DEBUG stream (`ReadName:` then `Read:` then `VisitedStates:`), so
interleaved output silently mispairs reads with sequences. And tied mutations are sorted
by count alone over a Python 2 dict, so a different arrival order can permute the
reported result. Production emits in read order; so does this.
"""
import logging
import threading

#: Per-read record handed from phase 1 to phases 2 and 3.
#:
#: `reject_before_decode` is a log message for reads the filters dropped. They are kept
#: in the list rather than discarded so phase 3 can emit their messages at the right
#: point in read order, reproducing production's interleaving exactly.
_FIELDS = ('sequence', 'reverse', 'mapq', 'reference_start', 'query_name',
           'is_low_quality', 'reject_before_decode')


class PendingRead(object):
    __slots__ = _FIELDS + ('logp', 'vpath', 'rev_logp', 'rev_vpath')

    def __init__(self, sequence=None, reverse=None, mapq=None, reference_start=None,
                 query_name=None, is_low_quality=False, reject_before_decode=None):
        self.sequence = sequence
        self.reverse = reverse
        self.mapq = mapq
        self.reference_start = reference_start
        self.query_name = query_name
        self.is_low_quality = is_low_quality
        self.reject_before_decode = reject_before_decode
        self.logp = None
        self.vpath = None
        self.rev_logp = None
        self.rev_vpath = None

    @property
    def needs_decoding(self):
        return self.reject_before_decode is None


def _decode_one(model, pending):
    pending.logp, pending.vpath = model.viterbi(pending.sequence)
    pending.rev_logp, pending.rev_vpath = model.viterbi(pending.reverse)


def decode_serially(model, pending_reads):
    for pending in pending_reads:
        if pending.needs_decoding:
            _decode_one(model, pending)


def decode_in_threads(model, pending_reads, n_threads):
    """Decode across `n_threads` worker threads, preserving nothing but correctness.

    Order is irrelevant here -- each worker writes only into its own PendingRead -- so
    a shared cursor is enough and no result marshalling is needed. Phase 3 restores
    order by walking the original list.

    The first worker exception is re-raised in the caller and the batch is discarded;
    a thread that dies silently would otherwise leave `logp` as None and produce a
    confusing failure far from the cause.
    """
    work = [pending for pending in pending_reads if pending.needs_decoding]
    if not work:
        return

    cursor = [0]
    cursor_lock = threading.Lock()
    errors = []

    def worker():
        while True:
            with cursor_lock:
                index = cursor[0]
                cursor[0] += 1
            if index >= len(work):
                return
            try:
                _decode_one(model, work[index])
            except BaseException as exc:  # recorded, re-raised on the calling thread
                errors.append(exc)
                return

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if errors:
        raise errors[0]


def resolve_thread_count(configured):
    """How many workers to use for `configured` (i.e. settings.CORES).

    Snapshotted by the caller before phase 2 so that a concurrent change to the global
    cannot alter the worker count midway.
    """
    try:
        count = int(configured)
    except (TypeError, ValueError):
        return 1
    return count if count > 1 else 1


def decode_pending(model, pending_reads, n_threads):
    """Phase 2. Serial at one thread, so `-t 1` runs no threading machinery at all."""
    if n_threads <= 1:
        decode_serially(model, pending_reads)
    else:
        decode_in_threads(model, pending_reads, n_threads)


def emit(message):
    """Phase-3 logging, kept in one place so the ordering rule is visible."""
    logging.debug(message)
