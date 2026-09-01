"""Threaded read decoding for `select_illumina_reads`.

Lives outside vntr_finder.py because that file is on the shrink-only list, and because
the threading concerns are worth reading on their own.

The shape is three phases:

1. **Serial, pysam.** Walk `fetch`, apply every filter, and materialise a plain record
   per read. The pysam handle is never touched again after this.
2. **Parallel, decoder only.** Two `viterbi` calls per read. Safe because `Model.viterbi`
   assigns nothing to `self` -- verified -- so one baked model is shared read-only, and
   every DP buffer is a per-call local. The DP releases the GIL, which is what makes
   threads worth anything here at all. Task 8's `--prune-reverse` keeps this true: the
   tighter DP threshold it uses for the reverse call is an ordinary per-call argument to
   `viterbi`, never a write to the shared model.
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

#: What phase 1 decided about a read, before any decoding.
#:
#: Every read the fetch loop saw gets a record, including the ones it dropped, so phase 3
#: can emit their log lines at the right point in read order. Production logs during
#: iteration, i.e. in read order; replaying in read order reproduces that exactly.
DUPLICATE = 'duplicate'      # unmapped or flagged duplicate -- logged, contributes no bp
TOO_SHORT = 'too_short'      # below MIN_READ_LENGTH -- logged, contributes no bp
OUT_OF_SPAN = 'out_of_span'  # fails the positional test -- silent, contributes no bp
HAS_N = 'has_n'              # spans but contains N -- silent, DOES contribute bp
DECODE = 'decode'            # the only outcome that reaches the decoder


class PendingRead(object):
    """One read the fetch loop saw.

    `bp` is the read's contribution to `vntr_bp_in_mapped_reads`. It is carried rather
    than recomputed because the original accumulates it *after* the decode-time
    `continue` statements, so a read rejected for low likelihood or low quality
    contributes nothing while a read containing N contributes normally. That asymmetry
    is easy to lose in a restructure and only shows up in a log line.
    """

    __slots__ = ('outcome', 'log_message', 'bp', 'sequence', 'reverse', 'mapq',
                 'reference_start', 'query_name', 'is_low_quality',
                 'logp', 'vpath', 'rev_logp', 'rev_vpath')

    def __init__(self, outcome=DECODE, log_message=None, bp=0, sequence=None,
                 reverse=None, mapq=None, reference_start=None, query_name=None,
                 is_low_quality=False):
        self.outcome = outcome
        self.log_message = log_message
        self.bp = bp
        self.sequence = sequence
        self.reverse = reverse
        self.mapq = mapq
        self.reference_start = reference_start
        self.query_name = query_name
        self.is_low_quality = is_low_quality
        self.logp = None
        self.vpath = None
        self.rev_logp = None
        self.rev_vpath = None

    @property
    def needs_decoding(self):
        return self.outcome == DECODE

    # Retained so older callers and tests that set `reject_before_decode` keep working.
    def _get_reject(self):
        return None if self.outcome == DECODE else self.outcome

    def _set_reject(self, value):
        self.outcome = DECODE if value is None else TOO_SHORT

    reject_before_decode = property(_get_reject, _set_reject)


#: Task 8's safety valve. `_viterbi_fill_core.pxi`'s threshold check is non-strict
#: (`log_prob >= threshold`), so given non-increasing path scores every prefix of a
#: surviving path also clears it -- the pruned reverse decode is provably bit-exact
#: whenever the true reverse score is >= max(dp_score_threshold, fwd_logp), ties
#: included. This valve is therefore defence-in-depth against a future change to that
#: comparator, not load-bearing for correctness today. It re-runs the reverse decode
#: unpruned whenever the pruned result comes within this margin of `fwd_logp`, at
#: negligible cost: measured to fire on ~0.2% of attempts in this fork's own
#: public-corpus measurement (see task-8-report.md).
_SAFETY_VALVE_MARGIN = 1e-6


def _decode_one(model, pending, prune_reverse=False):
    """Decode both orientations, keeping only the traceback that wins.

    Both log-probabilities are kept: phase 3 still makes the orientation decision, and it
    makes it from those two floats alone. What is dropped is the LOSING traceback, at the
    first moment the comparison that settles it is possible.

    Retaining both is what made peak memory scale with locus coverage. Every eligible read
    held two complete `(int, State)` lists of ~156 entries from the moment it finished
    decoding until `select_illumina_reads` returned: measured at `-t 16`, 16,678,600
    retained entries and 1.46 GB on the 50,619-read BAM, against 442 MB for pristine on the
    same input.

    Pristine was not O(1) either -- it retains the winning vpath of every SELECTED read for
    the life of the return value, which `find_frameshift_from_selected_reads` and
    `iteratively_update_model` then consume. This does NOT restore that profile, and saying
    so would be the same overclaim `e89bfa7` removed from the Tier 3 manifest: what is
    retained here is one traceback per ELIGIBLE read, and eligible exceeds selected because
    a read rejected in phase 3 still holds its path until the loop ends. On
    example_7a61 that is 1,617 eligible against 1,047 selected -- 570 tracebacks pristine
    had already freed. Two per eligible read has become one; one per selected read is still
    below us.

    The test below is exactly phase 3's (`vntr_finder.py:1141`), so the survivor is always
    the traceback phase 3 goes on to select -- including the tie, where `<` keeps forward.

    `prune_reverse` (Task 8, default-off `--prune-reverse`): if set, the reverse decode
    runs with `min_threshold=pending.logp` -- `hmm.hmm.Model.viterbi`'s per-call floor,
    OR'd with the model's own `dp_score_threshold` via max(). Sound because every DP edge
    weight is <= 0 (they are log probabilities), so a path's running score is
    non-increasing column by column: if the true best reverse path's final score beats
    `pending.logp`, every prefix of that path also beats it, so raising the threshold to
    `pending.logp` can only prune paths that could never have won anyway. Because the DP's
    own threshold check is non-strict (`_viterbi_fill_core.pxi`), that argument alone
    already makes the pruned result bit-exact, ties included -- `_SAFETY_VALVE_MARGIN` is
    defence-in-depth on top of that, re-running the reverse decode unpruned whenever the
    pruned result comes within the margin of `pending.logp`, at negligible cost.
    """
    pending.logp, pending.vpath = model.viterbi(pending.sequence)
    if prune_reverse:
        pending.rev_logp, pending.rev_vpath = model.viterbi(
            pending.reverse, min_threshold=pending.logp)
        if pending.rev_logp > pending.logp - _SAFETY_VALVE_MARGIN:
            pending.rev_logp, pending.rev_vpath = model.viterbi(pending.reverse)
    else:
        pending.rev_logp, pending.rev_vpath = model.viterbi(pending.reverse)
    if pending.logp < pending.rev_logp:
        pending.vpath = None
    else:
        pending.rev_vpath = None


def decode_serially(model, pending_reads, prune_reverse=False):
    for pending in pending_reads:
        if pending.needs_decoding:
            _decode_one(model, pending, prune_reverse)


def decode_in_threads(model, pending_reads, n_threads, prune_reverse=False):
    """Decode across `n_threads` worker threads, preserving nothing but correctness.

    Order is irrelevant here -- each worker writes only into its own PendingRead -- so
    a shared cursor is enough and no result marshalling is needed. Phase 3 restores
    order by walking the original list.

    The first worker exception is re-raised in the caller and the batch is discarded;
    a thread that dies silently would otherwise leave `logp` as None and produce a
    confusing failure far from the cause.

    `prune_reverse` is a plain bool, snapshotted by the caller (see `decode_pending`) --
    every worker reads the same immutable value, so passing it here adds no shared
    mutable state beyond what `n_threads` itself already required.
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
                _decode_one(model, work[index], prune_reverse)
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


def decode_pending(model, pending_reads, n_threads, prune_reverse=False):
    """Phase 2. Serial at one thread, so `-t 1` runs no threading machinery at all."""
    if n_threads <= 1:
        decode_serially(model, pending_reads, prune_reverse)
    else:
        decode_in_threads(model, pending_reads, n_threads, prune_reverse)


def emit(message):
    """Phase-3 logging, kept in one place so the ordering rule is visible."""
    logging.debug(message)
