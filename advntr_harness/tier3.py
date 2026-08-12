"""Occurrence-level oracle: runs the real `select_illumina_reads`.

Tiers 1 and 2 compare *unique sequences*, which is the right unit for the decoder and
the wrong unit for anything that changes the read loop. Threading `select_illumina_reads`
risks losing or reordering duplicate occurrences, mis-associating mapq/query_name, and
interleaving the DEBUG log that `hmm_alignment.generate_aln` parses without framing --
none of which a unique-sequence stream can see.

So this tier calls the production function itself and digests what it returned, in order.
"""
import hashlib

from advntr import settings
from advntr_harness.oracle import EMPTY_STREAM_SENTINEL, logp_to_hex


def selected_read_row(index, read):
    """One tab-separated row per SelectedRead, in the order the function returned them."""
    path = ','.join(str(int(state_index)) for state_index, _state in read.vpath) \
        if read.vpath else ''
    return '\t'.join([
        str(index),
        str(read.query_name),
        str(read.sequence),
        logp_to_hex(read.logp),
        str(read.mapq),
        str(bool(read.is_mapped)),
        path,
    ])


def selection_digest(selected_reads):
    """sha256 over the ordered selected-read stream, or a sentinel when empty.

    Order-sensitive deliberately: `generate_aln` consumes an unframed log stream whose
    correctness depends on read order, and tied mutations are sorted by count alone.
    """
    digest = hashlib.sha256()
    seen = False
    for index, read in enumerate(selected_reads):
        seen = True
        digest.update(selected_read_row(index, read))
        digest.update('\n')
    if not seen:
        return EMPTY_STREAM_SENTINEL
    return digest.hexdigest()


def run_selection(finder, alignment_path, threads=None):
    """Run the production selection path, optionally at a given thread count.

    `threads` sets `settings.CORES`, which is what `--threads` writes. Before the
    threading work it is inert on this path; afterwards it is the control. Snapshotted
    and restored so a test cannot leak it into the next one.
    """
    previous = settings.CORES
    if threads is not None:
        settings.CORES = threads
    try:
        return finder.select_illumina_reads(alignment_path, [])
    finally:
        settings.CORES = previous


def selection_evidence(finder, alignment_path, threads=None):
    """Everything a threading change could plausibly break, in one comparable dict."""
    selected = run_selection(finder, alignment_path, threads=threads)
    return {
        'count': len(selected),
        'digest': selection_digest(selected),
        'query_names': [read.query_name for read in selected],
        'model_states': int(finder.hmm.n_states) if finder.hmm is not None else None,
    }
