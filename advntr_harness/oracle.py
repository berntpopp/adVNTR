"""Decode-attempt oracle: the unit of comparison for every equivalence gate.

One record per (read, orientation) *before* any selection filter.

This granularity is deliberate. An earlier harness hashed only reads that survived
selection, and only the winning orientation. That hides a broken forward decode whenever
the reverse complement still wins, and compares nothing at all for reads later rejected
by -inf, low quality, or recruitment score. Matching digests from such an oracle prove
"same selected output", which is a useful smoke test and is not bit-exactness.
"""
import hashlib
import struct
from collections import namedtuple

#: One decode attempt.
#:
#: orientation:        'fwd' | 'rev'
#: logp_hex:           the IEEE-754 bits, not repr(). repr() is lossy and its precision
#:                     is platform-dependent; this harness exists to compare bits.
#: vpath_indices:      tuple of state indices along the Viterbi path.
#: exit_status:        'ok' | 'neg_inf' | 'exception:<Type>'
#: selection_decision: 'selected' | 'rejected:<reason>' | 'not_applicable'
DecodeAttempt = namedtuple('DecodeAttempt', [
    'source_file', 'fetch_ordinal', 'query_name', 'orientation', 'sequence',
    'logp_hex', 'vpath_indices', 'exit_status', 'selection_decision',
])

#: Returned instead of a digest when the stream is empty.
#:
#: sha256 of nothing is a perfectly valid-looking constant, so an extraction that
#: silently yields no reads would compare equal to a baseline that also silently
#: yielded none. That is reachable here: 13 files in the corpus (the GRCh37/38 remaps,
#: which use RefSeq accessions like NC_000001.11) extract zero reads.
EMPTY_STREAM_SENTINEL = 'EMPTY'


def logp_to_hex(value):
    """The exact 64 bits of a double, big-endian, as 16 hex characters."""
    return struct.pack('>d', value).encode('hex')


def hex_to_logp(value):
    """Inverse of :func:`logp_to_hex`."""
    return struct.unpack('>d', value.decode('hex'))[0]


def attempt_to_row(attempt):
    """Render one attempt as a single tab-separated line with no trailing newline."""
    return '\t'.join([
        str(attempt.source_file),
        str(attempt.fetch_ordinal),
        str(attempt.query_name),
        str(attempt.orientation),
        str(attempt.sequence),
        str(attempt.logp_hex),
        ','.join(str(int(index)) for index in attempt.vpath_indices),
        str(attempt.exit_status),
        str(attempt.selection_decision),
    ])


def stream_digest(attempts):
    """sha256 over the ordered attempt stream, or EMPTY_STREAM_SENTINEL if empty.

    Order-sensitive on purpose: visit order is semantic in this decoder, because the
    relaxation guard is `> 1e-10` rather than `> 0`, so a reordering can change the
    reported path without changing any score.
    """
    digest = hashlib.sha256()
    seen = False
    for attempt in attempts:
        seen = True
        digest.update(attempt_to_row(attempt))
        digest.update('\n')
    if not seen:
        return EMPTY_STREAM_SENTINEL
    return digest.hexdigest()
