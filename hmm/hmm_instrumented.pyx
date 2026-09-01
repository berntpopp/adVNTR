
#cython: boundscheck=False
#cython: cdivision=True

"""Test-only Viterbi DP fill: counters, and a runtime skip_enabled toggle.

A SEPARATE compiled extension from hmm.hmm (production), built from the same
_viterbi_fill_core.pxi via `include`, with `DEF INSTRUMENTED = True` instead of False.
hmm/__init__.py imports only `.hmm`, never this module, so nothing here is linked into
the production process at all -- not a guard, not a branch, not a parameter. See
_viterbi_fill_core.pxi's docstring for why that separation exists: a `counters != NULL`
guard directly in front of each increment, always false in production, still measured
4.2-4.5% on the real pristine-vs-final benchmark (Task 3 fix round 1; task-3-report.md).

`decode_instrumented` does not duplicate model CONSTRUCTION. It takes an already-baked
`hmm.hmm.Model` -- built the ordinary way, by the real construction path in
advntr/hmm_utils.py -- and reads its public baked attributes (nbr_indptr/nbr_indices/
nbr_logp/silent/emissions/dp_score_threshold/state_to_index/start/n_states) to drive
this module's own compiled `_viterbi_fill`. Only the DP fill is instrumented; the graph
it runs against is exactly the one production would have built and baked.
"""
import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, realloc, free

cimport cython

DEF INSTRUMENTED = True
include "_viterbi_fill_core.pxi"


def decode_instrumented(model, sequence, skip_enabled=True, dp_tables=None):
    """Run one instrumented Viterbi fill and return its counters.

    :param model: a baked hmm.hmm.Model (or anything exposing the same public baked
        attributes -- see the module docstring).
    :param sequence: the read to decode.
    :param skip_enabled: True (default) reproduces the shipped production behaviour --
        the pop-time skip fires exactly as it does in hmm.hmm. False disables only the
        skip's `continue`; every relaxation decision (which edges are scanned, which
        writes/pushes happen) is otherwise identical, which is what lets Finding 2/3's
        tests compare "skip on" against "skip off" on the SAME compiled fill.
    :param dp_tables: optional dict, filled in place with the raw
        'dynamic_table'/'vpath_row'/'vpath_col'/'silent' arrays this call produced, so
        a caller can check per-cell invariants beyond the single backtracked path.
    :return: an int32 numpy array [pops, noop_pops, edge_relaxations, successful_writes].
    """
    if not model.is_baked:
        raise ValueError('ERROR: model must be baked before instrumented decoding')

    cdef int sequence_length = len(sequence)
    cdef int[::1] encoded_sequence = model.get_encoded_sequence(sequence)

    cdef double[::1, :] dynamic_table = np.full(
        (model.n_states, sequence_length + 1), -np.inf, dtype=np.double, order='F')
    cdef int start_index = model.state_to_index[model.start]
    # log(1) == 0.0 exactly under IEEE-754 (a mandated special case, not a
    # not-correctly-rounded general log call -- see hmm.pyx's bake() comment on why
    # that distinction matters elsewhere), so this needs no libm import here.
    dynamic_table[start_index, 0] = 0.0

    cdef int[::1, :] vpath_table_row = np.zeros(
        (model.n_states, sequence_length + 1), dtype=np.intc, order='F')
    cdef int[::1, :] vpath_table_col = np.zeros(
        (model.n_states, sequence_length + 1), dtype=np.intc, order='F')

    cdef int[::1] indptr = model.nbr_indptr
    cdef int[::1] indices = model.nbr_indices
    cdef unsigned char[::1] silent = model.silent
    cdef double[:, ::1] emissions = model.emissions
    cdef double[::1] weights = model.nbr_logp
    cdef double threshold = model.dp_score_threshold

    counters = np.zeros(4, dtype=np.intc)
    cdef int[::1] counters_view = counters
    cdef bint skip_flag = 1 if skip_enabled else 0

    cdef int fill_status = 0
    with nogil:
        fill_status = _viterbi_fill(encoded_sequence, dynamic_table, vpath_table_row,
                      vpath_table_col, indptr, indices, silent, emissions,
                      weights, threshold, sequence_length,
                      start_index, &counters_view[0], skip_flag)
    if fill_status != 0:
        raise MemoryError('Viterbi work queue could not be grown')

    if dp_tables is not None:
        dp_tables['dynamic_table'] = np.asarray(dynamic_table)
        dp_tables['vpath_row'] = np.asarray(vpath_table_row)
        dp_tables['vpath_col'] = np.asarray(vpath_table_col)
        dp_tables['silent'] = np.asarray(silent)

    return counters
