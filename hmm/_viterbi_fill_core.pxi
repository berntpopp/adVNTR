@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _viterbi_fill(int[::1] encoded_sequence,
                        double[::1,:] dynamic_table,
                        int[::1,:] vpath_table_row,
                        int[::1] indptr,
                        int[::1] indices,
                        unsigned char[::1] silent,
                        double[:, ::1] emissions,
                        double[::1] weights,
                        double threshold,
                        int sequence_length,
                        int start_index,
                        int* counters,
                        bint skip_enabled) nogil:
    """Fill the Viterbi DP table. Pure C -- runs with the GIL released.

    This is the body that used to live inline in Model.viterbi plus
    Model.__update_dynamic_table. It touches no Python object: the graph arrives as
    CSR (`indptr`/`indices`), emissions as a flat (state, base) table, and silence as
    a byte flag, all built once in bake(). Relaxation order is unchanged -- neighbours
    are still visited in the ascending order bake() sorted them into -- so the
    resulting logp and path are bit-identical to the pre-nogil version.

    ONE hand-maintained source, compiled TWICE: hmm/hmm.pyx and hmm/hmm_instrumented.pyx
    each `include` this file after setting a different `DEF INSTRUMENTED`, producing two
    independent extension modules rather than one module with a runtime guard -- a
    `counters != NULL` check in front of each `counters[i] += 1`, always false in
    production, still measured 4.2-4.5% on the pristine-vs-final benchmark (Task 3 fix
    round 1; task-3-report.md), over the ~1% budget no matter how the branch is written.
    `INSTRUMENTED = False` (hmm.hmm, production, hmm/__init__.py's only import) deletes
    every counting site and the `skip_enabled` gate at COMPILE time; `INSTRUMENTED = True`
    (hmm.hmm_instrumented, test-only, never imported by production) compiles them in, plus
    a runtime `skip_enabled` flag so a test can run the identical fill with the skip
    forced off (decode_instrumented in hmm_instrumented.pyx).

    `counters`, when not NULL, is a caller-allocated int[4]:
    [pops, noop_pops, edge_relaxations, successful_writes]. `noop_pops` counts pops that
    REACHED the edge loop despite reading an unchanged cell -- the wasted work the skip
    eliminates -- not every pop matching that condition. Under `skip_enabled=True` that
    is exactly 0 by construction (the pop `continue`s before the counting site), and
    under `skip_enabled=False` every one of those pops falls through into the counting
    site instead, reproducing the 41.8%/46.6% measured with the skip's own condition
    evaluated but not acted on.

    No `vpath_table_col`: predecessor column of `vpath_table_row[r,c]` is `c if silent[row] else c-1` (task-4-report.md: 0 violations/1,041,573 cells).

    Score scratch (production only) is TWO rolling columns addressed by `col & 1`/`next_col & 1`, reset to -inf EVERY iteration -- even one about to `break` -- because the un-enqueued final relaxation (hmm.pyx) reads column `sequence_length` by name and must see -inf, not 2-columns-stale data, after an early break (task-5-report.md). `vpath_table_row` stays full-size and unreset: the traceback walks it, and every cell it reads was written by the relaxation that produced that path.
    """
    cdef int row, col, k, ch, neighbor_state_index, next_col, start, end, n_states, col_idx, next_col_idx
    cdef double log_prob, emission, current
    IF INSTRUMENTED:
        cdef bint is_noop

    # Two array-backed FIFOs replacing the linked-list Queue's 24-byte malloc/free per
    # relaxation (~1.5M pairs/call). Push at tail, pop at head, never wrap: same visit
    # ORDER the linked list produced -- load-bearing, since the relaxation guard is
    # `> 1e-10`, not `> 0`, so this is not an order-independent fixpoint and a LIFO
    # would silently change vpath.
    cdef int cur_cap = 4096, nxt_cap = 4096
    cdef int cur_head = 0, cur_tail = 0, nxt_head = 0, nxt_tail = 0
    cdef int* cur = <int*> malloc(cur_cap * sizeof(int))
    cdef int* nxt = <int*> malloc(nxt_cap * sizeof(int))
    cdef int* swap_buf
    cdef int* grown
    cdef int swap_int

    if cur == NULL or nxt == NULL:
        free(cur)
        free(nxt)
        return -1

    # Per-call scratch for the pop-time duplicate skip below, sized n_states. A state
    # can be pushed more than once per column (once per improving relaxation into it),
    # but its cell is only read at POP time, so a row popped twice with an unchanged
    # value redoes work already done. `last_col` inits to -1 (a real column never is).
    n_states = dynamic_table.shape[0]
    cdef double* last_val = <double*> malloc(n_states * sizeof(double))
    cdef int* last_col = <int*> malloc(n_states * sizeof(int))
    if last_val == NULL or last_col == NULL:
        free(cur)
        free(nxt)
        free(last_val)
        free(last_col)
        return -1
    for row in range(n_states):
        last_col[row] = -1

    nxt[0] = start_index
    nxt_tail = 1

    for col in range(sequence_length):
        swap_buf = cur; cur = nxt; nxt = swap_buf
        swap_int = cur_cap; cur_cap = nxt_cap; nxt_cap = swap_int
        cur_head = 0
        cur_tail = nxt_tail
        nxt_head = 0
        nxt_tail = 0

        next_col = col + 1
        IF INSTRUMENTED:
            col_idx, next_col_idx = col, next_col
        ELSE:
            col_idx, next_col_idx = col & 1, next_col & 1
            for row in range(n_states): dynamic_table[row, next_col_idx] = -INFINITY

        if cur_head == cur_tail:
            break

        ch = encoded_sequence[col]

        while cur_head < cur_tail:
            row = cur[cur_head]
            cur_head += 1
            IF INSTRUMENTED:
                if counters != NULL:
                    counters[0] += 1
            start = indptr[row]
            end = indptr[row + 1]
            current = dynamic_table[row, col_idx]

            IF INSTRUMENTED:
                # last_col[row]/last_val[row] update either way -- see the module
                # docstring: that write is idempotent when is_noop is True (it writes
                # back the exact values that made is_noop True), so ordering it before
                # the branch changes nothing about which future pop is a no-op.
                is_noop = last_col[row] == col and last_val[row] == current
                last_col[row] = col
                last_val[row] = current
                if skip_enabled and is_noop:
                    continue
                if is_noop:
                    if counters != NULL:
                        counters[1] += 1
            ELSE:
                if last_col[row] == col and last_val[row] == current:
                    continue
                last_col[row] = col
                last_val[row] = current

            if silent[row]:  # Silent state: stay in the same column
                for k in range(start, end):
                    neighbor_state_index = indices[k]
                    IF INSTRUMENTED:
                        if counters != NULL:
                            counters[2] += 1
                    log_prob = current + weights[k]

                    if log_prob - dynamic_table[neighbor_state_index, col_idx] > 1e-10 and log_prob >= threshold:
                        IF INSTRUMENTED:
                            if counters != NULL:
                                counters[3] += 1
                        if cur_tail == cur_cap:
                            grown = <int*> realloc(cur, 2 * cur_cap * sizeof(int))
                            if grown == NULL:
                                free(cur)
                                free(nxt)
                                free(last_val)
                                free(last_col)
                                return -1
                            cur = grown
                            cur_cap *= 2
                        cur[cur_tail] = neighbor_state_index
                        cur_tail += 1
                        dynamic_table[neighbor_state_index, col_idx] = log_prob
                        vpath_table_row[neighbor_state_index, col] = row
            else:  # Emitting state: consume a character and advance a column
                emission = emissions[row, ch]
                for k in range(start, end):
                    neighbor_state_index = indices[k]
                    IF INSTRUMENTED:
                        if counters != NULL:
                            counters[2] += 1
                    log_prob = current + weights[k] + emission

                    if log_prob - dynamic_table[neighbor_state_index, next_col_idx] > 1e-10 and log_prob >= threshold:
                        IF INSTRUMENTED:
                            if counters != NULL:
                                counters[3] += 1
                        if nxt_tail == nxt_cap:
                            grown = <int*> realloc(nxt, 2 * nxt_cap * sizeof(int))
                            if grown == NULL:
                                free(cur)
                                free(nxt)
                                free(last_val)
                                free(last_col)
                                return -1
                            nxt = grown
                            nxt_cap *= 2
                        nxt[nxt_tail] = neighbor_state_index
                        nxt_tail += 1
                        dynamic_table[neighbor_state_index, next_col_idx] = log_prob
                        vpath_table_row[neighbor_state_index, next_col] = row

    free(cur)
    free(nxt)
    free(last_val)
    free(last_col)
    return 0
