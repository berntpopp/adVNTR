@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _viterbi_fill(int[::1] encoded_sequence,
                        double[::1,:] dynamic_table,
                        int[::1,:] vpath_table_row,
                        int[::1,:] vpath_table_col,
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
    independent extension modules rather than one module with a runtime guard. That
    distinction is load-bearing, not stylistic -- a `counters != NULL` check directly in
    front of each `counters[i] += 1`, always false in production, still measured
    4.2-4.5% on the real pristine-vs-final benchmark (Task 3 fix round 1;
    task-3-report.md), because a per-event branch inside a nogil loop this hot never
    gets close to the ~1% budget, no matter how the branch is written. `INSTRUMENTED =
    False` (hmm.hmm, production, hmm/__init__.py's only import) deletes every counting
    site and the `skip_enabled` gate at COMPILE time -- not <1%, not in the generated C
    at all -- and the pop-time skip is unconditional, exactly as originally shipped.
    `INSTRUMENTED = True` (hmm.hmm_instrumented, test-only, never imported by production)
    compiles the counters in and adds the `skip_enabled` runtime flag so a test can run
    the identical fill with the skip forced off (see decode_instrumented in
    hmm_instrumented.pyx) -- a runtime branch is free to add there, because production
    never links this module in at all.

    `counters`, when not NULL, is a caller-allocated int[4]:
    [pops, noop_pops, edge_relaxations, successful_writes]. `noop_pops` counts pops that
    REACHED the edge loop despite reading an unchanged cell -- the wasted work the skip
    eliminates -- not every pop matching that condition. Under `skip_enabled=True` that
    is exactly 0 by construction: such a pop `continue`s before the counting site is
    ever reached, it is not skipped by a runtime check on the counter. Under
    `skip_enabled=False` every one of those pops falls through into the counting site
    instead, reproducing the 41.8%/46.6% measured with the skip's own condition
    evaluated but not acted on.
    """
    cdef int row, col, k, ch, neighbor_state_index, next_col, start, end, n_states
    cdef double log_prob, emission, current
    IF INSTRUMENTED:
        cdef bint is_noop

    # Two array-backed FIFOs replacing the linked-list Queue, whose queue_push_tail
    # malloc'd a 24-byte node per relaxation (~1.5M malloc/free pairs per call).
    # Push at tail, pop at head, never wrap: byte-for-byte the same visit ORDER the
    # linked list produced. Order is load-bearing -- the relaxation guard is
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
    # but its cell is only read at POP time -- so a row popped twice at the same
    # column with an unchanged value is redoing work already done. `last_col` inits
    # to -1, and a real column is never negative, so no separate "seen" flag is needed.
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

        if cur_head == cur_tail:
            break

        ch = encoded_sequence[col]
        next_col = col + 1

        while cur_head < cur_tail:
            row = cur[cur_head]
            cur_head += 1
            IF INSTRUMENTED:
                if counters != NULL:
                    counters[0] += 1
            start = indptr[row]
            end = indptr[row + 1]
            current = dynamic_table[row, col]

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

                    if log_prob - dynamic_table[neighbor_state_index, col] > 1e-10 and log_prob >= threshold:
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
                        dynamic_table[neighbor_state_index, col] = log_prob
                        vpath_table_row[neighbor_state_index, col] = row
                        vpath_table_col[neighbor_state_index, col] = col
            else:  # Emitting state: consume a character and advance a column
                emission = emissions[row, ch]
                for k in range(start, end):
                    neighbor_state_index = indices[k]
                    IF INSTRUMENTED:
                        if counters != NULL:
                            counters[2] += 1
                    log_prob = current + weights[k] + emission

                    if log_prob - dynamic_table[neighbor_state_index, next_col] > 1e-10 and log_prob >= threshold:
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
                        dynamic_table[neighbor_state_index, next_col] = log_prob
                        vpath_table_row[neighbor_state_index, next_col] = row
                        vpath_table_col[neighbor_state_index, next_col] = col

    free(cur)
    free(nxt)
    free(last_val)
    free(last_col)
    return 0
