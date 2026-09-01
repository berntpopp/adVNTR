"""Per-call Viterbi instrumentation for workload characterization. Not shipped;
imported by tests only -- see the Layout note in AGENTS.md for the pattern.

`Model.viterbi`'s `counters=`/`dp_tables=` parameters (hmm/hmm.pyx) are the test-only
surface this wraps. Every production call site still says `model.viterbi(sequence)`
positionally: both default to None, and when they are None the DP checks a NULL C
pointer, not a Python object, so this instrumentation costs the production path nothing
beyond that one pointer compare per pop/edge (measured -- see task-3-report.md).
"""
import numpy as np

#: Order matches what hmm/hmm.pyx `_viterbi_fill` writes into `counters[i]`.
COUNTER_NAMES = ('pops', 'noop_pops', 'edge_relaxations', 'successful_writes')


def decode_with_counters(model, sequence):
    """Decode one sequence and return {counter_name: count} for one `_viterbi_fill` call.

    `pops <= successful_writes + 1`: every push comes from exactly one successful
    relaxation and every push is popped at most once inside `_viterbi_fill` (the +1 is
    the seed push, `start_index` at column 0, written by `viterbi()` before the DP
    runs) -- not exactly once, because AGENTS.md's Traps section notes the main loop
    only drains `col in range(sequence_length)`, so states pushed into the final column
    are written but never popped here. That gap (measured: 3031 states on a 151bp read)
    is unaffected by the pop-time duplicate skip, and so is `noop_pops` -- the skip
    removes the WORK a redundant pop does, not the pop itself, which is why
    `noop_pops` cannot go to zero and `successful_writes` is the number Step 4 asserts
    unchanged.
    """
    counters = np.zeros(len(COUNTER_NAMES), dtype=np.intc)
    model.viterbi(sequence, counters=counters)
    return dict(zip(COUNTER_NAMES, (int(value) for value in counters)))
