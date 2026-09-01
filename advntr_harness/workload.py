"""Per-call Viterbi instrumentation for workload characterization. Not shipped;
imported by tests only -- see the Layout note in AGENTS.md for the pattern.

Wraps `hmm.hmm_instrumented.decode_instrumented` (hmm/hmm_instrumented.pyx), a SEPARATE
compiled extension from the production `hmm.hmm` module, built from the same
`hmm/_viterbi_fill_core.pxi` with a different compile-time `DEF` (Task 3 fix round 1;
task-3-report.md). Production's `Model.viterbi()` takes no counters/dp_tables/
skip_enabled arguments at all -- that instrumentation does not exist in the compiled
production module, not merely disabled by a default, because a runtime guard on it
still measured 4.2-4.5% even when always false.

Loads `hmm_instrumented.so` directly with `imp.load_dynamic` rather than
`import hmm.hmm_instrumented`: `hmm/__init__.py` calls `pyximport.install()` (AGENTS.md
Traps) right after its own `from .hmm import *`, and any LATER dotted import of a
submodule that `__init__.py` never touches itself -- which `hmm_instrumented` is, by
design, since production must never import it -- gets intercepted by that installed
pyximport hook and recompiled on the fly into `~/.pyxbld` with pyximport's own Cython
directives, not `build_config.py`'s, instead of loading the artifact `make build`
already produced. Loading the .so directly sidesteps the package import machinery (and
therefore pyximport) entirely, so every test here exercises exactly what `make build`/
`make gate` built.
"""
import imp
import os

_HMM_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'hmm')
_SO_PATH = os.path.join(_HMM_DIR, 'hmm_instrumented.so')

#: Order matches what hmm/_viterbi_fill_core.pxi writes into `counters[i]`.
COUNTER_NAMES = ('pops', 'noop_pops', 'edge_relaxations', 'successful_writes')


def _load_instrumented():
    if not os.path.isfile(_SO_PATH):
        raise ImportError(
            'hmm/hmm_instrumented.so not built; run `make build` first (%s missing)'
            % _SO_PATH)
    return imp.load_dynamic('hmm_instrumented', _SO_PATH)


_instrumented = _load_instrumented()
decode_instrumented = _instrumented.decode_instrumented


def decode_with_counters(model, sequence, skip_enabled=True):
    """Decode one sequence against the INSTRUMENTED build; return {counter_name: count}.

    `skip_enabled=True` (default) reproduces the shipped production behaviour --
    `pops`/`successful_writes` are identical to a production `model.viterbi(sequence)`
    call on the same read. `skip_enabled=False` disables only the pop-time skip's
    `continue`, leaving every other relaxation decision identical, so a caller can
    directly compare "skip on" against "skip off" on the SAME compiled fill (see
    hmm/hmm_instrumented.pyx and hmm/_viterbi_fill_core.pxi).

    `pops <= successful_writes + 1`: every push comes from exactly one successful
    relaxation and every push is popped at most once inside `_viterbi_fill` (the +1 is
    the seed push, `start_index` at column 0, written before the DP runs) -- not
    exactly once, because AGENTS.md's Traps section notes the main loop only drains
    `col in range(sequence_length)`, so states pushed into the final column are written
    but never popped here. That gap (measured: 3031 states on a 151bp read) holds
    whether or not the skip is enabled.

    Under `skip_enabled=True`, `noop_pops` is exactly 0 by construction: it counts pops
    that reached the edge loop despite an unchanged cell, and the skip's `continue`
    prevents that from ever happening -- it is not a runtime check on the counter that
    happens to always be false. Under `skip_enabled=False` it reproduces the
    41.8%/46.6% measured with the skip's own condition evaluated but never acted on.
    `successful_writes` (and `pops`) do not depend on `skip_enabled` at all: that
    identity is the empirical signature that the skip changes only which pops do
    wasted work, never what gets written or pushed.
    """
    counters = decode_instrumented(model, sequence, skip_enabled=skip_enabled)
    return dict(zip(COUNTER_NAMES, (int(value) for value in counters)))
