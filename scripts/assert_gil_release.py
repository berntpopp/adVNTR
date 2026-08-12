"""Assert the generated C emits the Viterbi DP call inside a GIL-released region.

That is what this proves, and the title says so deliberately: it reads `hmm.c`, not the
compiled `.so`. A build without `WITH_THREAD` strips the macros and would pass while
holding the GIL, and nothing in this suite measures wall-clock or scaling. The property
is "Cython was told to release it and has not silently stopped", not "the process ran
concurrently".

The check this replaces counted `Py_UNBLOCK_THREADS` anywhere in the generated `hmm.c`
and passed on one or more. That was true but unscoped: `hmm.pyx` happens to contain
exactly one `with nogil:` today, so the single match was the right one and the check was
not yet vacuous.

It becomes vacuous the moment a second one lands. `subseq_viterbi` and
`_update_tables_for_subseq` are still the old dict-and-object path reading
`self.transition_matrix`, and putting them under `nogil` is the obvious next optimisation.
After that the file-wide count stays >= 1 forever, and the main DP could go back to
running under the GIL with the gate green.

Nothing else would catch that. `tests/test_read_selection.py` drives a pure-Python stub
model, so its threading test passes with the GIL held; the Tier 3 thread-invariance test
compares outputs, which are identical either way; and no test in the suite asserts
wall-clock or scaling. The threading numbers would simply become fiction.

So the property checked here is not "a GIL release exists somewhere" but the one that
matters: **every call to `_viterbi_fill` sits between a GIL release and its matching
re-acquire.** The release is emitted in the caller -- `viterbi()`, at `hmm.pyx:925` -- and
not inside `_viterbi_fill` itself, so a check scoped to the callee's body would report a
DP that holds the GIL even when it does not.
"""
import sys

#: The C symbol Cython emits for `_viterbi_fill`, with its opening parenthesis so a
#: forward declaration or a comment mentioning the name is not read as a call.
DP_CALL = '__pyx_f_3hmm_3hmm__viterbi_fill('

#: What `with nogil:` expands to around the call. Cython emits the macro pair; the
#: underlying calls are accepted because a different Cython release may emit them directly.
RELEASE_TOKENS = ('Py_UNBLOCK_THREADS', 'PyEval_SaveThread')
REACQUIRE_TOKENS = ('Py_BLOCK_THREADS', 'PyEval_RestoreThread')


def _first_index(source, tokens, start):
    """Return the earliest offset at or after `start` of any of `tokens`, or -1.

    Args:
        source: The C source.
        tokens: Substrings to look for.
        start: Where to start looking.

    Returns:
        int: The earliest offset, or -1 if none appear.
    """
    found = [source.find(token, start) for token in tokens]
    found = [offset for offset in found if offset != -1]
    return min(found) if found else -1


def gil_released_spans(source):
    """Return the (start, end) offsets of each GIL-released region.

    A region runs from a release token to the FIRST re-acquire after it, which is the
    smallest defensible span: anything after that token is running with the GIL back.

    Args:
        source: The generated C.

    Returns:
        list: The spans, in order.
    """
    spans = []
    cursor = 0
    while True:
        release = _first_index(source, RELEASE_TOKENS, cursor)
        if release == -1:
            return spans
        reacquire = _first_index(source, REACQUIRE_TOKENS, release + 1)
        if reacquire == -1:
            # A release with no matching re-acquire is not a region this can vouch for.
            return spans
        spans.append((release, reacquire))
        cursor = reacquire + 1


def dp_call_offsets(source):
    """Return the offset of every *call* to the DP.

    Cython emits the symbol three times: a forward declaration, the definition, and the one
    call. The first two open with a storage class at the start of their line, and neither
    runs anything -- counting them would report two permanently unguarded "calls" and make
    this check fail on a correct build.

    The prefix examined is the enclosing *statement*, not the line, and the test is that it
    *begins* with the storage class rather than containing it anywhere. Both matter:
    `static double __pyx_v_x; fill_status = _viterbi_fill(...)` is one line holding a
    declaration and a genuine call, and either a line-scoped prefix or a substring test
    drops the call. A dropped call that leaves one guarded call behind makes this whole
    check exit 0 -- a false pass, which is the only direction that costs anything here.

    Args:
        source: The generated C.

    Returns:
        list: The offsets, in order.
    """
    offsets = []
    cursor = source.find(DP_CALL)
    while cursor != -1:
        statement_start = max(source.rfind(boundary, 0, cursor) for boundary in ('\n', ';', '{', '}')) + 1
        if not source[statement_start:cursor].lstrip().startswith('static'):
            offsets.append(cursor)
        cursor = source.find(DP_CALL, cursor + 1)
    return offsets


def unguarded_dp_calls(source):
    """Return the offsets of DP calls that are NOT inside a GIL-released region.

    Args:
        source: The generated C.

    Returns:
        list: The offending offsets. Empty means every call runs without the GIL.

    Raises:
        ValueError: If the DP is never called, which means this check is reading a file
            it does not understand rather than a passing one.
    """
    calls = dp_call_offsets(source)
    if not calls:
        raise ValueError('no call to %s found; the DP was renamed or is gone' % DP_CALL)
    spans = gil_released_spans(source)
    return [offset for offset in calls
            if not any(start < offset < end for start, end in spans)]


def main(argv=None):
    """Check one generated C file.

    Args:
        argv: Arguments without the program name; defaults to `sys.argv[1:]`.

    Returns:
        int: 0 if every DP call runs without the GIL, 1 otherwise.
    """
    argv = sys.argv[1:] if argv is None else argv
    path = argv[0] if argv else 'hmm/hmm.c'

    with open(path) as handle:
        source = handle.read()

    try:
        unguarded = unguarded_dp_calls(source)
    except ValueError as error:
        sys.stderr.write('%s: %s\n' % (path, error))
        return 1

    calls = len(dp_call_offsets(source))
    sys.stderr.write('%s: %d call(s) to the DP, %d outside a GIL-released region\n'
                     % (path, calls, len(unguarded)))
    if unguarded:
        sys.stderr.write('the Viterbi DP runs while holding the GIL; every threading '
                         'number this fork reports would be fiction\n')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
