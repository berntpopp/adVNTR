"""Enforce the file-size ratchet.

New files must be under 650 lines. Files already over that when the ratchet was
introduced may only shrink: their current size is the limit, so touching one means
leaving it smaller than you found it.
"""
import subprocess
import sys

NEW_FILE_LIMIT = 650

#: Files already over the limit when the ratchet was introduced. These are ceilings that
#: must come DOWN over time -- lower one when you shrink a file, never raise one.
GRANDFATHERED = {
    'advntr/plot.py': 1445,
    # Task 7 wires the shadow (k, N) counters in: +11 call-site lines, funded inside
    # the file by deleting two commented-out imports, two commented-out code lines,
    # a content-free `# Logging` header, `SelectedRead.is_mapped` (permanently
    # shadowed by the attribute `__init__` assigns) and the never-read local
    # `reason_why_rejected` -- 1233 -> 1232, no live calling code spent.
    # Task 8a collapses the triplicated coverage/expected-rate/`identify_frameshift`
    # block (three near-verbatim copies, 65 lines) into one nested `decide_and_record`
    # and adds the default-off exact caller's two call-site lines: 1232 -> 1212. The
    # three sites' differences -- repeat-unit index, count, and the third site's `ID:`
    # log prefix against the other two's `VID:` -- are parameters, not tidied away.
    # Task 9 wires the rare-unit coverage guard into decide_and_record: 1212 -> 1211.
    'advntr/vntr_finder.py': 1211,
    'hmm/hmm.pyx': 693,
    'advntr/hmm_utils.py': 900,
    # The actual Viterbi DP fill (Task 3 fix round 1): hmm.pyx and hmm_instrumented.pyx
    # each `include` this file, so it is hand-maintained exactly like hmm.pyx is, and
    # was invisible to this ratchet before this entry -- .pxi was not a checked
    # extension, so hmm.pyx (713) + this file (200) = 913 hand-maintained lines went
    # unnoticed against the 882 ceiling that stood before Task 3 fix round 1 (Task 3
    # fix round 2, Finding A). Grandfathered, not on the new-file limit, because it is
    # the direct continuation of hmm.pyx's own DP body, not a new module. Task 4
    # deleted `vpath_table_col` from both files (913 -> 911 combined); ceilings lowered
    # to match (713 -> 712, 200 -> 199). Task 6 (per-thread scratch, an off-GIL
    # traceback, a LUT encoder) touches only hmm.pyx (712 -> 694, funded by deleting
    # Model.log_probability plus condensing inherited docstring prose); fix round 1
    # dropped score-table reuse on measured evidence, moved the ablation narrative
    # into an AGENTS.md Traps entry (which does not count against this ratchet) and
    # left short pointers in the code, landing at 693 (694 -> 709 -> 693 --
    # task-6-report.md); this file is untouched, still 199.
    'hmm/_viterbi_fill_core.pxi': 199,
}

#: Not compiled, not maintained. See FORK.md.
EXCLUDED_PREFIXES = ('pomegranate/',)


def main():
    # Decode explicitly. check_output returns bytes, and on Python 3 comparing those to
    # the str suffixes below raises TypeError, so this ratchet only ever ran under the
    # Python 2 interpreter `make test` uses -- on CI's Python 3 it aborted before
    # checking a single file. Decoding makes it run on both.
    tracked = subprocess.check_output(['git', 'ls-files']).decode('utf-8').split()
    checked = 0
    failures = []
    for path in tracked:
        if not (path.endswith('.py') or path.endswith('.pyx') or path.endswith('.pxi')):
            continue
        if path.startswith(EXCLUDED_PREFIXES):
            continue
        try:
            count = sum(1 for _ in open(path))
        except IOError:
            continue
        checked += 1
        limit = GRANDFATHERED.get(path, NEW_FILE_LIMIT)
        if count > limit:
            kind = 'grandfathered ceiling' if path in GRANDFATHERED else 'new-file limit'
            failures.append('%s: %d lines > %d (%s)' % (path, count, limit, kind))
    if failures:
        sys.stderr.write('LOC ratchet failed:\n  %s\n' % '\n  '.join(failures))
        return 1
    print('loc ratchet: ok (%d files checked)' % checked)
    return 0


if __name__ == '__main__':
    sys.exit(main())
