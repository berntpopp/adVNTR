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
    'advntr/vntr_finder.py': 1406,
    'hmm/hmm.pyx': 713,
    'advntr/hmm_utils.py': 900,
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
        if not (path.endswith('.py') or path.endswith('.pyx')):
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
