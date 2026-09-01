"""Subprocess worker for the early-break regression in test_decoder_workload.py.

Runs the PRODUCTION `Model.viterbi` call OUTSIDE the test process, under an
external `timeout`, because the failure mode it guards against is a hang: a
naive rolled-score-table implementation can leave `logp` spuriously finite,
which sends `Model.viterbi`'s traceback into an unbounded walk over
`vpath_table_row` cells nothing legitimately wrote. That is a tight
`nogil`-compiled C loop holding the GIL -- it never reaches the bytecode
dispatch point that delivers a pending Python signal, so `signal.alarm`
cannot interrupt it (proved directly against Task 5's naive prototype; see
AGENTS.md's Traps entry and task-5-report.md). Only an external process
boundary can bound it, hence this file exists instead of an in-process call.

Usage: python _early_break_worker.py MODELS_DIR READS_GZ MODEL_KEY THRESHOLD
Prints repr(logp) to stdout and exits 0 on a normal (non-hanging) return.
"""
import gzip
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from advntr_harness.capture import _ModelCache


def _first_fixture(reads_path, model_key):
    with gzip.open(reads_path) as handle:
        content = handle.read()
    for line in content.split('\n'):
        if not line:
            continue
        key, sequence = line.split('\t', 1)
        if key == model_key:
            return sequence
    raise AssertionError('no %s fixture in %s' % (model_key, reads_path))


def main(argv):
    models_dir, reads_path, model_key, threshold = argv[1:5]
    assembly, read_length = model_key.split('@')
    cache = _ModelCache(models_dir)
    model, _fingerprint, _score = cache.get(assembly, int(read_length))
    model.dp_score_threshold = float(threshold)
    read = _first_fixture(reads_path, model_key)
    logp, _vpath = model.viterbi(read)
    sys.stdout.write('%r\n' % logp)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
