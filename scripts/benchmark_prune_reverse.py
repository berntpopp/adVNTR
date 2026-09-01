"""Benchmark Task 8's `--prune-reverse` against the unpruned decode.

    python scripts/benchmark_prune_reverse.py [--reads N] [--repeats N]

`scripts/benchmark_decoder.py` is the serial baseline tool (AGENTS.md), but it calls
`advntr_harness.capture.decode_read`, which always runs `model.viterbi(payload)`
unpruned -- it has no notion of Task 8's per-call threshold. This script drives the same
production code path Task 8 actually changed instead: `advntr.read_selection.decode_serially`
/ `_decode_one`, exactly as `select_illumina_reads` phase 2 calls it, just with
`prune_reverse` toggled.

Runs are INTERLEAVED (off, on, off, on, ...), not off-block-then-on-block, so a shared
drift in machine load lands on both arms rather than biasing one. Reports MIN and MEDIAN
ms/attempt -- never mean, which one slow outlier (a scheduler preemption, a GC pause)
distorts -- plus `os.getloadavg()` so a benchmark run can be judged against how busy the
machine was. `PendingRead` objects are rebuilt fresh every trial: `_decode_one` mutates
them in place, so reusing one across trials would silently make later trials free rides
on earlier tracebacks.

No cohort data anywhere here: only the public `example_*` BAMs this repo already names
elsewhere (AGENTS.md, scripts/benchmark_decoder.py) are ever passed as `--bam`.
"""
import argparse
import os
import sys
import time

# Runnable as `python scripts/benchmark_prune_reverse.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Bio.Seq import Seq

from advntr.read_selection import PendingRead, decode_serially
from advntr_harness.capture import _ModelCache
from advntr_harness.extract import eligible_reads

DEFAULT_BAM = ('/home/bernt-popp/development/VNtyper/tests/data/'
               'example_7a61_hg19_subset.bam')
MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'tests', 'golden', 'models')


def load_pending(bam, models_dir, read_length, limit):
    cache = _ModelCache(models_dir)
    model, fingerprint, _score = cache.get('hg19', read_length)
    _finder, reference = cache.finder('hg19')
    sequences = [read[2] for read in
                 eligible_reads(bam, reference, read_length=read_length)]
    if limit:
        sequences = sequences[:limit]
    return model, fingerprint, sequences


def _fresh_pending(sequences):
    return [PendingRead(sequence=seq,
                        reverse=str(Seq(seq).reverse_complement()).upper(),
                        query_name='r%d' % i)
           for i, seq in enumerate(sequences)]


def one_trial(model, sequences, prune_reverse):
    pending = _fresh_pending(sequences)
    start = time.time()
    decode_serially(model, pending, prune_reverse)
    return time.time() - start


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bam', default=DEFAULT_BAM)
    parser.add_argument('--models', default=MODELS)
    parser.add_argument('--read-length', type=int, default=151)
    parser.add_argument('--reads', type=int, default=400,
                        help='0 for every eligible read')
    parser.add_argument('--repeats', type=int, default=5,
                        help='interleaved off/on repetitions')
    args = parser.parse_args(argv)

    model, fingerprint, sequences = load_pending(
        args.bam, args.models, args.read_length, args.reads)
    attempts = 2 * len(sequences)
    print('model %d states @ read length %d; %d reads, %d decode attempts'
          % (fingerprint['n_states'], fingerprint['read_length'],
             len(sequences), attempts))
    print('load average before: %.2f %.2f %.2f' % os.getloadavg())

    off_times, on_times = [], []
    for _ in range(args.repeats):
        off_times.append(one_trial(model, sequences, False))
        on_times.append(one_trial(model, sequences, True))

    print('load average after:  %.2f %.2f %.2f' % os.getloadavg())

    for label, times in (('off', off_times), ('on', on_times)):
        times_sorted = sorted(times)
        minimum = times_sorted[0]
        n = len(times_sorted)
        median = (times_sorted[n // 2] if n % 2 else
                  (times_sorted[n // 2 - 1] + times_sorted[n // 2]) / 2.0)
        print('%-3s  min %7.3f s (%6.2f ms/attempt)   median %7.3f s (%6.2f ms/attempt)'
              % (label, minimum, minimum / attempts * 1000.0,
                 median, median / attempts * 1000.0))

    speedup_min = sorted(off_times)[0] / sorted(on_times)[0]
    speedup_median = ((sorted(off_times)[len(off_times) // 2]) /
                      (sorted(on_times)[len(on_times) // 2]))
    print('speedup (on vs off): %.2fx by min, %.2fx by median' % (speedup_min, speedup_median))
    return 0


if __name__ == '__main__':
    sys.exit(main())
