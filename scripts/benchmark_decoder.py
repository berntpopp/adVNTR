"""Benchmark the Viterbi decoder on real reads.

    python scripts/benchmark_decoder.py [--reads N] [--threads N]

Reports ms per decode attempt. The pristine 05fd98a baseline, measured on
example_7a61_hg19_subset.bam with the hg19@151 model, is 61.1 ms/attempt
(1617 reads, 3234 attempts, 197.5 s) -- quote that as the comparison point.
"""
import argparse
import os
import sys
import threading
import time

# Runnable as `python scripts/benchmark_decoder.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from advntr_harness.capture import _ModelCache, decode_read
from advntr_harness.extract import eligible_reads

DEFAULT_BAM = ('/home/bernt-popp/development/VNtyper/tests/data/'
               'example_7a61_hg19_subset.bam')
MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'tests', 'golden', 'models')
PRISTINE_MS_PER_ATTEMPT = 61.1


def load_sequences(bam, models_dir, read_length, limit):
    cache = _ModelCache(models_dir)
    model, fingerprint, _score = cache.get('hg19', read_length)
    _finder, reference = cache.finder('hg19')
    sequences = [read[2] for read in
                 eligible_reads(bam, reference, read_length=read_length)]
    if limit:
        sequences = sequences[:limit]
    return model, fingerprint, sequences


def run_serial(model, sequences):
    start = time.time()
    for sequence in sequences:
        decode_read(model, sequence)
    return time.time() - start


def run_threaded(model, sequences, n_threads):
    """Decode across threads. Only meaningful once the DP releases the GIL."""
    cursor = [0]
    lock = threading.Lock()

    def worker():
        while True:
            with lock:
                index = cursor[0]
                cursor[0] += 1
            if index >= len(sequences):
                return
            decode_read(model, sequences[index])

    start = time.time()
    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return time.time() - start


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bam', default=DEFAULT_BAM)
    parser.add_argument('--models', default=MODELS)
    parser.add_argument('--read-length', type=int, default=151)
    parser.add_argument('--reads', type=int, default=200,
                        help='0 for every eligible read')
    parser.add_argument('--threads', type=int, default=0,
                        help='also run a thread-scaling sweep up to this count')
    args = parser.parse_args(argv)

    model, fingerprint, sequences = load_sequences(
        args.bam, args.models, args.read_length, args.reads)
    attempts = 2 * len(sequences)
    print('model %d states @ read length %d; %d reads, %d decode attempts'
          % (fingerprint['n_states'], fingerprint['read_length'],
             len(sequences), attempts))

    elapsed = run_serial(model, sequences)
    per_attempt = elapsed / attempts * 1000.0
    print('serial      %7.2f s   %6.2f ms/attempt   %5.1fx vs pristine (%.1f ms)'
          % (elapsed, per_attempt, PRISTINE_MS_PER_ATTEMPT / per_attempt,
             PRISTINE_MS_PER_ATTEMPT))

    if args.threads:
        counts = [n for n in (2, 4, 8, 16, 32) if n <= args.threads]
        for n_threads in counts:
            threaded = run_threaded(model, sequences, n_threads)
            print('%2d threads  %7.2f s   %6.2f ms/attempt   %5.2fx vs serial'
                  % (n_threads, threaded, threaded / attempts * 1000.0,
                     elapsed / threaded))
    return 0


if __name__ == '__main__':
    sys.exit(main())
