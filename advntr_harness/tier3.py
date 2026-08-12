"""Occurrence-level oracle: runs the real `select_illumina_reads`.

Tiers 1 and 2 compare *unique sequences*, which is the right unit for the decoder and
the wrong unit for anything that changes the read loop. Threading `select_illumina_reads`
risks losing or reordering duplicate occurrences, mis-associating mapq/query_name, and
interleaving the DEBUG log that `hmm_alignment.generate_aln` parses without framing --
none of which a unique-sequence stream can see.

So this tier calls the production function itself and digests what it returned, in order.
"""
import hashlib
import os
import sys

from advntr import settings
from advntr_harness.oracle import EMPTY_STREAM_SENTINEL, logp_to_hex


def selected_read_row(index, read):
    """One tab-separated row per SelectedRead, in the order the function returned them."""
    path = ','.join(str(int(state_index)) for state_index, _state in read.vpath) \
        if read.vpath else ''
    return '\t'.join([
        str(index),
        str(read.query_name),
        str(read.sequence),
        logp_to_hex(read.logp),
        str(read.mapq),
        str(bool(read.is_mapped)),
        path,
    ])


def selection_digest(selected_reads):
    """sha256 over the ordered selected-read stream, or a sentinel when empty.

    Order-sensitive deliberately: `generate_aln` consumes an unframed log stream whose
    correctness depends on read order, and tied mutations are sorted by count alone.
    """
    digest = hashlib.sha256()
    seen = False
    for index, read in enumerate(selected_reads):
        seen = True
        digest.update(selected_read_row(index, read))
        digest.update('\n')
    if not seen:
        return EMPTY_STREAM_SENTINEL
    return digest.hexdigest()


def run_selection(finder, alignment_path, threads=None):
    """Run the production selection path, optionally at a given thread count.

    `threads` sets `settings.CORES`, which is what `--threads` writes. Before the
    threading work it is inert on this path; afterwards it is the control. Snapshotted
    and restored so a test cannot leak it into the next one.
    """
    previous = settings.CORES
    if threads is not None:
        settings.CORES = threads
    try:
        return finder.select_illumina_reads(alignment_path, [])
    finally:
        settings.CORES = previous


def selection_evidence(finder, alignment_path, threads=None):
    """What `select_illumina_reads` RETURNED, in order, as one comparable dict.

    This used to say "everything a threading change could plausibly break", which is not
    defensible. It digests the returned `SelectedRead` stream -- and all of it: index,
    query_name, sequence, logp, mapq, is_mapped and vpath are every attribute the class
    has. What it cannot see is everything that is not in the return value:

    * the DEBUG log stream, including the rejection lines and the final mapped-bp line,
      which `hmm_alignment.generate_aln` parses without framing. Ordering there is
      protected structurally -- all logging is deferred to the serial assembly phase, in
      fetch order -- rather than by this gate;
    * `vntr_bp_in_mapped_reads`, which is a local consumed only by a `logging.debug`;
    * peak memory. The loop retains one traceback per eligible read -- more than pristine,
      which keeps one per selected read -- and this cannot notice at any thread count;
    * exception behaviour, which `tests/test_read_selection.py` covers instead.

    Args:
        finder: A VNTRFinder from `capture.build_finder`.
        alignment_path: The BAM to select from.
        threads: `settings.CORES` for the call, or None to leave it alone.

    Returns:
        dict: count, digest, query_names and model_states.
    """
    selected = run_selection(finder, alignment_path, threads=threads)
    return {
        'count': len(selected),
        'digest': selection_digest(selected),
        'query_names': [read.query_name for read in selected],
        'model_states': int(finder.hmm.n_states) if finder.hmm is not None else None,
    }


#: What the Tier 3 baseline actually is.
#:
#: Tier 1 is the pristine gate: `tests/golden/tier1_manifest.json` records
#: `kernel_provenance["hmm/hmm.pyx"] = "e87fabf5e8633235"`, which is the digest at 05fd98a,
#: so its 4,000 expected rows were captured BEFORE the rewrite and byte-compare against it.
#:
#: Tier 3 is not that, and the manifest used to carry nothing that said so -- it held a
#: count and a digest and was added by 82b1c2b, the threading commit itself. Nothing in the
#: tree established which kernel produced it. It is a regression baseline for the read
#: loop, captured post-rewrite, and recording that is the difference between a gate and a
#: gate that is believed to be something stronger than it is.
BASELINE_KIND = 'post-rewrite regression baseline'

BASELINE_NOTE = (
    'Captured on the threaded read loop, not on pristine 05fd98a. It pins the returned '
    'SelectedRead stream against later read-loop changes; it does not prove equivalence '
    'with the pristine implementation. Tier 1 is the pristine gate -- see '
    'tests/golden/tier1_manifest.json, whose kernel_provenance names the 05fd98a kernel.'
)


#: Exactly what `tests/golden/tier3_manifest.json` carries.
#:
#: Named rather than implicit so the shipped artefact and the function that produces it
#: cannot drift apart: `tests/test_tier3_occurrence.py` asserts the file's key set against
#: this tuple, which costs nothing and does not need the corpus.
BASELINE_MANIFEST_KEYS = ('baseline_kind', 'count', 'digest', 'kernel_provenance',
                          'model_states', 'note', 'source_file')


def baseline_manifest(finder, alignment_path):
    """Build the Tier 3 baseline, stamped with what produced it.

    Args:
        finder: A VNTRFinder from `capture.build_finder`.
        alignment_path: The BAM the baseline is captured from.

    Returns:
        dict: The manifest, keyed by :data:`BASELINE_MANIFEST_KEYS`.
    """
    # Deferred, not circular -- capture.py never imports this module. It pulls in pysam
    # and Bio, which the rest of tier3 does not need.
    from advntr_harness.capture import kernel_provenance

    evidence = selection_evidence(finder, alignment_path)
    return {
        'count': evidence['count'],
        'digest': evidence['digest'],
        'model_states': evidence['model_states'],
        'source_file': os.path.basename(alignment_path),
        'baseline_kind': BASELINE_KIND,
        'note': BASELINE_NOTE,
        'kernel_provenance': kernel_provenance(),
    }


def main(argv=None):
    """Recapture the Tier 3 baseline and write it.

    This exists so the manifest has a committed producer. Without one, the only record of
    how `tests/golden/tier3_manifest.json` was made is a shell snippet in someone's
    history, which is the failure `golden_cohort_gate.py` was written to end on the
    VNtyper side.

    Args:
        argv: Arguments without the program name; defaults to `sys.argv[1:]`.

    Returns:
        int: 0 on success.
    """
    import argparse  # noqa: PLC0415 - only needed when run as a script
    import json

    from advntr_harness.capture import build_finder

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', default=os.path.join('tests', 'golden', 'models', 'hg19_muc1.db'))
    parser.add_argument('--bam', required=True)
    parser.add_argument('--out', default=os.path.join('tests', 'golden', 'tier3_manifest.json'))
    args = parser.parse_args(argv)

    finder, _reference = build_finder(args.db)
    manifest = baseline_manifest(finder, args.bam)
    with open(args.out, 'w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    sys.stderr.write('wrote %s (count=%d digest=%s)\n'
                     % (args.out, manifest['count'], manifest['digest'][:16]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
