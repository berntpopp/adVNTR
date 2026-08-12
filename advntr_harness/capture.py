"""Capture and verify equivalence baselines.

    python -m advntr_harness.capture --tier 1 --out tests/golden
    python -m advntr_harness.capture --tier 2 --out tests/golden
    python -m advntr_harness.capture --tier 2 --out /tmp/check --verify tests/golden

Baselines MUST be captured from the pristine tree before any decoder change. Fixtures
captured after an optimisation prove nothing.

Three things here are less obvious than they look:

* **Model context is per file, not global.** `select_illumina_reads` derives its read
  length from `samfile.head(5)` and rebuilds the model, so two BAMs can be decoded by
  two different models. Decoding everything against one hand-built model would compare
  a population the tool never decodes.
* **Tier 2 streams.** Retaining every attempt to hash at the end means holding hundreds
  of thousands of full Viterbi paths in memory; digests are folded incrementally instead.
* **Tier 1 fixtures are re-decoded under canonical identities.** The expected rows must
  be reproducible by a gate that has only the fixture sequences, so they cannot carry the
  original file/ordinal/query-name.
"""
import argparse
import glob
import gzip
import hashlib
import json
import logging
import os
import sys
import time

from Bio.Seq import Seq

from advntr import settings
from advntr import vntr_finder as vf
from advntr.models import load_unique_vntrs_data
from advntr.utils import is_low_quality_read  # noqa: F401  (documents the parity source)
from advntr_harness.extract import derive_read_length, eligible_reads, resolve_contig
from advntr_harness.fingerprint import (comparable_fingerprint, input_attestation,
                                        model_fingerprint)
from advntr_harness.oracle import (DecodeAttempt, EMPTY_STREAM_SENTINEL, attempt_to_row,
                                   hex_to_logp, logp_to_hex)
from advntr_harness.strata import empty_strata, select_strata

VID = 25561
DEFAULT_DATA = '/home/bernt-popp/development/VNtyper/tests/data'
GOLDEN_MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'tests', 'golden', 'models')
#: Tier 1 source files.
#:
#: Not every hg19 BAM: decoding all seven costs ~4.8 h on the pristine build (142,676
#: reads x 2 orientations x 61 ms). These three cost ~35 min and provably fill every
#: stratum -- measured, they contribute all 77 reads in the corpus where the reverse
#: complement wins (6 + 12 + 59). example_66bf is deliberately excluded from the default:
#: it contributes 0 such reads in 12,608, and a fixture set built from it would go green
#: without exercising the branch upstream PR #57 deletes.
TIER1_FILES = (
    'example_7a61_hg19_subset.bam',
    'example_b178_hg19_subset.bam',
    'example_a5c1_hg19_subset.bam',
)
#: Canonical identity used for Tier 1 expected rows. See the module docstring.
TIER1_SOURCE = 'tier1'


#: Sources whose content defines decoder behaviour. Digested into every manifest so a
#: baseline carries proof of which kernel produced it -- `git stash list` proves nothing
#: about HEAD, working-tree diffs, or whether the imported .so is stale.
KERNEL_SOURCES = ('hmm/hmm.pyx', 'hmm/base.pyx', 'hmm/base.pxd', 'hmm/cqueue.pxd',
                  'hmm/queue.c')


def kernel_provenance(repo_root=None):
    """Digest of every source that defines decoder behaviour, plus the loaded .so."""
    if repo_root is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    digests = {}
    for relative in KERNEL_SOURCES:
        path = os.path.join(repo_root, relative)
        if not os.path.isfile(path):
            digests[relative] = 'absent'
            continue
        with open(path, 'rb') as handle:
            digests[relative] = hashlib.sha256(handle.read()).hexdigest()[:16]
    import hmm.hmm
    so_path = hmm.hmm.__file__
    if os.path.isfile(so_path):
        with open(so_path, 'rb') as handle:
            digests['loaded_so'] = hashlib.sha256(handle.read()).hexdigest()[:16]
    return digests


def build_finder(db_path):
    """Return (VNTRFinder, ReferenceVNTR) configured the way `genotype -fs` configures it."""
    settings.MAX_ERROR_RATE = 0.05
    settings.TRAINED_MODELS_DB = db_path
    settings.TRAINED_HMMS_DIR = os.path.dirname(os.path.realpath(db_path)) + '/'
    reference = [v for v in load_unique_vntrs_data(target_vids=[VID]) if v.id == VID][0]
    return vf.VNTRFinder(reference, is_frameshift_mode=True), reference


def assembly_for(path):
    """Which MUC1 database a file needs, from its path."""
    name = os.path.basename(path)
    relative = path
    if 'hg38' in name or 'hg38' in relative or 'GRCh38' in relative or '40cf' in name:
        return 'hg38'
    return 'hg19'


def decode_read(model, sequence):
    """Decode both orientations.

    Returns a list of (logp, indices, status, payload, vpath). `vpath` is the raw
    list of (index, State) tuples and is kept because `recruit_read` needs the State
    objects, not the indices -- passing indices raises TypeError inside
    get_number_of_matches_in_vpath. It is transient per read and never retained.
    """
    results = []
    for payload in (sequence, str(Seq(sequence).reverse_complement()).upper()):
        vpath = None
        try:
            logp, vpath = model.viterbi(payload)
            status = 'neg_inf' if logp == float('-inf') else 'ok'
            indices = tuple(int(index) for index, _state in vpath) if vpath else ()
        except Exception as exc:  # recorded as data, not handled
            logp, indices, status = float('-inf'), (), 'exception:%s' % type(exc).__name__
        results.append((logp, indices, status, payload, vpath))
    return results


def evaluate_read(model, recruitment_score, sequence, is_low_quality):
    """Mirror advntr/vntr_finder.py:1144-1160 and report what it decided, and why.

    Returns (attempt_payloads, decision) where attempt_payloads is a list of
    (orientation, payload, logp, indices, status).
    """
    (fwd_logp, fwd_indices, fwd_status, fwd_seq, fwd_vpath), \
        (rev_logp, rev_indices, rev_status, rev_seq, rev_vpath) = decode_read(model, sequence)

    payloads = [('fwd', fwd_seq, fwd_logp, fwd_indices, fwd_status),
                ('rev', rev_seq, rev_logp, rev_indices, rev_status)]

    # advntr/vntr_finder.py:1146-1150 -- reverse wins on a strict <, so ties keep forward.
    if fwd_logp < rev_logp:
        winner, logp, vpath = rev_seq, rev_logp, rev_vpath
    else:
        winner, logp, vpath = fwd_seq, fwd_logp, fwd_vpath

    if logp == float('-inf'):                                    # :1151-1153
        return payloads, 'rejected:low_likelihood'
    if is_low_quality and not vf.VNTRFinder.recruit_read(         # :1154-1156
            logp, vpath, recruitment_score, len(winner)):
        return payloads, 'rejected:low_quality'
    return payloads, 'selected'


def attempts_for_file(model, recruitment_score, source_file, reads):
    """Yield DecodeAttempt rows for one file, carrying the real selection decision."""
    for ordinal, name, sequence, _mapq, _start, is_low_quality in reads:
        payloads, decision = evaluate_read(model, recruitment_score, sequence,
                                           is_low_quality)
        for orientation, payload, logp, indices, status in payloads:
            yield DecodeAttempt(source_file, ordinal, name, orientation, payload,
                                logp_to_hex(logp), indices, status, decision)


def discover(data_dir, tier):
    """Deterministically ordered source files."""
    if tier == 1:
        return [os.path.join(data_dir, name) for name in TIER1_FILES
                if os.path.isfile(os.path.join(data_dir, name))]
    found = []
    for root, _dirs, files in os.walk(data_dir):
        for name in sorted(files):
            if name.endswith('.bam'):
                found.append(os.path.join(root, name))
    return sorted(found)


class _ModelCache(object):
    """One baked model per (assembly, read_length). Building is ~0.35 s each."""

    def __init__(self, models_dir):
        self._models_dir = models_dir
        self._finders = {}
        self._models = {}

    def finder(self, assembly):
        if assembly not in self._finders:
            db = os.path.join(self._models_dir, '%s_muc1.db' % assembly)
            self._finders[assembly] = build_finder(db)
        return self._finders[assembly]

    def get(self, assembly, read_length):
        key = (assembly, read_length)
        if key not in self._models:
            finder, reference = self.finder(assembly)
            model = finder.get_vntr_matcher_hmm(read_length=read_length)
            fingerprint = model_fingerprint(model, reference)
            recruitment_score = finder.get_min_score_to_select_a_read(read_length)
            self._models[key] = (model, fingerprint, recruitment_score)
        return self._models[key]


def capture(tier, data_dir, models_dir, collect_attempts):
    """Decode every eligible read in scope.

    `collect_attempts` retains attempts for stratification (Tier 1 only). Tier 2 folds
    digests incrementally instead: retaining hundreds of thousands of full paths would
    exhaust memory before the manifest is written.
    """
    import pysam

    cache = _ModelCache(models_dir)
    manifest = {'tier': tier, 'files': [], 'model_contexts': {},
                'kernel_provenance': kernel_provenance()}
    retained = []
    global_digest = hashlib.sha256()

    for path in discover(data_dir, tier):
        name = os.path.relpath(path, data_dir)
        started = time.time()
        assembly = assembly_for(path)
        _finder, reference = cache.finder(assembly)

        samfile = pysam.AlignmentFile(path, 'rb')
        try:
            contig = resolve_contig(samfile, reference.chromosome)
            read_length = derive_read_length(samfile) if contig is not None else None
        except IndexError:
            contig, read_length = None, None
        finally:
            samfile.close()

        if contig is None or read_length is None:
            manifest['files'].append({
                'source_file': name, 'assembly': assembly, 'read_length': None,
                'model_key': None, 'eligible_count': 0,
                'input_digest': EMPTY_STREAM_SENTINEL,
                'output_digest': EMPTY_STREAM_SENTINEL,
                'skipped': 'no usable contig' if contig is None else 'short head',
            })
            sys.stderr.write('%-52s SKIPPED (%s)\n'
                             % (name, 'no contig' if contig is None else 'short head'))
            continue

        model, fingerprint, recruitment_score = cache.get(assembly, read_length)
        model_key = '%s@%d' % (assembly, read_length)
        manifest['model_contexts'][model_key] = fingerprint

        reads = list(eligible_reads(path, reference, read_length=read_length))
        attestation = input_attestation(name, [read[2] for read in reads])
        attestation['assembly'] = assembly
        attestation['read_length'] = read_length
        attestation['model_key'] = model_key

        file_digest = hashlib.sha256()
        count = 0
        for attempt in attempts_for_file(model, recruitment_score, name, reads):
            row = attempt_to_row(attempt)
            file_digest.update(row)
            file_digest.update('\n')
            global_digest.update(row)
            global_digest.update('\n')
            count += 1
            if collect_attempts:
                retained.append(attempt)
        attestation['attempt_count'] = count
        attestation['output_digest'] = (EMPTY_STREAM_SENTINEL if count == 0
                                        else file_digest.hexdigest())
        manifest['files'].append(attestation)
        sys.stderr.write('%-52s reads=%-6d %s %6.1fs\n'
                         % (name, attestation['eligible_count'],
                            attestation['output_digest'][:16], time.time() - started))

    manifest['global_digest'] = (EMPTY_STREAM_SENTINEL if not manifest['files']
                                 else global_digest.hexdigest())
    return manifest, retained


def canonical_fixture_rows(cache, fixtures):
    """Decode the fixture sequences under canonical identities.

    `fixtures` is an ordered list of (model_key, sequence). Each sequence is decoded
    under **the model that classified it**, not a single hardcoded one: read length is
    derived per file, and the corpus really does disagree -- example_7a61 and
    example_b178 derive 151 (2565 states) while example_a5c1 derives 149 (2553 states),
    and a5c1 supplies most of the reverse-complement stratum. Decoding its fixtures
    under the 151 model would label the one stratum this harness exists to protect with
    rows the model never produced.

    The gate only ever has the fixture file, so expected rows must be reproducible from
    (model_key, sequence) alone -- they cannot carry the original file, ordinal or query
    name. Low quality is forced False: quality lives on the BAM record, not the
    sequence, so a sequence-only gate cannot reproduce it.
    """
    rows = []
    for index, (model_key, sequence) in enumerate(fixtures):
        assembly, read_length = model_key.split('@')
        model, _fingerprint, recruitment_score = cache.get(assembly, int(read_length))
        payloads, decision = evaluate_read(model, recruitment_score, sequence, False)
        for orientation, payload, logp, indices, status in payloads:
            rows.append(attempt_to_row(DecodeAttempt(
                TIER1_SOURCE, index, 'fixture%d' % index, orientation, payload,
                logp_to_hex(logp), indices, status, decision)))
    return rows


def read_fixture_file(path):
    """Parse a `model_key<TAB>sequence` fixture file into an ordered list of pairs."""
    with gzip.open(path) as handle:
        content = handle.read()
    fixtures = []
    for line in content.split('\n'):
        if not line:
            continue
        model_key, sequence = line.split('\t', 1)
        fixtures.append((model_key, sequence))
    return fixtures


def verify(baseline, actual):
    """Return a list of human-readable mismatches between two manifests."""
    problems = []

    base_contexts = baseline.get('model_contexts', {})
    new_contexts = actual.get('model_contexts', {})
    if set(base_contexts) != set(new_contexts):
        problems.append('model contexts changed: %s -> %s'
                        % (sorted(base_contexts), sorted(new_contexts)))
    for key in sorted(set(base_contexts) & set(new_contexts)):
        before = comparable_fingerprint(base_contexts[key])
        after = comparable_fingerprint(new_contexts[key])
        if before != after:
            problems.append('model %s changed: %r -> %r' % (key, before, after))

    base_kernel = baseline.get('kernel_provenance', {})
    new_kernel = actual.get('kernel_provenance', {})
    changed = sorted(k for k in set(base_kernel) | set(new_kernel)
                     if base_kernel.get(k) != new_kernel.get(k))
    if changed:
        # Informational, never a failure: the kernel is SUPPOSED to change between a
        # pristine baseline and a post-rewrite verification. Printed so a "VERIFIED"
        # result states which kernel it was verified against.
        sys.stderr.write('note: kernel changed since baseline: %s\n' % ', '.join(changed))

    baseline_files = dict((entry['source_file'], entry) for entry in baseline['files'])
    actual_files = dict((entry['source_file'], entry) for entry in actual['files'])
    if not baseline_files:
        problems.append('baseline contains no files at all')
    if set(baseline_files) != set(actual_files):
        problems.append('file set changed: only-baseline=%s only-actual=%s'
                        % (sorted(set(baseline_files) - set(actual_files)),
                           sorted(set(actual_files) - set(baseline_files))))
    for name in sorted(set(baseline_files) & set(actual_files)):
        before, after = baseline_files[name], actual_files[name]
        for key in ('eligible_count', 'attempt_count', 'input_digest', 'output_digest',
                    'read_length', 'model_key'):
            if before.get(key) != after.get(key):
                problems.append('%s: %s %r -> %r'
                                % (name, key, before.get(key), after.get(key)))
    if baseline.get('global_digest') != actual.get('global_digest'):
        problems.append('global digest %r -> %r'
                        % (baseline.get('global_digest'), actual.get('global_digest')))
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tier', type=int, required=True, choices=(1, 2))
    parser.add_argument('--out', required=True)
    parser.add_argument('--vntyper-data', default=DEFAULT_DATA)
    parser.add_argument('--models', default=GOLDEN_MODELS)
    parser.add_argument('--target', type=int, default=2000)
    parser.add_argument('--verify', metavar='BASELINE_DIR', default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.CRITICAL)

    # Preflight, because everything below is expensive and this is not. `--verify` used to
    # be resolved only after the capture, so a missing baseline cost the whole run before
    # anything reported it: ~14 min on this kernel for tier 2, and the pristine capture the
    # baseline must come from is ~4.8 h. A comparison that cannot happen should not be paid
    # for first.
    baseline_path = None
    if args.verify:
        baseline_path = os.path.join(args.verify, 'tier%d_manifest.json' % args.tier)
        if not os.path.isfile(baseline_path):
            raise SystemExit(
                'no baseline manifest at %s, so there is nothing to verify against. '
                'Capture one from pristine 05fd98a (checkout, `make build`, then this same '
                'command with --out pointing at a scratch directory and no --verify), copy '
                'tier%d_manifest.json into %s, and re-run. Refusing to spend the capture '
                'first.' % (baseline_path, args.tier, args.verify))

    manifest, retained = capture(args.tier, args.vntyper_data, args.models,
                                 collect_attempts=(args.tier == 1))

    if not os.path.isdir(args.out):
        os.makedirs(args.out)

    if args.tier == 1:
        sequences, counts = select_strata(retained, target=args.target)
        missing = empty_strata(counts)
        if missing:
            raise SystemExit('empty strata, fixture set is not adversarial: %s' % missing)
        manifest['strata'] = counts

        # Bind each fixture to the model that classified it. Read length is derived per
        # file and the corpus disagrees (7a61/b178 -> 151, a5c1 -> 149), so a single
        # hardcoded model would mislabel most of the reverse-complement stratum.
        model_key_of_file = dict((entry['source_file'], entry['model_key'])
                                 for entry in manifest['files'] if entry['model_key'])
        first_file_of_sequence = {}
        for attempt in retained:
            if attempt.orientation == 'fwd':
                first_file_of_sequence.setdefault(attempt.sequence, attempt.source_file)
        fixtures = [(model_key_of_file[first_file_of_sequence[sequence]], sequence)
                    for sequence in sequences]

        cache = _ModelCache(args.models)
        manifest['fixture_models'] = {}
        for model_key in sorted(set(key for key, _sequence in fixtures)):
            assembly, read_length = model_key.split('@')
            _model, fingerprint, _score = cache.get(assembly, int(read_length))
            manifest['fixture_models'][model_key] = fingerprint
        rows = canonical_fixture_rows(cache, fixtures)

        with gzip.open(os.path.join(args.out, 'tier1_reads.txt.gz'), 'wb') as handle:
            handle.write('\n'.join('%s\t%s' % pair for pair in fixtures))
        with gzip.open(os.path.join(args.out, 'tier1_expected.tsv.gz'), 'wb') as handle:
            handle.write('\n'.join(rows))
        sys.stderr.write('strata: %r\nfixtures: %d sequences across models %s, '
                         '%d expected rows\n'
                         % (counts, len(fixtures),
                            sorted(manifest['fixture_models']), len(rows)))

    manifest_path = os.path.join(args.out, 'tier%d_manifest.json' % args.tier)
    with open(manifest_path, 'w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    sys.stderr.write('wrote %s\n' % manifest_path)

    if baseline_path:
        with open(baseline_path) as handle:
            problems = verify(json.load(handle), manifest)
        if problems:
            for problem in problems:
                sys.stderr.write('MISMATCH %s\n' % problem)
            return 1
        sys.stderr.write('VERIFIED identical against %s\n' % baseline_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
