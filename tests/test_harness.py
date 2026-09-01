"""Tests for the equivalence harness itself.

The harness is what every later correctness claim rests on, so it gets tested before it
is trusted.
"""
import copy
import json
import os
import shutil
import struct
import tempfile
import unittest

import pysam

from advntr_harness.capture import verify
from advntr_harness.extract import derive_read_length, eligible_reads, resolve_contig
from advntr_harness.fingerprint import (comparable_fingerprint, input_attestation,
                                        model_fingerprint)
from advntr_harness.oracle import (DecodeAttempt, EMPTY_STREAM_SENTINEL, attempt_to_row,
                                   hex_to_logp, logp_to_hex, stream_digest)

DATA = '/home/bernt-popp/development/VNtyper/tests/data'
BAM = os.path.join(DATA, 'example_7a61_hg19_subset.bam')
has_data = unittest.skipUnless(os.path.isfile(BAM), 'VNtyper test data not available')


def _attempt(**kwargs):
    base = dict(source_file='a.bam', fetch_ordinal=0, query_name='r1',
                orientation='fwd', sequence='ACGT', logp_hex=logp_to_hex(-1.5),
                vpath_indices=(1, 2, 3), exit_status='ok',
                selection_decision='selected')
    base.update(kwargs)
    return DecodeAttempt(**base)


class _Vntr(object):
    id = 25561
    chromosome = 'chr1'
    start_point = 155160983
    pattern = 'ACGT'

    def get_length(self):
        return 840

    def get_repeat_segments(self):
        return ['ACGT']


class TestLogpHex(unittest.TestCase):
    def test_hex_is_exact_ieee754(self):
        self.assertEqual(logp_to_hex(-1.5), struct.pack('>d', -1.5).encode('hex'))

    def test_distinguishes_neighbouring_doubles(self):
        self.assertNotEqual(logp_to_hex(-367.2180000000001),
                            logp_to_hex(-367.2180000000002))

    def test_negative_infinity_round_trips(self):
        self.assertEqual(hex_to_logp(logp_to_hex(float('-inf'))), float('-inf'))

    def test_round_trip_is_lossless_for_a_real_score(self):
        value = -332.37249889063224
        self.assertEqual(hex_to_logp(logp_to_hex(value)), value)


class TestRow(unittest.TestCase):
    def test_row_is_tab_separated_with_no_newline(self):
        row = attempt_to_row(_attempt())
        self.assertNotIn('\n', row)
        self.assertEqual(len(row.split('\t')), 9)

    def test_vpath_is_comma_joined(self):
        self.assertIn('4,5', attempt_to_row(_attempt(vpath_indices=(4, 5))))

    def test_empty_vpath_renders_as_an_empty_field(self):
        self.assertEqual(attempt_to_row(_attempt(vpath_indices=())).split('\t')[6], '')


class TestStreamDigest(unittest.TestCase):
    def test_digest_is_order_sensitive(self):
        a, b = _attempt(query_name='r1'), _attempt(query_name='r2')
        self.assertNotEqual(stream_digest([a, b]), stream_digest([b, a]))

    def test_digest_detects_a_single_changed_bit(self):
        self.assertNotEqual(
            stream_digest([_attempt()]),
            stream_digest([_attempt(logp_hex=logp_to_hex(-1.5000000000000002))]))

    def test_empty_stream_is_a_sentinel_not_a_hash(self):
        self.assertEqual(stream_digest([]), EMPTY_STREAM_SENTINEL)


class TestInputAttestation(unittest.TestCase):
    def test_records_the_count(self):
        self.assertEqual(input_attestation('a.bam', ['ACGT', 'TGCA'])['eligible_count'], 2)

    def test_empty_input_is_flagged_not_hashed(self):
        attestation = input_attestation('a.bam', [])
        self.assertEqual(attestation['eligible_count'], 0)
        self.assertEqual(attestation['input_digest'], EMPTY_STREAM_SENTINEL)

    def test_digest_is_order_sensitive(self):
        self.assertNotEqual(input_attestation('a.bam', ['ACGT', 'TGCA'])['input_digest'],
                            input_attestation('a.bam', ['TGCA', 'ACGT'])['input_digest'])


class _FakeModel(object):
    def __init__(self, n_states=2559, read_length=150):
        self.n_states = n_states
        self.read_length_used_to_build_model = read_length
        self.dp_score_threshold = -367.218


class TestModelFingerprint(unittest.TestCase):
    def test_has_every_required_key(self):
        fingerprint = model_fingerprint(_FakeModel(), _Vntr())
        for key in ('n_states', 'read_length', 'dp_score_threshold_hex',
                    'csr_digest', 'vntr_digest'):
            self.assertIn(key, fingerprint)

    def test_pristine_model_is_marked_not_faked(self):
        self.assertEqual(model_fingerprint(_FakeModel(), _Vntr())['csr_digest'],
                         'pre-csr')

    def test_state_count_change_is_visible(self):
        """2559 (read length 150) and 2565 (151) must never compare equal."""
        a = model_fingerprint(_FakeModel(n_states=2559, read_length=150), _Vntr())
        b = model_fingerprint(_FakeModel(n_states=2565, read_length=151), _Vntr())
        self.assertNotEqual(comparable_fingerprint(a), comparable_fingerprint(b))

    def test_comparable_fingerprint_excludes_the_csr_digest(self):
        """The CSR digest legitimately changes across the rewrite; the rest must not."""
        self.assertNotIn('csr_digest', comparable_fingerprint(
            model_fingerprint(_FakeModel(), _Vntr())))


@has_data
class TestExtraction(unittest.TestCase):
    def test_read_length_is_derived_the_way_production_derives_it(self):
        samfile = pysam.AlignmentFile(BAM, 'rb')
        try:
            self.assertEqual(derive_read_length(samfile), 151)
        finally:
            samfile.close()

    def test_ordinals_are_dense_and_ascending(self):
        reads = list(eligible_reads(BAM, _Vntr()))
        self.assertEqual([r[0] for r in reads], list(range(len(reads))))

    def test_extraction_is_not_empty(self):
        self.assertGreater(len(list(eligible_reads(BAM, _Vntr()))), 1000)

    def test_no_sequence_contains_N(self):
        for read in eligible_reads(BAM, _Vntr()):
            self.assertNotIn('N', read[2])

    def test_refseq_accession_contigs_resolve_to_none(self):
        path = os.path.join(DATA, 'remapped/bwa/GRCh38/example_7a61_GRCh38_bwa.bam')
        if not os.path.isfile(path):
            self.skipTest('remapped corpus not available')
        samfile = pysam.AlignmentFile(path, 'rb')
        try:
            self.assertIsNone(resolve_contig(samfile, 'chr1'))
        finally:
            samfile.close()

    def test_ensembl_style_contig_resolves_without_chr(self):
        path = os.path.join(DATA,
                            'remapped/bwa/hg19_ensembl/example_7a61_hg19_ensembl_bwa.bam')
        if not os.path.isfile(path):
            self.skipTest('remapped corpus not available')
        samfile = pysam.AlignmentFile(path, 'rb')
        try:
            self.assertEqual(resolve_contig(samfile, 'chr1'), '1')
        finally:
            samfile.close()




class TestVerifyPreflight(unittest.TestCase):
    """`--verify` was checked only AFTER the capture had already run.

    `make tier2` captures the whole corpus and only then opens the baseline manifest --
    which has never existed in this tree, in the working copy or in history. So every
    invocation spent the entire capture and ended in IOError. Reproduced at 1/8 scale:
    38.2 s of real decoding, the new manifest written, then the crash. On the full corpus
    that is ~14 min on this kernel, and ~4.8 h on the pristine one the baseline must come
    from.

    The check costs one `os.path.isfile` and belongs before the work, not after it.
    """

    def _run_with_stubbed_capture(self, stub, argv):
        import advntr_harness.capture as capture_module

        original = capture_module.capture
        capture_module.capture = stub
        try:
            return capture_module.main(argv)
        finally:
            capture_module.capture = original

    def test_a_missing_baseline_is_refused_before_any_decoding(self):
        calls = []

        def must_not_run(*args, **kwargs):
            calls.append(args)
            raise AssertionError('capture() ran despite a missing baseline manifest')

        out = tempfile.mkdtemp()
        baseline = tempfile.mkdtemp()
        try:
            self.assertRaises(
                SystemExit,
                self._run_with_stubbed_capture,
                must_not_run,
                ['--tier', '2', '--out', out, '--verify', baseline])
        finally:
            shutil.rmtree(out)
            shutil.rmtree(baseline)

        self.assertEqual(calls, [])

    def test_the_message_names_the_path_it_could_not_find(self):
        out = tempfile.mkdtemp()
        baseline = tempfile.mkdtemp()
        try:
            try:
                self._run_with_stubbed_capture(
                    lambda *a, **k: ({}, []),
                    ['--tier', '2', '--out', out, '--verify', baseline])
            except SystemExit as exc:
                message = str(exc)
            else:
                self.fail('a missing baseline must abort')
        finally:
            shutil.rmtree(out)
            shutil.rmtree(baseline)

        self.assertIn(os.path.join(baseline, 'tier2_manifest.json'), message)

    def test_a_present_baseline_still_reaches_the_capture(self):
        """The guard must refuse a missing baseline, not every baseline."""
        # A baseline with no files at all is refused by `verify` itself, which is a
        # separate and correct guard -- so this fixture has to be a realistic one.
        manifest = {'global_digest': 'abc',
                    'files': [{'source_file': 'a.bam', 'eligible_count': 1,
                               'attempt_count': 2, 'input_digest': 'i',
                               'output_digest': 'o', 'read_length': 151,
                               'model_key': 'hg19@151'}]}
        out = tempfile.mkdtemp()
        baseline = tempfile.mkdtemp()
        try:
            with open(os.path.join(baseline, 'tier2_manifest.json'), 'w') as handle:
                json.dump(manifest, handle)

            status = self._run_with_stubbed_capture(
                lambda *a, **k: (dict(manifest), []),
                ['--tier', '2', '--out', out, '--verify', baseline])
        finally:
            shutil.rmtree(out)
            shutil.rmtree(baseline)

        self.assertEqual(status, 0)


def _clean_manifest():
    """A small, self-contained, synthetic pair -- not derived from any corpus. Two files
    so a single-file perturbation can be told apart from a file-set change, and a full
    model_contexts entry because comparable_fingerprint() indexes every one of its four
    keys directly and raises KeyError on a partial fixture."""
    return {
        'tier': 2,
        'kernel_provenance': {'hmm/hmm.pyx': 'deadbeefcafebabe'},
        'model_contexts': {
            'hg19@151': {'n_states': 2565, 'read_length': 151,
                         'dp_score_threshold_hex': 'c076f4223f1451e6',
                         'vntr_digest': 'digest0', 'csr_digest': 'pre-csr'},
        },
        'files': [
            {'source_file': 'a.bam', 'eligible_count': 1, 'attempt_count': 2,
             'input_digest': 'ia', 'output_digest': 'oa', 'read_length': 151,
             'model_key': 'hg19@151'},
            {'source_file': 'b.bam', 'eligible_count': 3, 'attempt_count': 6,
             'input_digest': 'ib', 'output_digest': 'ob', 'read_length': 151,
             'model_key': 'hg19@151'},
        ],
        'global_digest': 'global0',
    }


class TestVerify(unittest.TestCase):
    """`verify()` is the pure function every equivalence claim rests on -- `make tier2`
    compiles down to one call to it. A gate nobody has seen fail is a gate nobody knows
    works (tests/test_ratchets.py's docstring makes the same point about a different
    ratchet), so this proves it catches a mismatch in two distinct shapes -- a single
    per-file output_digest and the whole-baseline global_digest -- not just that it stays
    quiet on a clean pair. See task-2-report.md's Phase B section for the equivalent
    demonstration run directly against the committed tests/golden/tier2_manifest.json.
    """

    def test_an_identical_pair_reports_no_problems(self):
        baseline = _clean_manifest()
        self.assertEqual(verify(baseline, copy.deepcopy(baseline)), [])

    def test_a_perturbed_per_file_output_digest_is_caught(self):
        baseline = _clean_manifest()
        actual = copy.deepcopy(baseline)
        actual['files'][0]['output_digest'] = 'deadbeefdeadbeef'

        problems = verify(baseline, actual)

        self.assertEqual(problems, ["a.bam: output_digest 'oa' -> 'deadbeefdeadbeef'"])

    def test_a_perturbed_global_digest_is_caught(self):
        baseline = _clean_manifest()
        actual = copy.deepcopy(baseline)
        actual['global_digest'] = 'global1'

        problems = verify(baseline, actual)

        self.assertEqual(problems, ["global digest 'global0' -> 'global1'"])

    def test_restoring_the_perturbation_passes_again(self):
        """The other half of the demonstration: a perturbation caught above must stop
        being caught once undone -- a gate that never goes green again would be useless
        even though it fails correctly."""
        baseline = _clean_manifest()
        actual = copy.deepcopy(baseline)
        actual['files'][0]['output_digest'] = 'deadbeefdeadbeef'
        self.assertTrue(verify(baseline, actual))

        actual['files'][0]['output_digest'] = baseline['files'][0]['output_digest']
        self.assertEqual(verify(baseline, actual), [])


if __name__ == '__main__':
    unittest.main()
