"""Native synthetic calls emit v2 only after completed decisions and source checks."""
import json
import os
import sqlite3
import unittest

from advntr.capture_assets import prepare_capture_assets
from advntr.capture_policy import resolve_capture_policy
from advntr.frameshift_capture_record import decode_capture
from advntr.models import load_unique_vntrs_data
from advntr.run_context import RunContext
from tests import test_exact_caller as reference


class TestCompletedCaptureWriter(unittest.TestCase):
    def setUp(self):
        self.fixture = reference._ExactCallerTestCase('setUp')
        self.fixture.setUp()
        self.path = os.path.join(self.fixture.tempdir, 'capture.jsonl')
        open(self.path, 'wb').close()
        model = os.path.join(self.fixture.tempdir, 'model.db')
        connection = sqlite3.connect(model)
        connection.execute('CREATE TABLE vntrs (id INTEGER, nonoverlapping TEXT, chromosome TEXT, ref_start INTEGER, '
                           'gene_name TEXT, annotation TEXT, pattern TEXT, left_flanking TEXT, right_flanking TEXT, '
                           'repeats TEXT, scaled_score REAL)')
        connection.execute('INSERT INTO vntrs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                           (1, 'True', 'chr1', 100, 'GENE', 'coding', reference.UNITS['1'],
                            'TTTTTTTT', 'GGGGGGGG', ','.join(reference.SEGMENTS), 0.0))
        connection.commit()
        connection.close()
        self.assets = prepare_capture_assets(model)
        self.fixture.finder.reference_vntr = load_unique_vntrs_data(self.assets.model_path, [1])[0]
        self.fixture.finder.run_context = RunContext(
            resolve_capture_policy(frameshift_mode=True, use_reference_alignment=False, minimum_read_match_ratio=0.6),
            self.fixture.finder.frameshift_policy, None, self.path, self.assets.model_path, 2, self.assets)

    def tearDown(self):
        self.assets.close()
        self.fixture.tearDown()

    def test_completed_native_calls_roundtrip_with_anonymous_evidence_and_verified_model(self):
        calls = self.fixture._run()
        with open(self.path) as handle:
            documents = [json.loads(line) for line in handle]
        self.assertEqual(1, len(documents))
        self.assertIn('schema_version', documents[0])
        record = documents[0]
        self.assertEqual('advntr-frameshift-capture-v2', record['schema_version'])
        decoded = decode_capture(record)
        replayed = [(visit.plan.state, visit.plan.read_support, visit.mean_coverage, visit.statistic.pvalue)
                    for visit in decoded.visits if visit.statistic is not None and visit.statistic.called]
        self.assertEqual(calls, replayed)
        self.assertEqual(len(reference._all_three_sites()), record['locus']['selected_read_count'])
        self.assertNotIn(self.fixture.tempdir, json.dumps(record))
        for read in reference._all_three_sites():
            if read.query_name:
                self.assertNotIn(read.query_name, json.dumps(record))

    def test_a_failed_decision_never_leaves_a_v2_completed_record(self):
        def fail(*args, **kwargs):
            raise ValueError('invented decision failure')
        self.fixture.finder.identify_frameshift = fail
        with self.assertRaises(ValueError):
            self.fixture._run()
        self.assertEqual(0, os.path.getsize(self.path))

    def test_reference_object_drift_is_refused_against_the_queried_snapshot(self):
        self.fixture.finder.reference_vntr.scaled_score = -0.25
        with self.assertRaises(ValueError):
            self.fixture._run()
        self.assertEqual(0, os.path.getsize(self.path))

    def test_duplicate_completion_or_torn_prior_record_never_gets_appended(self):
        self.fixture._run()
        with open(self.path, 'rb') as handle:
            original = handle.read()
        with self.assertRaises(ValueError):
            self.fixture._run()
        with open(self.path, 'rb') as handle:
            self.assertEqual(original, handle.read())
        with open(self.path, 'wb') as handle:
            handle.write('{"torn":')
        with self.assertRaises(ValueError):
            self.fixture._run()
        with open(self.path, 'rb') as handle:
            self.assertEqual('{"torn":', handle.read())

    def test_exact_native_capture_reproduces_its_original_loaded_background(self):
        self.assets.close()
        background_path = self.fixture._write_background()
        self.assets = prepare_capture_assets(os.path.join(self.fixture.tempdir, 'model.db'), background_path)
        self.fixture.finder.run_context = RunContext(
            resolve_capture_policy(frameshift_mode=True, caller_mode='exact', use_reference_alignment=False,
                                   minimum_read_match_ratio=0.6), self.fixture.finder.frameshift_policy,
            self.assets.background, self.path, self.assets.model_path, 2, self.assets)
        calls = self.fixture._run()
        with open(self.path) as handle:
            record = json.loads(handle.readline())
        decoded = decode_capture(record)
        self.assertIsNotNone(decoded.background)
        self.assertNotIn('provenance', record['loaded_background'])
        self.assertEqual(calls or [], [(visit.plan.state, visit.plan.read_support, visit.mean_coverage, visit.statistic.pvalue)
                                      for visit in decoded.visits if visit.statistic is not None and visit.statistic.called])

    def test_a_custom_legacy_callback_cannot_claim_the_native_replay_contract(self):
        original = self.fixture.finder.identify_frameshift
        self.fixture.finder.identify_frameshift = lambda *args, **kwargs: original(*args, **kwargs)
        with self.assertRaisesRegexp(ValueError, 'native legacy'):
            self.fixture._run()
        self.assertEqual(0, os.path.getsize(self.path))

    def test_missing_called_context_still_precedes_capture_completion(self):
        from advntr.frameshift_capture_writer import complete_frameshift_calls
        self.fixture.finder.last_frameshift_evidence = {'D1_1': ()}
        with self.assertRaises(AssertionError):
            complete_frameshift_calls(self.fixture.finder, [('D1_1', 3, 1.0, 0.0)], None, {}, [], 0)
        self.assertEqual(0, os.path.getsize(self.path))

    def test_reference_drift_during_serialization_is_detected_before_append(self):
        from advntr import frameshift_capture_writer as writer
        original = writer.model_document
        seen = []
        def drifting(reference):
            document = original(reference)
            if reference is self.fixture.finder.reference_vntr:
                seen.append(1)
                if len(seen) > 1:
                    document['scaled_score'] = -0.25
            return document
        writer.model_document = drifting
        try:
            with self.assertRaisesRegexp(ValueError, 'changed during completion'):
                self.fixture._run()
            self.assertEqual(0, os.path.getsize(self.path))
        finally:
            writer.model_document = original

    def test_append_refuses_ambiguous_prior_json_and_special_files(self):
        from advntr.frameshift_capture_writer import _append_completed
        from tests.test_frameshift_capture_record import capture_document
        raw = capture_document()
        for prior in ('{"x":1,"x":2}\n', '{"x":NaN}\n'):
            with open(self.path, 'wb') as handle:
                handle.write(prior)
            with self.assertRaises(ValueError):
                _append_completed(self.path, raw)
            with open(self.path, 'rb') as handle:
                self.assertEqual(prior, handle.read())
        fifo = os.path.join(self.fixture.tempdir, 'fifo')
        os.mkfifo(fifo)
        with self.assertRaises(ValueError):
            _append_completed(fifo, raw)

    def test_multiple_loci_require_the_same_run_bindings(self):
        from advntr.frameshift_capture_writer import _append_completed
        from advntr.frameshift_capture_record import digest
        from tests.test_frameshift_capture_record import capture_document
        raw = capture_document()
        _append_completed(self.path, raw)
        raw['model_locus']['vntr_id'] = raw['locus']['vntr_id'] = 18
        raw['assets']['model_locus_sha256'] = digest(raw['model_locus'])
        _append_completed(self.path, raw)
        with open(self.path) as handle:
            self.assertEqual(2, len(handle.readlines()))
        raw['model_locus']['vntr_id'] = raw['locus']['vntr_id'] = 19
        raw['assets']['model_locus_sha256'] = digest(raw['model_locus'])
        raw['producer']['build_id'] = 'd' * 64
        with self.assertRaisesRegexp(ValueError, 'incompatible run'):
            _append_completed(self.path, raw)
        raw['producer']['build_id'] = 'a' * 64
        raw['assets']['model_sha256'] = 'f' * 64
        with self.assertRaisesRegexp(ValueError, 'different run assets'):
            _append_completed(self.path, raw)
