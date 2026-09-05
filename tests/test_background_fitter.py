"""Unit tests for background fitting modules."""
import json
import os
import shutil
import tempfile
import unittest

from advntr.frameshift_background import SCHEMA, load_background_model
from advntr import background_fitter as bf

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
PUBLIC_RUNS = os.path.expanduser(
    '~/.cache/advntr-bench/sdd-external-calling-quality/capture/roots/sinkA/runs')
PUBLIC_RUNS_B = os.path.expanduser(
    '~/.cache/advntr-bench/sdd-external-calling-quality/capture/roots/sinkB/runs')


def span(pattern, reached=0, inserted=0, saw_start=False, saw_end=False, count=1):
    return [pattern, reached, inserted, bool(saw_start), bool(saw_end), count]


def row(candidate, opportunities, state_identities=None, legacy_support=0,
        support=None, pattern_index=None):
    state_identities = state_identities or {}
    identities = []
    for pairs in state_identities.values():
        identities.extend(pairs)
    return {
        'candidate': candidate,
        'legacy_states': sorted(state_identities),
        'legacy_support': legacy_support,
        'opportunities': opportunities,
        'pattern_index': pattern_index or bf.class_of(candidate),
        'support': len(identities) if support is None else support,
        'support_identities': identities,
        'state_identities': state_identities,
        'avg_bp_coverage': 100.0,
        'ru_bp_coverage': 6000,
        'ru_bp_coverage_ratio': 100,
        'ru_length': 60,
    }


def doc(spans, candidates, vntr_id=25561, read_length=151, is_haploid=False):
    return {'schema': bf.SINK_SCHEMA, 'version': bf.SINK_VERSION, 'vntr_id': vntr_id,
            'read_length': read_length, 'is_haploid': is_haploid,
            'spans': spans, 'candidates': candidates}


def label(sample_id, truth, pair_id, variant_class='vc', array_length=60):
    return {'sample_id': sample_id, 'truth': truth, 'partition': 'calibration',
            'pair_id': pair_id, 'variant_class': variant_class,
            'array_length': array_length}


class TempDirTestCase(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='task8k-')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write_sink(self, name, documents):
        path = os.path.join(self.tmp, name)
        with open(path, 'w') as handle:
            for document in documents:
                handle.write(json.dumps(document, sort_keys=True,
                                        separators=(',', ':')) + '\n')
        return path

# ---------------------------------------------------------------- sink parsing


class TestSinkParsing(TempDirTestCase):
    """Contract 1 of the sink review: abort on an unparseable line, never skip one."""

    def test_reads_every_line_of_a_well_formed_sink(self):
        path = self.write_sink('ok.jsonl', [doc([span('1', reached=8)],
                                                [row('D3_1', 1)])])
        documents = bf.read_sink(path)
        self.assertEqual(1, len(documents))
        self.assertEqual(25561, documents[0]['vntr_id'])

    def test_torn_line_aborts_and_names_the_file_and_line_number(self):
        path = os.path.join(self.tmp, 'torn.jsonl')
        good = json.dumps(doc([span('1', reached=8)], [row('D3_1', 1)]),
                          sort_keys=True, separators=(',', ':'))
        with open(path, 'w') as handle:
            handle.write(good + '\n')
            handle.write(good[:len(good) // 2] + '\n')
            handle.write(good + '\n')
        try:
            bf.read_sink(path)
        except bf.FitterError as error:
            message = str(error)
        else:
            self.fail('a torn line must abort the fit')
        self.assertIn('torn.jsonl', message)
        self.assertIn(':2', message)

    def test_a_blank_line_is_also_a_refusal_not_a_skip(self):
        path = os.path.join(self.tmp, 'blank.jsonl')
        good = json.dumps(doc([], []), sort_keys=True, separators=(',', ':'))
        with open(path, 'w') as handle:
            handle.write(good + '\n\n' + good + '\n')
        self.assertRaises(bf.FitterError, bf.read_sink, path)

    def test_foreign_schema_is_refused(self):
        document = doc([], [])
        document['schema'] = 'something.else'
        path = self.write_sink('foreign.jsonl', [document])
        self.assertRaises(bf.FitterError, bf.read_sink, path)

    def test_duplicate_vntr_line_is_refused_rather_than_doubling_a_denominator(self):
        document = doc([span('1', reached=8)], [row('D3_1', 1)])
        path = self.write_sink('dup.jsonl', [document, document])
        self.assertRaises(bf.FitterError, bf.load_capture, 'sampleX', path)


# --------------------------------------------------------------- the round trip


class TestOpportunityRoundTrip(TempDirTestCase):
    """Brief assertion 1: recompute every row's `opportunities` from `spans`."""

    def test_round_trip_agrees_on_a_hand_built_capture(self):
        spans = [span('1', reached=0b1000, count=5),
                 span('1', reached=0b1100, count=7),
                 span('2', reached=0b1000, count=11),
                 span('1', inserted=0b1000, count=3)]
        capture = bf.Capture('s', 'p', [doc(spans, [row('D3_1', 12),
                                                    row('I3_1_A_LEN1', 3)])])
        self.assertEqual([], capture.round_trip_failures())

    def test_a_wrong_stored_opportunities_is_reported_not_tolerated(self):
        spans = [span('1', reached=0b1000, count=5)]
        capture = bf.Capture('s', 'p', [doc(spans, [row('D3_1', 99)])])
        failures = capture.round_trip_failures()
        self.assertEqual(1, len(failures))
        self.assertEqual('D3_1', failures[0]['candidate'])
        self.assertEqual(99, failures[0]['stored'])
        self.assertEqual(5, failures[0]['recomputed'])

    def test_round_trip_holds_on_a_real_public_sink(self):
        path = os.path.join(PUBLIC_RUNS, 'example_66bf_hg19_subset',
                            'output', 'calibration.jsonl')
        if not os.path.isfile(path):
            self.skipTest('public sink not present')
        capture = bf.load_capture('example_66bf_hg19_subset', path)
        self.assertEqual([], capture.round_trip_failures())
        self.assertEqual(1014, len(capture.rows))


# ---------------------------------------------------------------- aggregation


class TestAggregationRule(TempDirTestCase):
    """Rulings 1-2: `k` unions `state_identities`; `N` is the state's OWN row."""

    def test_k_unions_identities_across_sibling_rows_and_never_sums_support(self):
        spans = [span('1', reached=0b1000, count=40)]
        rows = [row('D3_1', 40, {'D3_1': [[1, 0], [2, 0]]}, legacy_support=2),
                row('D3_1&D4_1', 40, {'D3_1': [[2, 0], [3, 0]],
                                      'D3_1&D4_1': [[9, 1]]}, legacy_support=1)]
        capture = bf.Capture('s', 'p', [doc(spans, rows)])
        # three distinct identities: (1,0), (2,0), (3,0) -- (2,0) counted once.
        self.assertEqual(3, capture.support_for('D3_1'))
        self.assertEqual(1, capture.support_for('D3_1&D4_1'))

    def test_n_is_the_states_own_row_when_it_has_one(self):
        spans = [span('1', reached=0b11000, count=40)]
        rows = [row('D3_1', 17, {'D3_1': [[1, 0]]}, legacy_support=1)]
        capture = bf.Capture('s', 'p', [doc(spans, rows)])
        # 17 is deliberately not the span recompute; the OWN ROW wins.
        self.assertEqual(17, capture.opportunities_for('D3_1'))

    def test_n_is_recomputed_from_spans_when_the_state_has_no_row(self):
        spans = [span('1', reached=0b11000, count=40),
                 span('1', reached=0b01000, count=6)]
        rows = [row('D3_1&D4_1', 40, {'D3_1&D4_1': [[1, 0]], 'D3_1': [[2, 0]]},
                    legacy_support=1)]
        capture = bf.Capture('s', 'p', [doc(spans, rows)])
        self.assertEqual(46, capture.opportunities_for('D3_1'))
        self.assertEqual(1, capture.support_for('D3_1'))

    def test_the_enumerated_state_set_is_candidates_plus_legacy_states(self):
        spans = [span('1', reached=0b11000, count=40)]
        rows = [row('D3_1&D4_1', 40, {'D3_1&D4_1': [[1, 0]], 'D4_1': [[2, 0]]})]
        capture = bf.Capture('s', 'p', [doc(spans, rows)])
        self.assertEqual(set(['D3_1&D4_1', 'D4_1']), capture.observed_states())

    def test_it_agrees_with_the_shipped_aggregate_evidence_on_every_row(self):
        path = os.path.join(PUBLIC_RUNS, 'example_b178_hg19_subset',
                            'output', 'calibration.jsonl')
        if not os.path.isfile(path):
            self.skipTest('public sink not present')
        capture = bf.load_capture('example_b178_hg19_subset', path)
        self.assertEqual([], capture.shipped_aggregation_disagreements())

    def test_k_greater_than_n_is_counted_and_never_clamped(self):
        spans = [span('1', reached=0b1000, count=1)]
        rows = [row('D3_1', 1, {'D3_1': [[1, 0], [2, 0], [3, 0]]}, legacy_support=3)]
        capture = bf.Capture('s', 'p', [doc(spans, rows)])
        self.assertEqual(3, capture.support_for('D3_1'))
        self.assertEqual(1, capture.opportunities_for('D3_1'))


# ------------------------------------------------------------- the decision log


class TestDecisionLog(TempDirTestCase):
    """Which states reach a decision site: `advntr/vntr_finder.py:472-531`."""

    LOG = '\n'.join([
        '2026-09-03 12:25:31,295 INFO:Frameshift Candidate and Occurrence D3_1: 2',
        '2026-09-03 12:25:31,295 INFO:Skipped due to too small number of occurrence D3_1: 2',
        '2026-09-03 12:25:31,295 INFO:Frameshift Candidate and Occurrence D9_2: 7',
        '2026-09-03 12:25:31,300 INFO:P-value: 1e-30',
        '2026-09-03 12:25:31,321 INFO:VID:25561, There is a mutation at D9_2',
        '2026-09-03 12:25:31,330 INFO:Frameshift Candidate and Occurrence I0_prefix_LEN1: 4',
        '2026-09-03 12:25:31,331 INFO:ID:25561, There is a mutation at I0_prefix_LEN1',
        '']) + '\n'

    def test_tested_states_are_the_logged_ones_that_cleared_the_support_floor(self):
        path = os.path.join(self.tmp, 'log')
        open(path, 'w').write(self.LOG)
        decisions = bf.parse_decision_log(path)
        self.assertEqual({'D9_2': 7, 'I0_prefix_LEN1': 4}, decisions['tested'])
        self.assertEqual({'D3_1': 2}, decisions['skipped'])

    def test_both_the_vid_and_the_id_call_prefixes_are_read(self):
        path = os.path.join(self.tmp, 'log')
        open(path, 'w').write(self.LOG)
        decisions = bf.parse_decision_log(path)
        self.assertEqual(set(['D9_2', 'I0_prefix_LEN1']), set(decisions['called']))

    def test_a_state_logged_but_never_resolved_is_an_error(self):
        path = os.path.join(self.tmp, 'log')
        open(path, 'w').write(
            'INFO:Skipped due to too small number of occurrence D3_1: 2\n')
        self.assertRaises(bf.FitterError, bf.parse_decision_log, path)

    def test_real_public_log_reproduces_the_runs_own_call_set(self):
        log = os.path.join(PUBLIC_RUNS, 'example_66bf_hg19_subset', 'work',
                           'log_example_66bf_hg19_subset.bam.log')
        if not os.path.isfile(log):
            self.skipTest('public log not present')
        decisions = bf.parse_decision_log(log)
        self.assertEqual(1011, len(decisions['logged']))
        self.assertEqual(942, len(decisions['skipped']))
        self.assertEqual(69, len(decisions['tested']))
        self.assertEqual(sorted(['I21_2_T_LEN1', 'I23_6_G_LEN1']),
                         sorted(decisions['called']))


class TestDecisionReplay(TempDirTestCase):

    def _model(self, states, default=0.5):
        path = os.path.join(self.tmp, 'bg.json')
        json.dump({'schema': SCHEMA, 'version': 1,
                   'provenance': 'test fixture', 'default_probability': default,
                   'states': states}, open(path, 'w'))
        return load_background_model(path)

    def test_a_strong_tail_is_called_and_a_weak_one_is_not(self):
        spans = [span('1', reached=0b1000, count=1000)]
        rows = [row('D3_1', 1000, {'D3_1': [[i, 0] for i in range(20)]},
                    legacy_support=20),
                row('D9_1', 1000, {'D9_1': [[i, 0] for i in range(3)]},
                    legacy_support=3)]
        capture = bf.Capture('s', 'p', [doc(spans, rows)])
        model = self._model({'D3_1': 1e-4, 'D9_1': 0.5})
        result = bf.replay_sample(capture, {'D3_1': 20, 'D9_1': 3}, model, 0.001)
        self.assertTrue(result['called'])
        self.assertEqual(['D3_1'], sorted(result['called_states']))

    def test_k_above_n_is_refused_exactly_as_the_caller_refuses_it(self):
        spans = [span('1', reached=0b1000, count=2)]
        rows = [row('D3_1', 2, {'D3_1': [[1, 0], [2, 0], [3, 0], [4, 0]]},
                    legacy_support=4)]
        capture = bf.Capture('s', 'p', [doc(spans, rows)])
        model = self._model({'D3_1': 1e-6})
        result = bf.replay_sample(capture, {'D3_1': 4}, model, 0.001)
        self.assertFalse(result['called'])
        self.assertEqual(1, result['k_exceeds_n'])

    def test_an_unlisted_state_scores_against_the_declared_default(self):
        spans = [span('1', reached=0b1000, count=1000)]
        rows = [row('D3_1', 1000, {'D3_1': [[i, 0] for i in range(20)]},
                    legacy_support=20)]
        capture = bf.Capture('s', 'p', [doc(spans, rows)])
        strict = bf.replay_sample(capture, {'D3_1': 20}, self._model({}, 0.9), 0.001)
        loose = bf.replay_sample(capture, {'D3_1': 20}, self._model({}, 1e-9), 0.001)
        self.assertFalse(strict['called'])
        self.assertTrue(loose['called'])

    def test_the_sample_call_is_any_state_called(self):
        spans = [span('1', reached=0b1000, count=1000)]
        rows = [row('D3_1', 1000, {'D3_1': [[1, 0], [2, 0], [3, 0]]},
                    legacy_support=3)]
        capture = bf.Capture('s', 'p', [doc(spans, rows)])
        self.assertFalse(bf.replay_sample(capture, {}, self._model({}, 1e-9),
                                          0.001)['called'])

    def test_a_tested_state_with_no_row_at_all_is_reported_and_not_called(self):
        spans = [span('1', reached=0b1000, count=1000)]
        capture = bf.Capture('s', 'p', [doc(spans, [])])
        result = bf.replay_sample(capture, {'D3_1': 5}, self._model({}, 1e-9), 0.001)
        self.assertFalse(result['called'])
        self.assertEqual(1, result['missing_rows'])


class TestBlockedFolds(unittest.TestCase):

    def test_a_pair_never_spans_two_folds(self):
        labels = []
        for index in range(10):
            labels.append(label('s%d_mut' % index, True, 'pair_%d' % index))
            labels.append(label('s%d_nrm' % index, False, 'pair_%d' % index))
        folds = bf.blocked_folds(labels, 5)
        self.assertEqual(5, len(folds))
        placement = {}
        for number, fold in enumerate(folds):
            for sample_id in fold:
                placement[sample_id] = number
        for index in range(10):
            self.assertEqual(placement['s%d_mut' % index],
                             placement['s%d_nrm' % index])
        self.assertEqual(20, sum(len(fold) for fold in folds))

    def test_it_never_returns_an_empty_fold(self):
        labels = [label('a', True, 'p1'), label('b', False, 'p1'),
                  label('c', True, 'p2'), label('d', False, 'p2')]
        folds = bf.blocked_folds(labels, 5)
        self.assertEqual(2, len(folds))
        for fold in folds:
            self.assertTrue(fold)

    def test_it_is_deterministic(self):
        labels = [label('s%d' % index, index % 2 == 0, 'pair_%d' % (index // 2))
                  for index in range(12)]
        self.assertEqual(bf.blocked_folds(labels, 5), bf.blocked_folds(labels, 5))


class TestLabels(TempDirTestCase):

    def test_only_the_requested_partition_is_read(self):
        path = os.path.join(self.tmp, 'manifest.json')
        json.dump({'samples': [
            label('keep', True, 'p1'),
            {'sample_id': 'drop', 'partition': 'locked-validation'},
        ]}, open(path, 'w'))
        kept = bf.load_labels(path, 'calibration')
        self.assertEqual(['keep'], [record['sample_id'] for record in kept])

    def test_a_missing_required_field_in_the_kept_partition_is_an_error(self):
        path = os.path.join(self.tmp, 'manifest.json')
        json.dump({'samples': [{'sample_id': 'x', 'partition': 'calibration'}]},
                  open(path, 'w'))
        self.assertRaises(bf.FitterError, bf.load_labels, path, 'calibration')

    def test_a_duplicate_sample_id_is_an_error(self):
        path = os.path.join(self.tmp, 'manifest.json')
        json.dump({'samples': [label('x', True, 'p1'), label('x', False, 'p2')]},
                  open(path, 'w'))
        self.assertRaises(bf.FitterError, bf.load_labels, path, 'calibration')


class TestAggregationFidelity(unittest.TestCase):
    """8b 5.2: for every tested state the row's `legacy_support` equals the log count."""

    def test_agreement_is_reported_as_no_divergence(self):
        spans = [span('1', reached=0b1000, count=100)]
        rows = [row('D3_1', 100, {'D3_1': [[1, 0], [2, 0], [3, 0]]}, legacy_support=5)]
        capture = bf.Capture('s', 'p', [doc(spans, rows)])
        self.assertEqual([], bf.aggregation_fidelity(capture, {'D3_1': 5}))

    def test_a_divergence_names_both_numbers(self):
        spans = [span('1', reached=0b1000, count=100)]
        rows = [row('D3_1', 100, {'D3_1': [[1, 0]]}, legacy_support=4)]
        capture = bf.Capture('s', 'p', [doc(spans, rows)])
        divergences = bf.aggregation_fidelity(capture, {'D3_1': 5})
        self.assertEqual(1, len(divergences))
        self.assertEqual(4, divergences[0]['legacy_support'])
        self.assertEqual(5, divergences[0]['log_count'])


class TestDiscrimination(unittest.TestCase):
    """8b 1.7: a build-time diagnostic that FLAGS, and never silently excludes."""

    def test_the_ratio_is_carrier_mean_k_over_control_mean_k(self):
        controls = [bf.Observation('c%d' % i, {'D3_1': (2, 100)}) for i in range(4)]
        carriers = [bf.Observation('m%d' % i, {'D3_1': (8, 100)}) for i in range(4)]
        ratios = bf.discrimination_ratios(carriers, controls, ['D3_1'])
        self.assertAlmostEqual(4.0, ratios['D3_1']['ratio'], places=12)
        self.assertTrue(ratios['D3_1']['flagged'])

    def test_a_non_discriminating_state_is_not_flagged(self):
        controls = [bf.Observation('c%d' % i, {'D28_5': (4, 100)}) for i in range(4)]
        carriers = [bf.Observation('m%d' % i, {'D28_5': (4, 100)}) for i in range(4)]
        ratios = bf.discrimination_ratios(carriers, controls, ['D28_5'])
        self.assertFalse(ratios['D28_5']['flagged'])


class TestAccuracyBenchImport(unittest.TestCase):

    def test_the_shipped_metrics_helpers_import_without_writing_anything(self):
        """The worktree is READ-ONLY for this task, and `imp.load_source` would drop a
        `.pyc` inside it. A stale `.pyc` from somebody else's earlier run may already be
        there (`*.pyc` is gitignored), so the check is that nothing in `scripts/` is
        created OR modified by the import, not that the directory is bare."""
        worktree = REPO_ROOT
        scripts = os.path.join(worktree, 'scripts')
        if not os.path.isdir(scripts):
            self.skipTest('worktree not present')
        before = dict((name, os.stat(os.path.join(scripts, name)).st_mtime)
                      for name in os.listdir(scripts))
        bench = bf.load_accuracy_bench(worktree)
        after = dict((name, os.stat(os.path.join(scripts, name)).st_mtime)
                     for name in os.listdir(scripts))
        self.assertTrue(hasattr(bench, 'wilson_ci'))
        self.assertTrue(hasattr(bench, 'mcnemar_exact'))
        self.assertTrue(hasattr(bench, 'build_report'))
        self.assertEqual(before, after)


class TestCrossValidation(TempDirTestCase):
    """8b 5.1: 5-fold blocked on `pair_id`, everything refit inside the training fold."""

    def _record(self, sample_id, truth, pair_id, k, n, baseline_call):
        spans = [span('1', reached=0b1000, count=n)]
        rows = [row('D3_1', n, {'D3_1': [[i, 0] for i in range(k)]}, legacy_support=k)]
        capture = bf.Capture(sample_id, 'p', [doc(spans, rows)])
        return {'sample_id': sample_id, 'truth': truth, 'pair_id': pair_id,
                'variant_class': 'vc', 'array_length': 60, 'capture': capture,
                'tested': {'D3_1': k} if k >= 3 else {},
                'baseline_call': baseline_call}

    def _records(self):
        """Ten pairs. The carriers sit where the multiplicity floor is the ONLY thing
        keeping them uncalled: `k = 4` on `N = 400` against a class floor solved at the
        controls' median `N = 200`, which is `P(K >= 4 | 200, p) = 0.001` -- so with the
        floor the tail is ~1.3e-2 and with it ablated it is ~0."""
        records = []
        for index in range(10):
            records.append(self._record('m%d' % index, True, 'pair_%d' % index,
                                        4, 400, True))
            records.append(self._record('n%d' % index, False, 'pair_%d' % index,
                                        0, 200, False))
        return records

    def test_a_fold_never_trains_on_its_own_held_out_samples(self):
        records = self._records()
        result = bf.cross_validate(records, ['D3_1'], bf.HYPERPARAMETERS, 5, 0.001)
        for fold in result['folds']:
            self.assertEqual(set(), set(fold['held_out']) & set(fold['trained_on']))
            self.assertTrue(fold['trained_on'])

    def test_every_sample_is_scored_exactly_once(self):
        records = self._records()
        result = bf.cross_validate(records, ['D3_1'], bf.HYPERPARAMETERS, 5, 0.001)
        scored = [entry['sample_id'] for entry in result['records']]
        self.assertEqual(sorted(scored),
                         sorted(entry['sample_id'] for entry in records))

    def test_the_scored_records_have_the_accuracy_bench_shape(self):
        records = self._records()
        result = bf.cross_validate(records, ['D3_1'], bf.HYPERPARAMETERS, 5, 0.001)
        for entry in result['records']:
            for field in ('sample_id', 'truth', 'baseline_call', 'candidate_call',
                          'variant_class', 'array_length'):
                self.assertIn(field, entry)

    def test_hyperparameters_are_identical_in_every_fold(self):
        records = self._records()
        result = bf.cross_validate(records, ['D3_1'], bf.HYPERPARAMETERS, 5, 0.001)
        seen = [fold['hyperparameters'] for fold in result['folds']]
        for entry in seen:
            self.assertEqual(bf.HYPERPARAMETERS['phi'], entry['phi'])
            self.assertEqual(bf.HYPERPARAMETERS['kprot'], entry['kprot'])

    def test_dropping_the_floor_is_an_ablation_and_changes_the_calls(self):
        records = self._records()
        with_floor = bf.cross_validate(records, ['D3_1'], bf.HYPERPARAMETERS, 5, 0.001)
        ablated = dict(bf.HYPERPARAMETERS)
        ablated['apply_floor'] = False
        without = bf.cross_validate(records, ['D3_1'], ablated, 5, 0.001)
        self.assertNotEqual(
            [entry['candidate_call'] for entry in with_floor['records']],
            [entry['candidate_call'] for entry in without['records']])


class TestFalsificationHelpers(unittest.TestCase):

    def test_the_label_shuffle_permutes_rates_without_changing_the_multiset(self):
        probabilities = dict(('D%d_1' % index, 1e-4 * (index + 1))
                             for index in range(20))
        shuffled = bf.shuffle_probabilities(probabilities, seed=7)
        self.assertEqual(sorted(probabilities.values()), sorted(shuffled.values()))
        self.assertEqual(sorted(probabilities), sorted(shuffled))
        self.assertNotEqual(probabilities, shuffled)

    def test_the_shuffle_is_reproducible_from_its_seed(self):
        probabilities = dict(('D%d_1' % index, 1e-4 * (index + 1))
                             for index in range(20))
        self.assertEqual(bf.shuffle_probabilities(probabilities, seed=7),
                         bf.shuffle_probabilities(probabilities, seed=7))
        self.assertNotEqual(bf.shuffle_probabilities(probabilities, seed=7),
                            bf.shuffle_probabilities(probabilities, seed=8))

    def test_truncation_ratios_compare_the_k_ge_3_population_with_the_whole_one(self):
        # four controls; two clear the legacy floor, two do not.
        samples = [
            {'observation': bf.Observation('a', {'D3_1': (5, 1000)}),
             'legacy_support': {'D3_1': 5}},
            {'observation': bf.Observation('b', {'D3_1': (4, 1000)}),
             'legacy_support': {'D3_1': 4}},
            {'observation': bf.Observation('c', {'D3_1': (0, 1000)}),
             'legacy_support': {}},
            {'observation': bf.Observation('d', {'D3_1': (1, 1000)}),
             'legacy_support': {'D3_1': 1}},
        ]
        ratios = bf.truncation_ratios(samples, ['D3_1'],
                                      bf.SETTINGS_MIN_SUPPORTING_READ_COUNT)
        entry = ratios['D3_1']
        self.assertAlmostEqual(9 / 2000.0, entry['truncated_rate'], places=12)
        self.assertAlmostEqual(10 / 4000.0, entry['untruncated_rate'], places=12)
        self.assertAlmostEqual(entry['truncated_rate'] / entry['untruncated_rate'],
                               entry['ratio'], places=12)

    def test_the_k_versus_n_regression_returns_a_through_origin_slope(self):
        observations = [bf.Observation('c%d' % index,
                                       {'D3_1': (index, index * 1000)})
                        for index in range(1, 30)]
        result = bf.k_versus_n_regression(observations, ['D3_1'], min_events=20)
        self.assertIn('D3_1', result)
        self.assertAlmostEqual(1e-3, result['D3_1']['slope'], places=12)

    def test_the_regression_skips_states_below_the_event_threshold(self):
        observations = [bf.Observation('c%d' % index, {'D3_1': (0, 1000)})
                        for index in range(5)]
        self.assertEqual({}, bf.k_versus_n_regression(observations, ['D3_1'],
                                                      min_events=20))


if __name__ == '__main__':
    unittest.main()

if __name__ == "__main__":
    unittest.main()
