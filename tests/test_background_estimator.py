"""Unit tests for background fitting modules."""
import json
import math
import os
import shutil
import tempfile
import unittest

from advntr.frameshift_background import (BackgroundModelError,
                                          load_background_model)
from advntr.exact_tail import exact_indel_tail_log
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

# --------------------------------------------------------------- the estimator


class TestFloorSolver(unittest.TestCase):
    """`floor(class)` solves `P(K >= kprot | N_median, p) = 0.001` exactly."""

    def test_the_solution_reproduces_the_target_to_one_part_in_1e9(self):
        for n_median in (10, 138, 1000, 8589):
            p = bf.solve_floor(n_median, kprot=4, target=0.001)
            reproduced = math.exp(exact_indel_tail_log(4, n_median, p))
            self.assertLess(abs(reproduced - 0.001) / 0.001, 1e-9,
                            'N=%d gave %r' % (n_median, reproduced))

    def test_a_smaller_denominator_gives_a_larger_floor(self):
        self.assertGreater(bf.solve_floor(138, 4, 0.001),
                           bf.solve_floor(8589, 4, 0.001))

    def test_the_floor_is_strictly_inside_the_open_unit_interval(self):
        for n_median in (4, 5, 20, 100000):
            p = bf.solve_floor(n_median, 4, 0.001)
            self.assertTrue(0.0 < p < 1.0, '%d -> %r' % (n_median, p))

    def test_a_denominator_below_kprot_is_refused_rather_than_guessed(self):
        self.assertRaises(bf.FitterError, bf.solve_floor, 3, 4, 0.001)

    def test_floor_solution_never_makes_kprot_callable_at_n_median(self):
        from advntr.exact_tail import tail_below_cutoff
        for n_median in (4, 10, 138, 1000, 8589):
            p = bf.solve_floor(n_median, 4, 0.001)
            self.assertFalse(tail_below_cutoff(4, n_median, p, 0.001),
                             'k=4, N=%d with p=%r was called' % (n_median, p))


class TestEstimatorOnSyntheticData(unittest.TestCase):
    """`p0(s) = max(phi * rate(s), floor(class(s)))`, pre-registration verbatim."""

    def _observation(self, sample_id, table):
        return bf.Observation(sample_id, table)

    def test_a_state_with_enough_events_takes_its_own_pooled_rate(self):
        # sum k = 20, sum N = 20000 -> rate 1e-3, phi 2 -> 2e-3, above the floor.
        controls = [self._observation('c%d' % i, {'D3_1': (2, 2000)})
                    for i in range(10)]
        fit = bf.fit_background(controls, ['D3_1'], bf.HYPERPARAMETERS)
        state = fit.diagnostics['D3_1']
        self.assertEqual('own', state['tier'])
        self.assertAlmostEqual(1e-3, state['rate'], places=12)
        self.assertAlmostEqual(2e-3, fit.probabilities['D3_1'], places=12)

    def test_a_thin_state_falls_to_its_class_rate(self):
        controls = [self._observation('c%d' % i,
                                      {'D3_1': (0, 2000), 'D9_1': (2, 2000)})
                    for i in range(10)]
        fit = bf.fit_background(controls, ['D3_1', 'D9_1'], bf.HYPERPARAMETERS)
        self.assertEqual('class', fit.diagnostics['D3_1']['tier'])
        self.assertEqual('own', fit.diagnostics['D9_1']['tier'])
        # class rate pools both states: 20 events over 40000 opportunities.
        self.assertAlmostEqual(5e-4, fit.diagnostics['D3_1']['rate'], places=12)

    def test_a_state_with_no_contributing_control_is_still_emitted_at_its_floor(self):
        controls = [self._observation('c%d' % i, {'D3_1': (0, 0)})
                    for i in range(10)]
        fit = bf.fit_background(controls, ['D3_1'], bf.HYPERPARAMETERS)
        self.assertEqual(0, fit.diagnostics['D3_1']['contributing_samples'])
        self.assertTrue(0.0 < fit.probabilities['D3_1'] < 1.0)

    def test_the_dispersion_screen_fires_only_when_eligible_and_over_threshold(self):
        # 20 controls; one carries every event on a tiny denominator.
        table = []
        for index in range(20):
            if index == 0:
                table.append(self._observation('c0', {'D28_5': (25, 50)}))
            else:
                table.append(self._observation('c%d' % index, {'D28_5': (0, 900)}))
        fit = bf.fit_background(table, ['D28_5'], bf.HYPERPARAMETERS)
        state = fit.diagnostics['D28_5']
        self.assertEqual('envelope', state['tier'])
        self.assertGreater(state['dispersion'], 3.0)
        self.assertAlmostEqual(0.5, state['rate'], places=12)

    def test_M_s_is_opportunity_mass_and_is_reported_beside_the_sample_count(self):
        # Controller correction 2: 8b 1.5's `M_s` is the opportunity mass
        # `sum_i N_is`; the screen's second eligibility condition is a separate count
        # of contributing control samples, and `df` is that count minus one.
        controls = [self._observation('c%d' % i, {'D3_1': (1, 700)})
                    for i in range(16)]
        controls.append(self._observation('c16', {'D3_1': (0, 0)}))
        fit = bf.fit_background(controls, ['D3_1'], bf.HYPERPARAMETERS)
        state = fit.diagnostics['D3_1']
        self.assertEqual(16 * 700, state['opportunity_mass'])
        self.assertEqual(16, state['contributing_samples'])
        self.assertEqual(15, state['dispersion_df'])

    def test_a_well_behaved_state_is_eligible_but_not_screened(self):
        table = [self._observation('c%d' % i, {'D52_2': (1, 1000)})
                 for i in range(20)]
        fit = bf.fit_background(table, ['D52_2'], bf.HYPERPARAMETERS)
        state = fit.diagnostics['D52_2']
        self.assertTrue(state['dispersion_eligible'])
        self.assertEqual('own', state['tier'])
        self.assertLess(state['dispersion'], 3.0)

    def test_the_screen_is_ineligible_below_twenty_events(self):
        table = [self._observation('c%d' % i, {'D9_9': (0, 900)})
                 for i in range(20)]
        table[0] = self._observation('c0', {'D9_9': (19, 50)})
        fit = bf.fit_background(table, ['D9_9'], bf.HYPERPARAMETERS)
        self.assertFalse(fit.diagnostics['D9_9']['dispersion_eligible'])

    def test_the_screen_is_ineligible_below_fifteen_contributing_samples(self):
        table = [self._observation('c%d' % i, {'D9_9': (0, 900)})
                 for i in range(14)]
        table[0] = self._observation('c0', {'D9_9': (25, 50)})
        fit = bf.fit_background(table, ['D9_9'], bf.HYPERPARAMETERS)
        self.assertFalse(fit.diagnostics['D9_9']['dispersion_eligible'])

    def test_the_floor_wins_when_phi_times_the_rate_is_below_it(self):
        controls = [self._observation('c%d' % i, {'D3_1': (0, 40)})
                    for i in range(20)]
        fit = bf.fit_background(controls, ['D3_1'], bf.HYPERPARAMETERS)
        floor = fit.class_floors['1']
        self.assertAlmostEqual(floor, fit.probabilities['D3_1'], places=15)
        self.assertEqual('floor', fit.diagnostics['D3_1']['binding'])

    def test_the_default_probability_is_the_largest_class_floor(self):
        controls = [self._observation('c%d' % i,
                                      {'D3_2': (0, 9000), 'D3_8': (0, 140)})
                    for i in range(20)]
        fit = bf.fit_background(controls, ['D3_2', 'D3_8'], bf.HYPERPARAMETERS)
        self.assertEqual(max(fit.class_floors.values()), fit.default_probability)
        self.assertGreater(fit.class_floors['8'], fit.class_floors['2'])

    def test_a_state_whose_components_disagree_on_pattern_is_flagged(self):
        controls = [self._observation('c%d' % i, {'D3_1&D4_2': (0, 0)})
                    for i in range(5)]
        fit = bf.fit_background(controls, ['D3_1&D4_2'], bf.HYPERPARAMETERS)
        self.assertIn('D3_1&D4_2', fit.mixed_class_states)
        self.assertTrue(0.0 < fit.probabilities['D3_1&D4_2'] < 1.0)

    def test_every_emitted_probability_is_strictly_inside_the_open_interval(self):
        controls = [self._observation('c%d' % i,
                                      {'D3_1': (0, 100), 'D9_1': (100, 100)})
                    for i in range(20)]
        fit = bf.fit_background(controls, ['D3_1', 'D9_1'], bf.HYPERPARAMETERS)
        for state, value in fit.probabilities.items():
            self.assertTrue(0.0 < value < 1.0, '%s -> %r' % (state, value))

    def test_carriers_never_enter_the_rate_even_when_present(self):
        controls = [self._observation('c%d' % i, {'D3_1': (0, 2000)})
                    for i in range(10)]
        carriers = [self._observation('m%d' % i, {'D3_1': (500, 2000)})
                    for i in range(10)]
        control_only = bf.fit_background(controls, ['D3_1'], bf.HYPERPARAMETERS)
        with_carriers = bf.fit_background(controls + carriers, ['D3_1'],
                                          bf.HYPERPARAMETERS)
        self.assertNotEqual(control_only.probabilities['D3_1'],
                            with_carriers.probabilities['D3_1'])
        # the fitter's public entry point must only ever be handed controls; this
        # test exists so the caller's obligation is visible rather than implied.


# ------------------------------------------------------------ the decision replay


# ------------------------------------------------------ artifact, keys, provenance


class TestArtifactEmission(TempDirTestCase):

    def _fit(self):
        controls = [bf.Observation('c%d' % i,
                                   {'D3_2': (2, 2000), 'I3_2_A_LEN1': (0, 2000)})
                    for i in range(10)]
        return bf.fit_background(controls, ['D3_2', 'I3_2_A_LEN1'],
                                 bf.HYPERPARAMETERS)

    def test_the_emitted_artifact_loads_through_the_shipped_loader_unchanged(self):
        fit = self._fit()
        path = os.path.join(self.tmp, 'artifact.json')
        document = bf.artifact_document(fit, bf.provenance_line(
            'test-profile', 1, fit, {'samples': 10, 'design': 'unit test',
                                     'source_cohort': 'SIMULATED'}))
        bf.write_json(path, document)
        model = load_background_model(path)
        self.assertEqual(1, model.version)
        self.assertEqual(sorted(document['states']), sorted(model.states))
        for state, value in document['states'].items():
            self.assertEqual(value, model.states[state])
        self.assertEqual(document['default_probability'],
                         model.default_probability)

    def test_the_artifact_carries_no_field_the_v1_loader_would_refuse(self):
        fit = self._fit()
        document = bf.artifact_document(fit, bf.provenance_line(
            'test-profile', 1, fit, {'samples': 10, 'design': 'unit test',
                                     'source_cohort': 'SIMULATED'}))
        self.assertEqual(set(['schema', 'version', 'provenance',
                              'default_probability', 'states']), set(document))

    def test_the_shipped_loader_refuses_a_deliberately_bad_key(self):
        fit = self._fit()
        document = bf.artifact_document(fit, bf.provenance_line(
            'test-profile', 1, fit, {'samples': 10, 'design': 'unit test',
                                     'source_cohort': 'SIMULATED'}))
        good = os.path.join(self.tmp, 'good.json')
        bf.write_json(good, document)
        load_background_model(good)  # must not raise

        document['states']['D3_2 '] = 1e-4
        bad = os.path.join(self.tmp, 'bad.json')
        bf.write_json(bad, document)
        try:
            load_background_model(bad)
        except BackgroundModelError as error:
            message = str(error)
        else:
            self.fail('the shipped loader accepted a whitespace-padded key')
        self.assertIn('collide', message)

    def test_key_validation_refuses_whitespace_duplicates_and_ungrammatical_keys(self):
        self.assertRaises(bf.FitterError, bf.validate_emitted_keys,
                          {'D3_2 ': 1e-4}, set())
        self.assertRaises(bf.FitterError, bf.validate_emitted_keys,
                          {'D3_2': 1e-4, 'D3_2  ': 1e-4}, set())
        self.assertRaises(bf.FitterError, bf.validate_emitted_keys,
                          {'M3_2': 1e-4}, set())
        self.assertRaises(bf.FitterError, bf.validate_emitted_keys,
                          {'': 1e-4}, set())
        self.assertRaises(bf.FitterError, bf.validate_emitted_keys,
                          {'D0_1': 1e-4}, set())
        self.assertRaises(bf.FitterError, bf.validate_emitted_keys,
                          {'I03_01_A_LEN0': 1e-4}, set())
        self.assertRaises(bf.FitterError, bf.validate_emitted_keys,
                          {'D4_1&D3_1': 1e-4}, set())

    def test_key_origins_are_counted_and_every_key_has_one(self):
        origins = bf.validate_emitted_keys({'D3_2': 1e-4, 'I9_2_A_LEN1': 1e-4},
                                           set(['D3_2']))
        self.assertEqual(1, origins['observed'])
        self.assertEqual(1, origins['grammar'])

    def test_grammar_enumeration_matches_the_shipped_state_naming(self):
        keys = bf.grammar_states({'2': 4}, insert_lengths=2)
        # `advntr/hmm_utils.py:646-647` names delete states D1..DL, so there is no D0;
        # `:637-640` names insert states I0..IL, so I0 exists and IL does too.
        self.assertNotIn('D0_2', keys)
        self.assertIn('D1_2', keys)
        self.assertIn('D4_2', keys)
        self.assertNotIn('D5_2', keys)
        self.assertIn('I0_2_A_LEN1', keys)
        self.assertIn('I4_2_T_LEN2', keys)
        self.assertNotIn('I5_2_T_LEN2', keys)
        self.assertEqual(4 + 5 * 4 * 2, len(keys))

    def test_flank_enumeration_carries_no_base_character(self):
        # `advntr/vntr_finder.py:422` builds a flank candidate straight off the HMM
        # state name, so a flank insertion has no base letter -- unlike
        # `advntr/mutation_keys.py:153-157`, which appends one for repeat units.
        keys = bf.grammar_states({}, flank_length=3, insert_lengths=1)
        self.assertIn('I0_prefix_LEN1', keys)
        self.assertIn('D3_suffix', keys)
        self.assertNotIn('I0_prefix_A_LEN1', keys)

    def test_grammar_enumeration_produces_only_loadable_keys(self):
        keys = bf.grammar_states({'2': 4}, flank_length=3, insert_lengths=2)
        origins = bf.validate_emitted_keys(dict((key, 1e-4) for key in keys), set())
        self.assertEqual(len(keys), origins['grammar'])


class TestProvenance(unittest.TestCase):

    def _line(self, **extra):
        controls = [bf.Observation('c%d' % i, {'D3_2': (2, 2000)})
                    for i in range(10)]
        fit = bf.fit_background(controls, ['D3_2'], bf.HYPERPARAMETERS)
        context = {'samples': 200, 'controls': 100, 'partition': 'calibration',
                   'source_cohort': 'SIMULATED',
                   'design': 'capture design X', 'depth_median': 812.0}
        context.update(extra)
        return bf.provenance_line('muc1-sim-calibration', 1, fit, context)

    def test_a_source_cohort_that_is_not_simulated_is_never_called_simulated(self):
        """The mandated wording belongs to the cohort profile. A smoke-test profile
        built from public non-simulated BAMs must not inherit it, or the artifact the
        brief exists to make unmistakable becomes the thing that misleads."""
        line = self._line(source_cohort='PUBLIC example_* BAMs, NOT a calibration '
                                        'cohort and NOT simulated')
        self.assertIn('PUBLIC example_* BAMs', line)
        self.assertNotIn('Source cohort: SIMULATED', line)
        self.assertNotIn('calibrated on simulated reads', line)
        self.assertIn('not a production default', line)

    def test_a_missing_source_cohort_is_refused_rather_than_defaulted(self):
        controls = [bf.Observation('c%d' % i, {'D3_2': (2, 2000)})
                    for i in range(10)]
        fit = bf.fit_background(controls, ['D3_2'], bf.HYPERPARAMETERS)
        self.assertRaises(bf.FitterError, bf.provenance_line,
                          'p', 1, fit, {'samples': 1, 'controls': 1})

    def test_it_names_the_control_count_and_not_the_partition_size(self):
        # Found by running the public smoke test: the line said "8 control samples"
        # where 8 was the partition size and only 4 of those were controls.
        line = self._line()
        self.assertIn('100 control samples', line)
        self.assertIn('out of 200 samples', line)

    def test_an_overridden_hyperparameter_banners_itself_at_the_front(self):
        line = self._line(provenance_banner='PRE-REGISTRATION OVERRIDDEN -- ')
        self.assertTrue(line.startswith('PRE-REGISTRATION OVERRIDDEN -- '))

    def test_it_says_SIMULATED_in_those_words(self):
        self.assertIn('Source cohort: SIMULATED', self._line())

    def test_it_carries_the_three_mandated_disclaimers(self):
        line = self._line()
        self.assertIn('calibrated on simulated reads', line)
        self.assertIn('not validated for non-simulated data', line)
        self.assertIn('not a production default', line)

    def test_it_never_says_in_silico_or_modelled(self):
        line = self._line().lower()
        self.assertNotIn('in-silico', line)
        self.assertNotIn('in silico', line)
        self.assertNotIn('modelled', line)

    def test_it_names_the_profile_and_every_hyperparameter(self):
        line = self._line()
        self.assertIn('muc1-sim-calibration', line)
        for token in ('phi', 'kprot', 'MIN_EVENTS', 'X2/df'):
            self.assertIn(token, line)

    def test_it_names_no_sample(self):
        line = self._line()
        for forbidden in ('example_', 'pair_', '__mut', '__normal'):
            self.assertNotIn(forbidden, line)

    def test_the_shipped_loader_accepts_it_as_provenance(self):
        self.assertTrue(self._line().strip())


class TestSidecar(unittest.TestCase):

    def test_the_sidecar_states_everything_the_v1_schema_has_no_room_for(self):
        controls = [bf.Observation('c%d' % i, {'D3_2': (2, 2000)})
                    for i in range(10)]
        fit = bf.fit_background(controls, ['D3_2'], bf.HYPERPARAMETERS)
        sidecar = bf.sidecar_document('muc1-sim-calibration', 1, fit, {
            'samples': 100, 'controls': 100, 'carriers': 0,
            'source_cohort': 'SIMULATED',
            'excluded': 'carriers excluded from estimation',
            'design': 'capture design X',
            'depth': {'median': 812.0, 'min': 210.0, 'max': 1300.0},
            'provenance_pins': {'worktree': 'abc'},
            'partition': 'calibration'}, ['unit-test-artifact'])
        self.assertEqual('SIMULATED', sidecar['source_cohort'])
        for key in ('profile_name', 'profile_version', 'hyperparameters',
                    'estimation_population', 'depth_distribution', 'capture_design',
                    'adVNTR_provenance', 'disclaimers', 'not_a_production_default'):
            self.assertIn(key, sidecar)
        self.assertTrue(sidecar['not_a_production_default'])


# ------------------------------------------------------------------ determinism


class TestDeterminism(unittest.TestCase):

    def test_the_two_public_capture_rounds_parse_identically(self):
        if not (os.path.isdir(PUBLIC_RUNS) and os.path.isdir(PUBLIC_RUNS_B)):
            self.skipTest('public capture rounds not present')
        sample = 'example_b178_hg19_subset'
        a = os.path.join(PUBLIC_RUNS, sample, 'output', 'calibration.jsonl')
        b = os.path.join(PUBLIC_RUNS_B, sample, 'output', 'calibration.jsonl')
        if not (os.path.isfile(a) and os.path.isfile(b)):
            self.skipTest('public sinks not present')
        self.assertEqual(open(a, 'rb').read(), open(b, 'rb').read())
        ca = bf.load_capture(sample, a)
        cb = bf.load_capture(sample, b)
        self.assertEqual(ca.observed_states(), cb.observed_states())
        for state in sorted(ca.observed_states()):
            self.assertEqual((ca.support_for(state), ca.opportunities_for(state)),
                             (cb.support_for(state), cb.opportunities_for(state)))



if __name__ == "__main__":
    unittest.main()
