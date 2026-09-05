"""Task 8c: Reporting, prediction checks, and diagnostic tables for background fitting."""
from advntr import settings
from advntr import background_fitter as bf

#: 8b 1.3's proxy-derived class denominators, quoted so prediction 5 has something to
#: compare against: "Measured floors at `kprot = 4` are ~3e-5 (RU2, median N ~ 8,589) up
#: to ~3e-3 (RU8, median N ~ 138)". These are the REJECTED proxy's numbers; the whole
#: point of prediction 5 is that the honest `N` should push the floors ~1.5x above them.
EIGHT_B_PROXY_N_MEDIAN = {'2': 8589, '8': 138}

#: Prediction 5's stop-signal band, from the pre-registration verbatim.
PREDICTION_5_BAND = (1.1, 3.0)

#: Prediction 4 says the median `X^2/df` is "near 1" without quantifying it. This band is
#: chosen HERE, before any cohort number is seen, so the verdict is not fitted to the
#: outcome; it is stated in the report as a ruling.
PREDICTION_4_BAND = (0.5, 1.5)

#: Prediction 3's band, from the pre-registration verbatim.
PREDICTION_3_BAND = (2.0, 8.0)

#: Pre-registered acceptance, restated. Not renegotiable.
ACCEPTANCE = {'mcnemar_p': 0.01, 'baseline_sensitivity_point': 0.880,
              'baseline_specificity_point': 0.870}

def _percentiles(values):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None

    def at(fraction):
        if len(values) == 1:
            return values[0]
        index = int(round(fraction * (len(values) - 1)))
        return values[index]

    return {'n': len(values), 'min': values[0], 'p25': at(0.25),
            'median': bf.median(values), 'p75': at(0.75), 'max': values[-1]}


def write_state_table(path, fit, discrimination):
    columns = ('state', 'class', 'tier', 'binding', 'sum_k', 'opportunity_mass_M_s',
               'contributing_control_samples', 'dispersion_df', 'dispersion_X2_df',
               'dispersion_eligible', 'pooled_rate', 'envelope', 'class_rate', 'floor',
               'p0', 'clamped', 'compound', 'components', 'carrier_mean_k',
               'control_mean_k', 'discrimination_ratio', 'discrimination_flagged')
    with open(path, 'w') as handle:
        handle.write('\t'.join(columns) + '\n')
        for state in sorted(fit.diagnostics):
            entry = fit.diagnostics[state]
            ratio = discrimination.get(state, {})
            values = [
                state, entry['class'], entry['tier'], entry['binding'],
                entry['sum_k'], entry['opportunity_mass'],
                entry['contributing_samples'], entry['dispersion_df'],
                '' if entry['dispersion'] is None else '%.6g' % entry['dispersion'],
                int(bool(entry['dispersion_eligible'])),
                '%.6g' % entry['pooled_rate'], '%.6g' % entry['envelope'],
                '%.6g' % fit.class_rates.get(entry['class'], 0.0),
                '%.6g' % entry['floor'], '%.17g' % entry['p0'],
                int(bool(entry['clamped'])), int(bool(entry['compound'])),
                '' if entry['components'] is None else entry['components'],
                '' if not ratio else '%.6g' % ratio['carrier_mean_k'],
                '' if not ratio else '%.6g' % ratio['control_mean_k'],
                '' if not ratio or ratio['ratio'] is None else '%.6g' % ratio['ratio'],
                '' if not ratio else int(bool(ratio['flagged'])),
            ]
            handle.write('\t'.join(str(value) for value in values) + '\n')


def write_markdown_report(path, report, cv_summary, verdicts, falsification_result):
    """A human-readable face on the JSON build report. Nothing new is computed here."""
    lines = []
    add = lines.append
    add('# Build report -- %s v%s' % (report['profile'], report['profile_version']))
    add('')
    if report['preregistration_overrides']:
        add('> **PRE-REGISTRATION OVERRIDDEN**: `%s`. This output is a code-path '
            'exercise, not a fit.' % report['preregistration_overrides'])
        add('')
    add('- partition: `%s`; samples %s (controls %s, carriers %s)'
        % (report['partition'], report['samples']['total'],
           report['samples']['controls'], report['samples']['carriers']))
    add('- capture root: `%s`' % report['capture_root'])
    add('- labels: `%s`' % report['labels'])
    add('- states emitted: %s; default_probability %.6g'
        % (report['key_origins']['observed'] + report['key_origins']['grammar'],
           report['default_probability']))
    add('- key origins: observed %s, grammar-generated %s'
        % (report['key_origins']['observed'], report['key_origins']['grammar']))
    add('- enumeration: grammar %s (grammar-only %s), observed %s '
        '(simple-in-grammar %s, compound %s, simple-outside-grammar %s)'
        % (report['enumeration']['grammar_total'],
           report['enumeration']['grammar_only'],
           report['enumeration']['observed_total'],
           report['enumeration']['observed_simple_in_grammar'],
           report['enumeration']['observed_compound'],
           len(report['enumeration']['observed_simple_outside_grammar'])))
    add('- tiers: %s' % report['tiers'])
    add('- screened (dispersion) states: %s -> `%s`'
        % (len(report['screened_states']), report['screened_states'][:10]))
    add('- clamped to MAX_PROBABILITY: %s' % len(report['clamped_states']))
    add('- mixed-class states: %s; degenerate classes: %s'
        % (len(report['mixed_class_states']), report['degenerate_classes']))
    add('- discrimination-flagged states: %s' % report['discrimination_flagged_count'])
    add('- depth distribution: %s' % report['depth_distribution'])
    add('')
    add('## Assertions')
    add('')
    for name in ('round_trip_failures', 'shipped_aggregation_disagreements',
                 'aggregation_fidelity_divergences', 'baseline_mismatches'):
        add('- `%s`: %s' % (name, len(report['assertions'][name])))
    add('- `k > N` by sample: %s'
        % sum(report['assertions']['k_exceeds_n_by_sample'].values()))
    add('- %s' % report['assertions']['subset_property_note'])
    add('')
    add('## Compound states (8b 1.6)')
    add('')
    add('- enumerated compounds: %s' % report['compound_states']['enumerated'])
    add('- tested compounds: %s' % len(report['compound_states']['tested_compounds']))
    add('- tested compounds falling to `default_probability`: %s'
        % len(report['compound_states']['tested_compounds_falling_to_default']))
    add('')
    add('## Cross-validation (blocked on pair_id)')
    add('')
    add('- folds: %s' % cv_summary['fold_count'])
    add('- sensitivity: %s' % cv_summary['sensitivity'])
    add('- specificity: %s' % cv_summary['specificity'])
    add('- McNemar carriers: %s' % cv_summary['mcnemar_carriers'])
    add('- McNemar controls: %s' % cv_summary['mcnemar_controls'])
    add('- acceptance gates: %s' % cv_summary['acceptance_gates'])
    add('- discordant samples: %s'
        % [entry['sample_id'] for entry in cv_summary['discordant']])
    add('')
    add('## Pre-registered predictions')
    add('')
    for name in sorted(verdicts):
        entry = verdicts[name]
        verdict = {True: 'HELD', False: 'MISSED', None: 'NOT COMPUTABLE'}[entry['held']]
        add('- **%s -- %s**: %s' % (name, verdict, entry['prediction']))
    add('')
    add('## Falsification (8b 5.2)')
    add('')
    add('- dispersion screen count: %s'
        % falsification_result['dispersion_screen_count'])
    add('- label shuffle metrics: %s'
        % falsification_result['label_shuffle']['report']['metrics'].get('candidate'))
    for name, summary in sorted(falsification_result['component_ablation'].items()):
        add('- ablation `%s`: sensitivity %s, specificity %s'
            % (name, summary['sensitivity']['candidate'],
               summary['specificity']['candidate']))
    add('- %s' % falsification_result['ablation_note'])
    for name, reason in sorted(falsification_result['not_run_here'].items()):
        add('- NOT RUN HERE -- `%s`: %s' % (name, reason))
    add('')
    with open(path, 'w') as handle:
        handle.write('\n'.join(lines) + '\n')

def summarise_cv(bench, result):
    """Pre-registered acceptance, computed separately for carriers and controls."""
    records = [dict((key, entry[key]) for key in
                    ('sample_id', 'truth', 'baseline_call', 'candidate_call',
                     'variant_class', 'array_length'))
               for entry in result['records']]
    report = bench.build_report(records, compare=True)
    summary = {'fold_count': result['fold_count'], 'accuracy_bench_report': report}

    def strata(subset):
        baseline_only = candidate_only = 0
        for entry in subset:
            baseline_correct = entry['baseline_call'] == entry['truth']
            candidate_correct = entry['candidate_call'] == entry['truth']
            if baseline_correct and not candidate_correct:
                baseline_only += 1
            elif candidate_correct and not baseline_correct:
                candidate_only += 1
        return bench.mcnemar_exact(baseline_only, candidate_only)

    carriers = [entry for entry in records if entry['truth']]
    controls = [entry for entry in records if not entry['truth']]
    summary['mcnemar_carriers'] = strata(carriers) if carriers else None
    summary['mcnemar_controls'] = strata(controls) if controls else None

    def rate(subset, field, want):
        hits = sum(1 for entry in subset if bool(entry[field]) == want)
        if not subset:
            return None
        low, high = bench.wilson_ci(hits, len(subset))
        return {'numerator': hits, 'denominator': len(subset),
                'estimate': hits / float(len(subset)), 'ci95': [low, high]}

    summary['sensitivity'] = {
        'baseline': rate(carriers, 'baseline_call', True),
        'candidate': rate(carriers, 'candidate_call', True)}
    summary['specificity'] = {
        'baseline': rate(controls, 'baseline_call', False),
        'candidate': rate(controls, 'candidate_call', False)}
    summary['discordant'] = [entry for entry in result['records']
                             if entry['baseline_call'] != entry['candidate_call']]
    gates = {}
    for name in ('carriers', 'controls'):
        test = summary['mcnemar_%s' % name]
        gates['mcnemar_%s_below_%s' % (name, ACCEPTANCE['mcnemar_p'])] = (
            None if test is None else test['p_value'] < ACCEPTANCE['mcnemar_p'])
    specificity = summary['specificity']
    gates['specificity_did_not_fall'] = (
        None if not (specificity['baseline'] and specificity['candidate'])
        else specificity['candidate']['estimate'] >= specificity['baseline']['estimate'])
    sensitivity = summary['sensitivity']
    gates['sensitivity_lower_bound_at_or_above_%s'
          % ACCEPTANCE['baseline_sensitivity_point']] = (
        None if not sensitivity['candidate']
        else sensitivity['candidate']['ci95'][0]
        >= ACCEPTANCE['baseline_sensitivity_point'])
    summary['acceptance_gates'] = gates
    return summary


def falsification(samples, states, fit, cv_summary, args, bench, hyperparameters):
    """The 8b 5.2 checks this fitter can run, and an honest note on the two it cannot."""
    controls = [sample for sample in samples if not sample['truth']]
    control_observations = [bf.CaptureObservation(sample['capture'])
                            for sample in controls]
    truncation_input = [
        {'observation': bf.CaptureObservation(sample['capture']),
         'legacy_support': dict((state, row['legacy_support'])
                                for state, row in sample['capture'].rows.items())}
        for sample in controls]

    shuffled = bf.shuffle_probabilities(fit.probabilities, args.shuffle_seed)
    shuffled_model = bf._StaticModel(shuffled, fit.default_probability)  # noqa: SLF001
    shuffled_records = []
    for sample in samples:
        replay = bf.replay_sample(sample['capture'], sample['tested'], shuffled_model,
                                  settings.INDEL_MUTATION_MIN_PVALUE)
        shuffled_records.append({'sample_id': sample['sample_id'],
                                 'truth': sample['truth'],
                                 'baseline_call': sample['baseline_call'],
                                 'candidate_call': bool(replay['called']),
                                 'variant_class': sample['variant_class'],
                                 'array_length': sample['array_length']})

    ablations = {}
    for name, override in (('no_screen', {'dispersion_threshold': float('inf')}),
                           ('no_floor', {'apply_floor': False}),
                           ('phi_1', {'phi': 1.0})):
        adjusted = dict(hyperparameters)
        adjusted.update(override)
        result = bf.cross_validate(samples, states, adjusted, args.folds,
                                   settings.INDEL_MUTATION_MIN_PVALUE)
        ablations[name] = summarise_cv(bench, result)

    regression = bf.k_versus_n_regression(control_observations, sorted(states),
                                          hyperparameters['dispersion_min_events'])
    truncation = bf.truncation_ratios(truncation_input, sorted(states),
                                      bf.SETTINGS_MIN_SUPPORTING_READ_COUNT)
    return {
        'label_shuffle': {
            'seed': args.shuffle_seed,
            'note': ('8b 5.2: if performance survives permuting the state -> rate '
                     'assignment, the per-state structure was never carrying the gain'),
            'report': bench.build_report(shuffled_records, compare=True),
        },
        'component_ablation': ablations,
        'ablation_note': ('"no_floor" cannot use p0 = 0, which the v1 loader refuses; '
                          'it uses ABLATION_EPSILON = %r instead'
                          % bf.ABLATION_EPSILON),
        'dispersion_screen_count': len(fit.screened_states),
        'dispersion_screened_states': fit.screened_states,
        'k_versus_n_regression': regression,
        'truncation': truncation,
        'not_run_here': {
            'byte_identical_result_tsv': (
                'the capture controller owns this check; it recorded '
                'result_tsv_matched_baseline per run and this fitter does not re-run '
                'adVNTR'),
            'gold_flag_on': (
                'requires running advntr genotype with --exact-frameshift-caller, '
                'which this task is forbidden to do'),
        },
    }


def predictions(samples, fit, cv_summary, falsification_result, hyperparameters):
    """The five pre-registered predictions, each HELD or MISSED with its number."""
    verdicts = {}

    screened = fit.screened_states
    baseline_false_positive_states = set()
    for sample in samples:
        if not sample['truth'] and sample['baseline_call']:
            baseline_false_positive_states.update(sample['called_by_baseline'])
    verdicts['1_dispersion_screen_fires_on_one_state'] = {
        'prediction': ('the dispersion screen fires on ONE state, and it is the state '
                       'behind the baseline false positives'),
        'screened_count': len(screened),
        'screened_states': screened,
        'baseline_false_positive_states': sorted(baseline_false_positive_states),
        'held': (len(screened) == 1
                 and set(screened) == baseline_false_positive_states),
    }

    no_floor = falsification_result['component_ablation']['no_floor']
    specificity = no_floor['specificity']['candidate']
    verdicts['2_specificity_falls_without_the_floor'] = {
        'prediction': ('without the multiplicity floor, calibration specificity falls '
                       'below the 0.870 baseline'),
        'specificity_without_floor': specificity,
        'held': (None if specificity is None
                 else specificity['estimate'] < ACCEPTANCE['baseline_specificity_point']),
    }

    ratios = [entry['ratio'] for entry in falsification_result['truncation'].values()
              if entry['ratio'] is not None]
    ratio_median = bf.median(ratios) if ratios else None
    clustering = []
    for state, entry in falsification_result['truncation'].items():
        if entry['truncated_rate'] and entry['truncated_samples']:
            denominator = (entry['truncated_denominator']
                           / float(entry['truncated_samples']))
            if denominator > 0:
                clustering.append(entry['truncated_rate']
                                  / (bf.SETTINGS_MIN_SUPPORTING_READ_COUNT
                                     / denominator))
    clustering_stats = _percentiles(clustering)
    clustering_median = clustering_stats['median'] if clustering_stats else None
    clause1_held = (None if ratio_median is None
                    else PREDICTION_3_BAND[0] <= ratio_median <= PREDICTION_3_BAND[1])
    clause2_held = (None if clustering_median is None
                    else 0.5 <= clustering_median <= 2.0)
    verdicts['3_truncation_bias'] = {
        'prediction': ('truncated/untruncated per-state rate ratios have a median in '
                       '2-8, and truncated rates cluster near '
                       'MIN_SUPPORTING_READ_COUNT / N'),
        'states_with_a_ratio': len(ratios),
        'median_ratio': ratio_median,
        'band': list(PREDICTION_3_BAND),
        'truncated_rate_over_3_over_N': clustering_stats,
        'clause1_median_ratio_in_band': clause1_held,
        'clause2_clustering_near_one': clause2_held,
        'held': (None if (clause1_held is None or clause2_held is None)
                 else (clause1_held and clause2_held)),
    }

    dispersions = [entry['dispersion'] for entry in fit.diagnostics.values()
                   if entry['dispersion'] is not None
                   and entry['sum_k'] >= hyperparameters['dispersion_min_events']]
    dispersion_median = bf.median(dispersions) if dispersions else None
    verdicts['4_median_dispersion_near_one'] = {
        'prediction': 'median X2/df across states with sum k >= 20 is near 1',
        'states': len(dispersions),
        'median': dispersion_median,
        'band_chosen_before_the_data': list(PREDICTION_4_BAND),
        'held': (None if dispersion_median is None
                 else PREDICTION_4_BAND[0] <= dispersion_median <= PREDICTION_4_BAND[1]),
    }

    comparisons = {}
    for klass, proxy_n in sorted(EIGHT_B_PROXY_N_MEDIAN.items()):
        if klass not in fit.class_floors:
            continue
        proxy_floor = bf.solve_floor(proxy_n, hyperparameters['kprot'],
                                     hyperparameters['floor_target'])
        comparisons[klass] = {
            'proxy_N_median_8b': proxy_n,
            'proxy_floor': proxy_floor,
            'captured_N_median': fit.class_n_median[klass],
            'captured_floor': fit.class_floors[klass],
            'ratio': fit.class_floors[klass] / proxy_floor,
        }
    ratio_values = [entry['ratio'] for entry in comparisons.values()]
    verdicts['5_floors_above_the_8b_proxy'] = {
        'prediction': ('floors recomputed on captured N sit roughly 1.5x above the 8b '
                       'proxy-derived floors; a ratio outside 1.1x-3x refutes the '
                       'transfer'),
        'band': list(PREDICTION_5_BAND),
        'per_class': comparisons,
        'held': (None if not ratio_values
                 else all(PREDICTION_5_BAND[0] <= value <= PREDICTION_5_BAND[1]
                          for value in ratio_values)),
    }
    return verdicts

