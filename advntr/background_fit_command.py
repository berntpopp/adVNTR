"""Task 8c: CLI command advntr fit-background.

Ingests calibration sinks, fits the frozen background model, and emits
the v1 artifact, sidecar, diagnostic tables, and validation reports.
"""
import argparse
import json
import os
import sys

from advntr import settings
from advntr import background_fitter as bf
from advntr import background_fit_reports as bfr
from advntr.frameshift_background import BackgroundModelError, load_background_model

def add_fit_background_arguments(parser):
    parser.add_argument('--capture-root', required=True,
                        help='capture root holding runs/<sample>/output/<sink>')
    parser.add_argument('--labels', required=True,
                        help='labels manifest with a "samples" list')
    parser.add_argument('--partition', required=True,
                        help='the ONLY partition read; every other record is skipped '
                             'before any of its other fields is touched')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--profile', required=True,
                        help='profile name, carried in the provenance line')
    parser.add_argument('--profile-version', type=int, default=1)
    parser.add_argument('--sink-name', default='calibration.jsonl')
    parser.add_argument('--source-cohort', default='SIMULATED',
                        help='source cohort string carried in the provenance line, e.g. "SIMULATED"')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--insert-lengths', type=int, default=8,
                        help='largest _LEN<n> the grammar enumeration emits (8b 4.2)')
    parser.add_argument('--baseline-records', default=None,
                        help='JSONL of baseline calls; without it the baseline call is '
                             'taken from each run\'s result.json data_rows')
    default_worktree = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument('--worktree',
                        default=default_worktree,
                        help='read-only: only scripts/accuracy_bench.py is loaded')
    parser.add_argument('--design', default='unstated',
                        help='one line describing the capture design this profile '
                             'reflects')
    parser.add_argument('--note', action='append', default=[],
                        help='free-text note carried into the sidecar; repeatable')
    parser.add_argument('--shuffle-seed', type=int, default=20260905)
    parser.add_argument('--screen-min-samples', type=int, default=None,
                        help='SMOKE TEST ONLY. Overrides the pre-registered dispersion '
                             'screen eligibility (>= %d contributing control samples) '
                             'so the screen code path can be exercised on a capture '
                             'with fewer controls. Any artifact built with this set '
                             'carries PRE-REGISTRATION OVERRIDDEN at the front of its '
                             'provenance line and is not a fit.'
                             % bf.HYPERPARAMETERS['dispersion_min_samples'])


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_fit_background_arguments(parser)
    return parser.parse_args(argv)


def effective_hyperparameters(args):
    """The frozen estimator, or a self-identifying override for a code-path exercise."""
    hyperparameters = dict(bf.HYPERPARAMETERS)
    banner = ''
    overrides = {}
    if args.screen_min_samples is not None:
        overrides['dispersion_min_samples'] = args.screen_min_samples
        hyperparameters['dispersion_min_samples'] = args.screen_min_samples
        banner = ('PRE-REGISTRATION OVERRIDDEN (%s) -- NOT A FIT, A CODE-PATH EXERCISE '
                  '-- ' % ', '.join('%s=%r' % item for item in sorted(overrides.items())))
    return hyperparameters, banner, overrides


def _fail(message):
    raise bf.FitterError(message)

def discover_sinks(capture_root, labels, sink_name):
    """One sink per label, and no two labels resolving to one file."""
    mapping = []
    by_real_path = {}
    for record in labels:
        sample_id = record['sample_id']
        run_dir = os.path.join(capture_root, 'runs', sample_id)
        if not os.path.isdir(run_dir):
            _fail('no run directory for sample %r under %s' % (sample_id, capture_root))
        sink = os.path.join(run_dir, 'output', sink_name)
        if not os.path.isfile(sink):
            _fail('no sink for sample %r at %s' % (sample_id, sink))
        real = os.path.realpath(sink)
        if real in by_real_path:
            _fail('samples %r and %r both resolve to the sink %s; a sink must map to '
                  'exactly one sample or its partition is unrecoverable offline'
                  % (by_real_path[real], sample_id, real))
        by_real_path[real] = sample_id
        if os.path.basename(os.path.dirname(os.path.dirname(sink))) != sample_id:
            _fail('sink %s does not sit under runs/%s; sample identity comes from the '
                  'path and the manifest, never from the line' % (sink, sample_id))
        mapping.append((record, sink, run_dir))
    return mapping


def _baseline_from_result_json(run_dir):
    path = os.path.join(run_dir, 'result.json')
    if not os.path.isfile(path):
        _fail('no result.json at %s' % path)
    with open(path) as handle:
        document = json.load(handle)
    if 'data_rows' not in document:
        _fail('%s has no data_rows' % path)
    return int(document['data_rows'])


def _load_baseline_records(path):
    records = {}
    with open(path) as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError as error:
                _fail('%s:%d: invalid JSON (%s)' % (path, number, error))
            records[record['sample_id']] = record
    return records


def ingest(args, labels):
    """Every per-sample check, run before a single rate is estimated."""
    baseline_records = (_load_baseline_records(args.baseline_records)
                        if args.baseline_records else None)
    samples = []
    checks = {'round_trip_failures': [], 'shipped_disagreements': [],
              'aggregation_divergences': [], 'baseline_mismatches': [],
              'read_lengths': {}, 'repeat_units': None, 'repeat_unit_mismatch': [],
              'vntr_ids': set(), 'depth_by_sample': {}, 'span_counts': {},
              'row_counts': {}, 'tested_counts': {}, 'log_call_counts': {}}
    for record, sink, run_dir in discover_sinks(args.capture_root, labels,
                                                args.sink_name):
        sample_id = record['sample_id']
        capture = bf.load_capture(sample_id, sink)
        failures = capture.round_trip_failures()
        if failures:
            checks['round_trip_failures'].append({'sample_id': sample_id,
                                                  'failures': failures[:20],
                                                  'count': len(failures)})
        disagreements = capture.shipped_aggregation_disagreements()
        if disagreements:
            checks['shipped_disagreements'].append({'sample_id': sample_id,
                                                    'disagreements': disagreements[:20],
                                                    'count': len(disagreements)})
        log_path = os.path.join(run_dir, 'work', 'log_%s.bam.log' % sample_id)
        if not os.path.isfile(log_path):
            work = os.path.join(run_dir, 'work')
            if os.path.isdir(work):
                logs = sorted(n for n in os.listdir(work) if n.endswith('.log'))
                if len(logs) == 1:
                    log_path = os.path.join(work, logs[0])
        decisions = bf.parse_decision_log(log_path)
        divergences = bf.aggregation_fidelity(capture, decisions['tested'])
        if divergences:
            checks['aggregation_divergences'].append({'sample_id': sample_id,
                                                      'divergences': divergences[:20],
                                                      'count': len(divergences)})
        data_rows = _baseline_from_result_json(run_dir)
        log_calls = len(decisions['called'])
        baseline_call = data_rows > 0
        expected = baseline_call
        if baseline_records is not None:
            if sample_id not in baseline_records:
                _fail('no baseline record for %r in %s'
                      % (sample_id, args.baseline_records))
            expected = bool(baseline_records[sample_id]['baseline_call'])
        if not (expected == baseline_call == (log_calls > 0)):
            checks['baseline_mismatches'].append({
                'sample_id': sample_id, 'data_rows': data_rows,
                'log_calls': log_calls, 'baseline_record': expected})
        if checks['repeat_units'] is None:
            checks['repeat_units'] = decisions['repeat_units']
        elif decisions['repeat_units'] != checks['repeat_units']:
            checks['repeat_unit_mismatch'].append(sample_id)
        checks['read_lengths'][sample_id] = capture.read_length
        checks['vntr_ids'].update(capture.vntr_ids)
        checks['span_counts'][sample_id] = capture.span_count()
        checks['row_counts'][sample_id] = len(capture.rows)
        checks['tested_counts'][sample_id] = len(decisions['tested'])
        checks['log_call_counts'][sample_id] = log_calls
        checks['depth_by_sample'][sample_id] = _sample_depth(capture)
        samples.append({'sample_id': sample_id, 'truth': bool(record['truth']),
                        'pair_id': record['pair_id'],
                        'variant_class': record['variant_class'],
                        'array_length': record['array_length'],
                        'capture': capture, 'tested': decisions['tested'],
                        'logged': decisions['logged'],
                        'called_by_baseline': sorted(decisions['called']),
                        'baseline_call': baseline_call})
    checks['vntr_ids'] = sorted(checks['vntr_ids'])
    return samples, checks


def _sample_depth(capture):
    """The `avg_bp_coverage` of the sample's DOMINANT pattern.

    `avg_bp_coverage` is a per-pattern quantity carried on every row of that pattern
    (`advntr/frameshift_opportunities.py:_record`), and it is the `MeanCoverage` the
    caller prints. Measured on the public capture: it is constant within a pattern in
    every sample, and pattern 2 is dominant in all eight. "Dominant" is defined here as
    the pattern carrying the most span mass, which needs no model geometry.
    """
    mass = {}
    for pattern, entries in capture._spans_by_pattern.items():  # noqa: SLF001
        mass[pattern] = sum(count for _signature, count in entries)
    if not mass:
        return None
    dominant = max(sorted(mass), key=lambda pattern: mass[pattern])
    for row in capture.rows.values():
        if row['pattern_index'] == dominant:
            return row['avg_bp_coverage']
    return None


# ------------------------------------------------------------------ enumeration


def enumerate_states(samples, checks, insert_lengths):
    """Ruling 18 / 8b 4.2: the whole shipped grammar, plus every observed compound."""
    if checks['repeat_unit_mismatch']:
        _fail('the repeat-unit set differs across samples (%s); one artifact cannot '
              'describe two models' % ', '.join(checks['repeat_unit_mismatch']))
    repeat_units = checks['repeat_units'] or {}
    lengths = dict((pattern, len(sequence))
                   for pattern, sequence in repeat_units.items())
    if not lengths:
        _fail('no INFO:RU lines in any DEBUG log; the repeat-unit lengths that fix the '
              'admissible position range are not recoverable')
    flank_length = max(checks['read_lengths'].values())
    grammar = bf.grammar_states(lengths, flank_length=flank_length,
                               insert_lengths=insert_lengths)
    observed = set()
    for sample in samples:
        observed.update(sample['capture'].observed_states())
    states = grammar | observed
    origins = {
        'grammar_total': len(grammar),
        'grammar_only': len(grammar - observed),
        'observed_total': len(observed),
        'observed_simple_in_grammar': len(
            set(state for state in observed & grammar if not bf.is_compound(state))),
        'observed_simple_outside_grammar': sorted(
            state for state in observed - grammar if not bf.is_compound(state)),
        'observed_compound': len(
            set(state for state in observed if bf.is_compound(state))),
        'total': len(states),
        'repeat_unit_lengths': lengths,
        'flank_length': flank_length,
        'insert_lengths': insert_lengths,
    }
    return states, observed, origins, grammar

def prove_loader_acceptance(artifact_path, document, out_dir):
    """The artifact loads through the SHIPPED loader unchanged; a bad key is refused."""
    model = load_background_model(artifact_path)
    proof = {'loaded': True, 'version': model.version,
             'state_count': len(model.states),
             'default_probability': model.default_probability,
             'provenance_roundtrip': model.provenance == document['provenance'].strip(),
             'describe': model.describe()}
    if len(model.states) != len(document['states']):
        _fail('the loader returned %d states for an artifact carrying %d'
              % (len(model.states), len(document['states'])))
    mismatched = [state for state in document['states']
                  if model.states.get(state) != document['states'][state]]
    if mismatched:
        _fail('the loader changed %d values, first %r' % (len(mismatched),
                                                          mismatched[:3]))
    if model.default_probability != document['default_probability']:
        _fail('the loader changed default_probability')
    proof['values_identical'] = True

    victim = sorted(document['states'])[0]
    bad = dict(document)
    bad['states'] = dict(document['states'])
    bad['states'][victim + ' '] = bad['states'][victim]
    probe_path = os.path.join(out_dir, 'loader-refusal-probe.json')
    bf.write_json(probe_path, bad, compact=True)
    try:
        load_background_model(probe_path)
    except BackgroundModelError as error:
        proof['bad_key_probe'] = {'key': victim + ' ', 'refused': True,
                                  'message': str(error)[:400], 'path': probe_path}
    else:
        _fail('the shipped loader ACCEPTED a whitespace-padded key; the artifact cannot '
              'be trusted to be looked up byte-exactly')
    return proof

def run(args):
    if bf.SETTINGS_MIN_SUPPORTING_READ_COUNT != settings.MIN_SUPPORTING_READ_COUNT:
        _fail('this fitter assumes MIN_SUPPORTING_READ_COUNT = %d but the shipped '
              'settings say %d' % (bf.SETTINGS_MIN_SUPPORTING_READ_COUNT,
                                   settings.MIN_SUPPORTING_READ_COUNT))
    if abs(bf.HYPERPARAMETERS['floor_target']
           - settings.INDEL_MUTATION_MIN_PVALUE) > 0:
        _fail('the pre-registered floor target %r is not the shipped cutoff %r'
              % (bf.HYPERPARAMETERS['floor_target'],
                 settings.INDEL_MUTATION_MIN_PVALUE))
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    bench = bf.load_accuracy_bench(args.worktree)
    hyperparameters, banner, overrides = effective_hyperparameters(args)

    labels = bf.load_labels(args.labels, args.partition)
    samples, checks = ingest(args, labels)
    blocking = [key for key in ('round_trip_failures', 'shipped_disagreements',
                                'baseline_mismatches', 'aggregation_divergences')
                if checks[key]]
    if blocking:
        _fail('refusing to fit: %s. The sink and the run disagree; see the build '
              'report.' % ', '.join('%s on %d sample(s)' % (key, len(checks[key]))
                                    for key in blocking))

    states, observed, origins, grammar = enumerate_states(samples, checks, args.insert_lengths)
    controls = [sample for sample in samples if not sample['truth']]
    carriers = [sample for sample in samples if sample['truth']]
    if not controls:
        _fail('no control samples in partition %r; there is nothing to estimate a null '
              'from' % args.partition)
    control_observations = [bf.CaptureObservation(sample['capture'])
                            for sample in controls]
    carrier_observations = [bf.CaptureObservation(sample['capture'])
                            for sample in carriers]

    fit = bf.fit_background(control_observations, sorted(states), hyperparameters)
    discrimination = bf.discrimination_ratios(carrier_observations,
                                              control_observations, sorted(states))

    cardinality = {}
    for sample in samples:
        violations = sample['capture'].cardinality_violations(sample['tested'])
        cardinality[sample['sample_id']] = violations

    depth = bfr._percentiles(list(checks['depth_by_sample'].values()))
    context = {
        'samples': len(samples), 'controls': len(controls), 'carriers': len(carriers),
        'partition': args.partition,
        'source_cohort': args.source_cohort,
        'design': args.design,
        'depth': depth,
        'depth_median': None if depth is None else depth['median'],
        'excluded': ('carriers are excluded from every rate (8b 1.4); they are used '
                     'only for the discrimination diagnostic and for sensitivity in CV'),
        'provenance_pins': _provenance_pins(args.capture_root),
        'provenance_banner': banner,
    }
    provenance = bf.provenance_line(args.profile, args.profile_version, fit, context)
    document = bf.artifact_document(fit, provenance, observed=observed, grammar_states=grammar)
    key_origins = bf.validate_emitted_keys(document['states'], observed, grammar_states=grammar)

    artifact_path = os.path.join(args.out_dir, '%s.background.json' % args.profile)
    bf.write_json(artifact_path, document, compact=True)
    loader_proof = prove_loader_acceptance(artifact_path, document, args.out_dir)

    sidecar = bf.sidecar_document(args.profile, args.profile_version, fit, context,
                                  args.note)
    sidecar['key_origins'] = key_origins
    sidecar['enumeration'] = origins
    sidecar['capture'] = {'root': args.capture_root, 'sink_name': args.sink_name,
                          'vntr_ids': checks['vntr_ids'],
                          'read_lengths': sorted(set(checks['read_lengths'].values()))}
    sidecar['loader_proof'] = loader_proof
    sidecar['preregistration_overrides'] = overrides
    sidecar['preregistered_hyperparameters'] = bf.HYPERPARAMETERS
    sidecar['class_n_median_raw'] = fit.class_n_median_raw
    sidecar['class_n_median_clamped_to_kprot'] = fit.class_n_median_clamped_to_kprot
    bf.write_json(os.path.join(args.out_dir, '%s.sidecar.json' % args.profile), sidecar)

    cv_result = bf.cross_validate(samples, sorted(grammar), hyperparameters,
                                  args.folds, settings.INDEL_MUTATION_MIN_PVALUE)
    cv_summary = bfr.summarise_cv(bench, cv_result)
    cv_summary['folds'] = cv_result['folds']
    bf.write_json(os.path.join(args.out_dir, '%s.cv.json' % args.profile), cv_summary)

    falsification_result = bfr.falsification(samples, sorted(states), fit, cv_summary, args,
                                         bench, hyperparameters)
    bf.write_json(os.path.join(args.out_dir, '%s.falsification.json' % args.profile),
                  falsification_result)

    verdicts = bfr.predictions(samples, fit, cv_summary, falsification_result,
                           hyperparameters)
    bf.write_json(os.path.join(args.out_dir, '%s.predictions.json' % args.profile),
                  verdicts)

    bfr.write_state_table(os.path.join(args.out_dir, '%s.states.tsv' % args.profile),
                      fit, discrimination)

    tiers = {}
    for entry in fit.diagnostics.values():
        tiers[entry['tier']] = tiers.get(entry['tier'], 0) + 1
    flagged = sorted(state for state, entry in discrimination.items()
                     if entry['flagged'])
    build_report = {
        'profile': args.profile, 'profile_version': args.profile_version,
        'partition': args.partition, 'capture_root': args.capture_root,
        'labels': args.labels,
        'samples': {'total': len(samples), 'controls': len(controls),
                    'carriers': len(carriers)},
        'per_sample': {
            'spans': checks['span_counts'], 'rows': checks['row_counts'],
            'tested': checks['tested_counts'],
            'baseline_calls': checks['log_call_counts'],
            'depth': checks['depth_by_sample'],
            'read_length': checks['read_lengths'],
        },
        'assertions': {
            'round_trip_failures': checks['round_trip_failures'],
            'shipped_aggregation_disagreements': checks['shipped_disagreements'],
            'aggregation_fidelity_divergences': checks['aggregation_divergences'],
            'baseline_mismatches': checks['baseline_mismatches'],
            'k_exceeds_n_by_sample': dict(
                (sample_id, len(violations))
                for sample_id, violations in cardinality.items()),
            'k_exceeds_n_detail': dict(
                (sample_id, violations) for sample_id, violations
                in cardinality.items() if violations),
            'subset_property_note': (
                'only cardinality (k <= N) is checkable offline: the sink stores a '
                'COUNT per span signature, not the identities behind it, so set '
                'membership is unanswerable here and is not claimed. The set property '
                'is pinned in-process by tests/test_frameshift_calibration.py\'s '
                'TestSubsetObligation.'),
        },
        'enumeration': origins,
        'key_origins': key_origins,
        'tiers': tiers,
        'screened_states': fit.screened_states,
        'clamped_states': fit.clamped_states,
        'mixed_class_states': fit.mixed_class_states,
        'degenerate_classes': fit.degenerate_classes,
        'class_rates': fit.class_rates,
        'class_floors': fit.class_floors,
        'class_n_median': fit.class_n_median,
        'default_probability': fit.default_probability,
        'discrimination_flagged': flagged,
        'discrimination_flagged_detail': dict(
            (state, discrimination[state]) for state in flagged[:200]),
        'compound_states': {
            'enumerated': len([state for state in states if bf.is_compound(state)]),
            'tested_compounds': sorted(set(
                state for sample in samples for state in sample['tested']
                if bf.is_compound(state))),
        },
        'depth_distribution': depth,
        'hyperparameters': hyperparameters,
        'preregistration_overrides': overrides,
        'preregistered_hyperparameters': bf.HYPERPARAMETERS,
        'class_n_median_raw': fit.class_n_median_raw,
        'class_n_median_clamped_to_kprot': fit.class_n_median_clamped_to_kprot,
        'discrimination_flagged_count': len(flagged),
        'loader_proof': loader_proof,
        'state_table': '%s.states.tsv' % args.profile,
    }
    tested_compounds = build_report['compound_states']['tested_compounds']
    build_report['compound_states']['tested_compounds_falling_to_default'] = sorted(
        state for state in tested_compounds if state not in document['states'])
    bf.write_json(os.path.join(args.out_dir, '%s.build-report.json' % args.profile),
                  build_report)
    bfr.write_markdown_report(
        os.path.join(args.out_dir, '%s.build-report.md' % args.profile),
        build_report, cv_summary, verdicts, falsification_result)

    summary = {
        'artifact': artifact_path,
        'states': len(document['states']),
        'default_probability': fit.default_probability,
        'screened': len(fit.screened_states),
        'clamped': len(fit.clamped_states),
        'cv_folds': cv_summary['fold_count'],
        'predictions': dict((name, entry['held']) for name, entry in verdicts.items()),
    }
    print(json.dumps(summary, sort_keys=True, indent=1))
    return 0


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    return run(args)


def _provenance_pins(capture_root):
    """The capture's own recorded adVNTR provenance, copied verbatim, no sample id."""
    path = os.path.join(capture_root, 'control', 'manifest.json')
    if not os.path.isfile(path):
        return {'note': 'no control/manifest.json under the capture root'}
    with open(path) as handle:
        manifest = json.load(handle)
    pins = manifest.get('provenance', {})
    return {'binaries': pins.get('binaries'), 'models': pins.get('models'),
            'sources': pins.get('sources'), 'worktree': pins.get('worktree'),
            'controller_version': manifest.get('controller_version'),
            'controller_sha256': manifest.get('controller_sha256'),
            'source_manifest_sha256': (manifest.get('source_manifest') or {}).get(
                'sha256')}


if __name__ == '__main__':
    sys.exit(main())
