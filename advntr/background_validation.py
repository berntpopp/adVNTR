"""Task 8c: Validation, cross-validation, and falsification checks.

Implements flag-on decision replay, blocked k-fold cross-validation, and statistical
falsification diagnostics (label shuffle, k vs N regression, truncation ratios).
"""
import json
import math
import os

from advntr.background_capture import CaptureObservation
from advntr.background_estimator import FitterError, fit_background
from advntr.exact_caller import aggregate_evidence
from advntr.exact_tail import tail_below_cutoff

def replay_sample(capture, tested, model, cutoff):
    """Flag-on replay of one sample: exactly what `advntr/exact_caller.py` would decide.

    `tested` is the log-derived map of states that reached a decision site with count
    >= `settings.MIN_SUPPORTING_READ_COUNT`. For each: take `(k, N)` from the SHIPPED
    `aggregate_evidence`, refuse when it is `None` (no own row) and when `k > N`, and
    otherwise call iff the exact tail is below the cutoff. The sample-level call is
    "any state called", which is the rule the baseline used (`call = data_rows > 0`).
    """
    called_states = []
    details = []
    k_exceeds_n = 0
    missing_rows = 0
    for state in sorted(tested):
        evidence = aggregate_evidence(capture.rows, state)
        if evidence is None:
            missing_rows += 1
            details.append({'state': state, 'k': None, 'N': None, 'called': False,
                            'reason': 'no opportunity row'})
            continue
        support, opportunities = evidence
        if support > opportunities:
            k_exceeds_n += 1
            details.append({'state': state, 'k': support, 'N': opportunities,
                            'called': False, 'reason': 'k > N, refused by the guard'})
            continue
        probability = model.probability_for(state)
        called = tail_below_cutoff(support, opportunities, probability, cutoff)
        details.append({'state': state, 'k': support, 'N': opportunities,
                        'p0': probability, 'called': bool(called)})
        if called:
            called_states.append(state)
    return {'sample_id': capture.sample_id, 'called': bool(called_states),
            'called_states': called_states, 'k_exceeds_n': k_exceeds_n,
            'missing_rows': missing_rows, 'tested': len(tested), 'details': details}


class _StaticModel(object):
    """A `BackgroundModel` stand-in for CV folds, which never touch a file.

    Identical lookup semantics to `advntr/frameshift_background.py:probability_for`
    (`states.get(state, default)`), which is the only method `replay_sample` uses.
    """

    def __init__(self, states, default_probability):
        self.states = states
        self.default_probability = default_probability

    def probability_for(self, state):
        return self.states.get(state, self.default_probability)


def model_from_fit(fit):
    return _StaticModel(dict(fit.probabilities), fit.default_probability)


# ------------------------------------------------------------- cross-validation


REQUIRED_LABEL_FIELDS = ('sample_id', 'truth', 'partition', 'pair_id',
                         'variant_class', 'array_length')


def load_labels(path, partition):
    """The label records of ONE partition, filtered on `partition` before anything else.

    The holdout is one-shot and this fitter is not what evaluates it, so `partition` is
    the first and only field read off a record until it matches: a record of another
    partition is skipped without its `truth`, `pair_id` or any other field ever being
    touched (dispatch context, "Filter the `samples` list on `partition ==
    'calibration'` before you look at any other field").
    """
    if not os.path.isfile(path):
        raise FitterError('labels manifest %s: file not found' % path)
    try:
        with open(path) as handle:
            document = json.load(handle)
    except ValueError as error:
        raise FitterError('labels manifest %s: not valid JSON (%s)' % (path, error))
    if not isinstance(document, dict) or 'samples' not in document:
        raise FitterError('labels manifest %s: no top-level "samples" list' % path)
    kept = []
    seen = set()
    for index, record in enumerate(document['samples']):
        if not isinstance(record, dict):
            raise FitterError('labels manifest %s: record %d is not an object'
                              % (path, index + 1))
        if record.get('partition') != partition:
            continue
        missing = [field for field in REQUIRED_LABEL_FIELDS if field not in record]
        if missing:
            raise FitterError('labels manifest %s: record %d (%r) is missing %s'
                              % (path, index + 1, record.get('sample_id'),
                                 ', '.join(missing)))
        if not isinstance(record['truth'], bool):
            raise FitterError('labels manifest %s: record %r field "truth" must be a '
                              'boolean, got %r' % (path, record['sample_id'], record['truth']))
        if record['sample_id'] in seen:
            raise FitterError('labels manifest %s: duplicate sample_id %r'
                              % (path, record['sample_id']))
        seen.add(record['sample_id'])
        kept.append(record)
    if not kept:
        raise FitterError('labels manifest %s: no records in partition %r'
                          % (path, partition))
    return kept


def aggregation_fidelity(capture, tested):
    """8b 5.2: the row's `legacy_support` must equal the DEBUG log's own count.

    The log count is what the shipped floor gate at `advntr/vntr_finder.py:477` tested,
    and `legacy_support` is the sink's copy of the same number. A divergence means the
    sink and the run disagree about which candidates were tested, and nothing derived
    from either is usable on its own.
    """
    divergences = []
    for state in sorted(tested):
        row = capture.rows.get(state)
        if row is None:
            divergences.append({'state': state, 'legacy_support': None,
                                'log_count': tested[state],
                                'problem': 'no row for a tested state'})
            continue
        if row['legacy_support'] != tested[state]:
            divergences.append({'state': state,
                                'legacy_support': row['legacy_support'],
                                'log_count': tested[state],
                                'problem': 'legacy_support disagrees with the log'})
    return divergences


def discrimination_ratios(carrier_observations, control_observations, states,
                          flag_above=3.0):
    """`carrier mean k / control mean k` per state -- a FLAG, never a filter.

    8b 1.7: automatic exclusion is rejected because it would push a discriminating state
    to a conservative default and suppress it just as effectively as learning it would.
    """
    ratios = {}
    for state in states:
        carrier_values = [obs.evidence(state)[0] for obs in carrier_observations]
        control_values = [obs.evidence(state)[0] for obs in control_observations]
        carrier_mean = (sum(carrier_values) / float(len(carrier_values))
                        if carrier_values else 0.0)
        control_mean = (sum(control_values) / float(len(control_values))
                        if control_values else 0.0)
        if control_mean > 0.0:
            ratio = carrier_mean / control_mean
            flagged = ratio > flag_above
            infinite = False
        elif carrier_mean > 0.0:
            ratio = None
            flagged = True
            infinite = True
        else:
            ratio = None
            flagged = False
            infinite = False
        ratios[state] = {'carrier_mean_k': carrier_mean,
                         'control_mean_k': control_mean, 'ratio': ratio,
                         'ratio_infinite': infinite,
                         'flagged': flagged}
    return ratios


def load_accuracy_bench(worktree):
    """`scripts/accuracy_bench.py` as a module, without writing a byte to the worktree.

    `imp.load_source` would drop a `.pyc` beside the source, i.e. inside the repository,
    which this task may not touch. Compiling the text into a fresh module namespace has
    the same effect and writes nothing. `__name__` is not `'__main__'`, so the script's
    own entry point does not run.
    """
    import types
    path = os.path.join(worktree, 'scripts', 'accuracy_bench.py')
    if not os.path.isfile(path):
        raise FitterError('accuracy_bench not found at %s' % path)
    module = types.ModuleType('advntr_bench_accuracy')
    module.__file__ = path
    with open(path) as handle:
        source = handle.read()
    exec(compile(source, path, 'exec'), module.__dict__)
    return module


def shuffle_probabilities(probabilities, seed):
    """8b 5.2's label shuffle: permute the state -> rate assignment, nothing else.

    The multiset of rates is unchanged, so if performance survives this, the per-state
    STRUCTURE was never carrying the gain and the design is a global threshold in
    disguise. Seeded and reproducible: the seed goes in the falsification output.
    """
    import random
    states = sorted(probabilities)
    values = [probabilities[state] for state in states]
    generator = random.Random(seed)
    generator.shuffle(values)
    return dict(zip(states, values))


def truncation_ratios(samples, states, floor):
    """Prediction 3: the rate on the `legacy_support >= floor` population vs the whole.

    `samples` are dicts with an `observation` and a `legacy_support` map. The TRUNCATED
    rate uses only the control samples in which the state cleared the legacy support
    floor -- the population a naive calibration on the caller's own inputs would see --
    and the UNTRUNCATED rate uses every contributing control. 8b 1.3 predicts a median
    ratio in 2-8 and truncated rates clustering near `MIN_SUPPORTING_READ_COUNT / N`.
    """
    result = {}
    for state in states:
        trunc_k = trunc_n = whole_k = whole_n = 0
        truncated_samples = 0
        for sample in samples:
            support, opportunities = sample['observation'].evidence(state)
            if opportunities <= 0:
                continue
            whole_k += support
            whole_n += opportunities
            if sample['legacy_support'].get(state, 0) >= floor:
                truncated_samples += 1
                trunc_k += support
                trunc_n += opportunities
        truncated_rate = trunc_k / float(trunc_n) if trunc_n else None
        untruncated_rate = whole_k / float(whole_n) if whole_n else None
        ratio = None
        if truncated_rate is not None and untruncated_rate:
            ratio = truncated_rate / untruncated_rate
        result[state] = {'truncated_rate': truncated_rate,
                         'untruncated_rate': untruncated_rate, 'ratio': ratio,
                         'truncated_samples': truncated_samples,
                         'truncated_denominator': trunc_n,
                         'untruncated_denominator': whole_n}
    return result


def k_versus_n_regression(control_observations, states, min_events):
    """8b 5.2: is `k` proportional to `N` for the states that carry the estimate?

    Through-origin least squares `slope = sum(k_i N_i) / sum(N_i^2)`, which is the
    Bernoulli null's own prediction (`E[k_i] = N_i p`), plus the Pearson correlation.
    A slope far from the pooled rate, or `k` uncorrelated with `N`, refutes the form for
    that state. Only states at or above `min_events` are reported; below that the
    regression is noise.
    """
    result = {}
    for state in states:
        pairs = []
        for observation in control_observations:
            support, opportunities = observation.evidence(state)
            if opportunities > 0:
                pairs.append((support, opportunities))
        total_k = sum(support for support, _ in pairs)
        if total_k < min_events or len(pairs) < 3:
            continue
        sum_kn = sum(support * opportunities for support, opportunities in pairs)
        sum_nn = sum(opportunities * opportunities for _, opportunities in pairs)
        slope = sum_kn / float(sum_nn) if sum_nn else None
        pooled = total_k / float(sum(n for _, n in pairs))
        result[state] = {'slope': slope, 'pooled_rate': pooled,
                         'samples': len(pairs), 'sum_k': total_k,
                         'correlation': _pearson_correlation(pairs)}
    return result


def _pearson_correlation(pairs):
    count = len(pairs)
    if count < 2:
        return None
    mean_k = sum(support for support, _ in pairs) / float(count)
    mean_n = sum(n for _, n in pairs) / float(count)
    covariance = sum((support - mean_k) * (n - mean_n) for support, n in pairs)
    variance_k = sum((support - mean_k) ** 2 for support, _ in pairs)
    variance_n = sum((n - mean_n) ** 2 for _, n in pairs)
    if variance_k <= 0 or variance_n <= 0:
        return None
    return covariance / math.sqrt(variance_k * variance_n)


def cross_validate(records, states, hyperparameters, folds, cutoff):
    """5-fold CV blocked on `pair_id`, refitting EVERYTHING inside each training fold.

    8b 5.1: per-state rates, class rates, the dispersion screen and the floors are all
    refit on the training fold; the hyperparameters are held fixed across folds, because
    2.3 showed that selecting `phi` inside a fold is unstable. Each held-out sample is
    scored exactly once, by the model its own fold never saw.
    """
    by_id = dict((record['sample_id'], record) for record in records)
    buckets = blocked_folds(records, folds)
    scored = []
    fold_reports = []
    for number, held_out in enumerate(buckets):
        held = set(held_out)
        training = [record for record in records if record['sample_id'] not in held]
        control_observations = [CaptureObservation(record['capture'])
                                for record in training if not record['truth']]
        training_observed = set()
        has_capture = False
        for record in training:
            if 'capture' in record and hasattr(record['capture'], 'observed_states'):
                has_capture = True
                training_observed.update(record['capture'].observed_states())
        if has_capture:
            fold_states = sorted(set(states) | training_observed)
        else:
            fold_states = states
        fit = fit_background(control_observations, fold_states, hyperparameters)
        model = model_from_fit(fit)
        calls = {}
        for sample_id in held_out:
            record = by_id[sample_id]
            replay = replay_sample(record['capture'], record['tested'], model, cutoff)
            calls[sample_id] = replay
            scored.append({'sample_id': sample_id, 'truth': bool(record['truth']),
                           'baseline_call': bool(record['baseline_call']),
                           'candidate_call': bool(replay['called']),
                           'variant_class': record['variant_class'],
                           'array_length': record['array_length'],
                           'called_states': replay['called_states'],
                           'k_exceeds_n': replay['k_exceeds_n']})
        fold_reports.append({
            'fold': number, 'held_out': sorted(held_out),
            'trained_on': sorted(record['sample_id'] for record in training),
            'training_controls': len(control_observations),
            'hyperparameters': dict(hyperparameters),
            'screened_states': fit.screened_states,
            'default_probability': fit.default_probability,
            'calls': dict((sample_id, replay['called'])
                          for sample_id, replay in calls.items()),
        })
    scored.sort(key=lambda entry: entry['sample_id'])
    return {'records': scored, 'folds': fold_reports, 'fold_count': len(buckets)}


def blocked_folds(labels, folds, block_field='pair_id'):
    """`folds` groups of samples, blocked so a `pair_id` never spans two folds.

    8b 5.1: the two members of a pair share one simulated array, so splitting a pair
    across folds leaks. Blocks are assigned to folds in sorted order, largest first,
    to the currently smallest fold -- deterministic, and no random seed to record.
    """
    blocks = {}
    for record in labels:
        blocks.setdefault(record[block_field], []).append(record['sample_id'])
    ordered = sorted(blocks.items(), key=lambda item: (-len(item[1]), item[0]))
    folds = max(1, min(folds, len(ordered)))
    buckets = [[] for _ in range(folds)]
    for _block, members in ordered:
        target = min(range(folds), key=lambda index: (len(buckets[index]), index))
        buckets[target].extend(sorted(members))
    return [sorted(bucket) for bucket in buckets]
