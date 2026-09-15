"""Pure accuracy evaluation used by the installed fitter and benchmark script."""
from __future__ import division

import math

from scipy.stats import binom_test, norm


try:
    INTEGER_TYPES = (int, long)
    STRING_TYPES = (basestring,)
except NameError:  # pragma: no cover - permits local inspection with Python 3
    INTEGER_TYPES = (int,)
    STRING_TYPES = (str,)


STRATUM_INTERPRETATION_MINIMUM = 20


def _validate_count(name, value):
    if isinstance(value, bool) or not isinstance(value, INTEGER_TYPES):
        raise TypeError('%s must be an integer count' % name)
    if value < 0:
        raise ValueError('%s must be non-negative' % name)


def wilson_ci(successes, total, confidence=0.95):
    """Return a Wilson score interval for integer ``successes`` of ``total``."""
    _validate_count('successes', successes)
    _validate_count('total', total)
    if total == 0:
        raise ValueError('total must be positive')
    if successes > total:
        raise ValueError('successes cannot exceed total')
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise TypeError('confidence must be numeric')
    if math.isnan(confidence) or math.isinf(confidence):
        raise ValueError('confidence must be finite')
    if confidence <= 0.0 or confidence >= 1.0:
        raise ValueError('confidence must be between zero and one')

    z_value = float(norm.isf((1.0 - confidence) / 2.0))
    proportion = successes / total
    z_squared = z_value * z_value
    denominator = 1.0 + z_squared / total
    center = (proportion + z_squared / (2.0 * total)) / denominator
    margin = (z_value * math.sqrt(
        proportion * (1.0 - proportion) / total
        + z_squared / (4.0 * total * total)) / denominator)
    lower = 0.0 if successes == 0 else max(0.0, center - margin)
    upper = 1.0 if successes == total else min(1.0, center + margin)
    return lower, upper


def mcnemar_exact(baseline_only, candidate_only):
    """Run two-sided exact McNemar on the two discordant-pair counts."""
    _validate_count('baseline_only', baseline_only)
    _validate_count('candidate_only', candidate_only)
    discordant_total = baseline_only + candidate_only
    if discordant_total:
        p_value = float(binom_test(
            min(baseline_only, candidate_only), discordant_total,
            p=0.5, alternative='two-sided'))
    else:
        p_value = 1.0
    return {
        'baseline_only': baseline_only,
        'candidate_only': candidate_only,
        'discordant_total': discordant_total,
        'p_value': p_value,
    }


def _validate_records(records, compare):
    if not records:
        raise ValueError('records must not be empty')
    validated = []
    sample_ids = set()
    required = ('sample_id', 'truth', 'baseline_call',
                'variant_class', 'array_length')
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError('record %d must be a JSON object' % (index + 1))
        missing = [field for field in required if field not in record]
        if missing:
            raise ValueError('record %d is missing %s'
                             % (index + 1, ', '.join(missing)))
        sample_id = record['sample_id']
        if not isinstance(sample_id, STRING_TYPES) or not sample_id:
            raise ValueError('record %d sample_id must be a non-empty string'
                             % (index + 1))
        if sample_id in sample_ids:
            raise ValueError('duplicate sample_id: %s' % sample_id)
        sample_ids.add(sample_id)
        for field in ('truth', 'baseline_call'):
            if type(record[field]) is not bool:
                raise TypeError('record %s %s must be boolean'
                                % (sample_id, field))
        if compare:
            if 'candidate_call' not in record:
                raise ValueError('comparison record %s lacks candidate_call'
                                 % sample_id)
            if type(record['candidate_call']) is not bool:
                raise TypeError('record %s candidate_call must be boolean'
                                % sample_id)
        elif 'candidate_call' in record:
            raise ValueError('baseline mode records must omit candidate_call')
        variant_class = record['variant_class']
        if not isinstance(variant_class, STRING_TYPES) or not variant_class:
            raise ValueError('record %s variant_class must be a non-empty string'
                             % sample_id)
        array_length = record['array_length']
        _validate_count('record %s array_length' % sample_id, array_length)
        validated.append(record)

    carrier_count = sum(1 for record in validated if record['truth'])
    control_count = len(validated) - carrier_count
    if carrier_count == 0 or control_count == 0:
        raise ValueError('carrier and control partitions must both be non-empty')
    return validated


def _one_metric(successes, total, mark_interpretation=False):
    result = {'numerator': successes, 'denominator': total}
    if total:
        result['estimate'] = successes / total
        result['ci95'] = list(wilson_ci(successes, total))
    else:
        result['estimate'] = None
        result['ci95'] = None
    if mark_interpretation:
        result['interpretation'] = (
            'interpreted' if total >= STRATUM_INTERPRETATION_MINIMUM
            else 'report_only')
    return result


def _metrics(records, call_field, mark_interpretation=False):
    true_positive = sum(1 for record in records
                        if record['truth'] and record[call_field])
    false_negative = sum(1 for record in records
                         if record['truth'] and not record[call_field])
    true_negative = sum(1 for record in records
                        if not record['truth'] and not record[call_field])
    false_positive = sum(1 for record in records
                         if not record['truth'] and record[call_field])
    return {
        'sensitivity': _one_metric(true_positive,
                                   true_positive + false_negative,
                                   mark_interpretation),
        'specificity': _one_metric(true_negative,
                                   true_negative + false_positive,
                                   mark_interpretation),
        'ppv': _one_metric(true_positive, true_positive + false_positive,
                           mark_interpretation),
        'npv': _one_metric(true_negative, true_negative + false_negative,
                           mark_interpretation),
    }


def _metrics_by_caller(records, compare, mark_interpretation=False):
    metrics = {'baseline': _metrics(
        records, 'baseline_call', mark_interpretation)}
    if compare:
        metrics['candidate'] = _metrics(
            records, 'candidate_call', mark_interpretation)
    return metrics


def _stratify(records, field, compare):
    values = sorted(set(record[field] for record in records))
    result = []
    for value in values:
        members = [record for record in records if record[field] == value]
        result.append({
            'value': value,
            'n': len(members),
            'metrics': _metrics_by_caller(
                members, compare, mark_interpretation=True),
        })
    return result


def _paired_counts(records):
    baseline_only = 0
    candidate_only = 0
    for record in records:
        baseline_correct = record['baseline_call'] == record['truth']
        candidate_correct = record['candidate_call'] == record['truth']
        if baseline_correct and not candidate_correct:
            baseline_only += 1
        elif candidate_correct and not baseline_correct:
            candidate_only += 1
    return mcnemar_exact(baseline_only, candidate_only)


def _discordance(record):
    baseline_correct = record['baseline_call'] == record['truth']
    if baseline_correct:
        direction = 'baseline_correct_to_candidate_incorrect'
        cause = ('candidate_false_negative' if record['truth']
                 else 'candidate_false_positive')
    else:
        direction = 'baseline_incorrect_to_candidate_correct'
        cause = ('candidate_fixed_false_negative' if record['truth']
                 else 'candidate_fixed_false_positive')
    return {
        'sample_id': record['sample_id'],
        'truth': record['truth'],
        'baseline_call': record['baseline_call'],
        'candidate_call': record['candidate_call'],
        'variant_class': record['variant_class'],
        'array_length': record['array_length'],
        'direction': direction,
        'cause': cause,
    }


def _comparison(records, metrics):
    carriers = [record for record in records if record['truth']]
    controls = [record for record in records if not record['truth']]
    carrier_test = _paired_counts(carriers)
    control_test = _paired_counts(controls)
    discordances = sorted(
        (_discordance(record) for record in records
         if record['baseline_call'] != record['candidate_call']),
        key=lambda item: item['sample_id'])
    baseline_sensitivity = metrics['baseline']['sensitivity']['estimate']
    candidate_sensitivity_lower = metrics['candidate']['sensitivity']['ci95'][0]
    return {
        'mcnemar': {
            'carriers': carrier_test,
            'controls': control_test,
        },
        'discordances': discordances,
        'decision': {
            'alpha': 0.01,
            'carrier_p_below_0_01': carrier_test['p_value'] < 0.01,
            'control_p_below_0_01': control_test['p_value'] < 0.01,
            'specificity_fell': (
                metrics['candidate']['specificity']['estimate']
                < metrics['baseline']['specificity']['estimate']),
            'candidate_sensitivity_lower_ci_below_baseline_point': (
                candidate_sensitivity_lower < baseline_sensitivity),
        },
    }


def build_report(records, compare=False):
    """Build a baseline or paired-comparison report without writing files."""
    records = _validate_records(records, compare)
    metrics = _metrics_by_caller(records, compare)
    report = {
        'schema_version': 1,
        'mode': 'comparison' if compare else 'baseline',
        'sample_count': len(records),
        'partitions': {
            'carriers': sum(1 for record in records if record['truth']),
            'controls': sum(1 for record in records if not record['truth']),
        },
        'metrics': metrics,
        'strata': {
            'variant_class': _stratify(records, 'variant_class', compare),
            'array_length': _stratify(records, 'array_length', compare),
        },
    }
    if compare:
        report['comparison'] = _comparison(records, metrics)
    return report
