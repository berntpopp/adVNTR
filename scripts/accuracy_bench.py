#!/usr/bin/env python
"""Compute accuracy metrics from prepared truth/call JSONL records.

The input is one JSON object per line. Every record has these fields::

    {"sample_id": "sample-1", "truth": true, "baseline_call": true,
     "variant_class": "deletion", "array_length": 31}

Comparison mode additionally requires a boolean ``candidate_call`` on every
record. ``sample_id`` and ``variant_class`` are non-empty strings;
``array_length`` is a non-negative integer. Calls and truth are booleans.

Wilson intervals are used because accuracy proportions near 1.0 make the
normal interval collapse at the boundary. Exact McNemar tests are used because
baseline and candidate are evaluated on the same samples, so their outcomes
are paired. Carrier and control outcomes are always tested separately.

Reports contain sample identifiers only outside the repository. ``--out`` and
``ADVNTR_BENCH_OUT`` name an external directory that must already exist;
otherwise the directory is ``~/.cache/advntr-bench``. The harness never creates
or removes that directory. The published file is ``accuracy-report.json``.
Every stratified metric carries its own interpretation marker, based on that
caller's metric denominator rather than the stratum's total size.
"""
from __future__ import division

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile

from scipy.stats import binom_test, norm


try:
    INTEGER_TYPES = (int, long)
    STRING_TYPES = (basestring,)
except NameError:  # pragma: no cover - permits local inspection with Python 3
    INTEGER_TYPES = (int,)
    STRING_TYPES = (str,)


REPO_ROOT = os.path.realpath(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUTPUT_DIR = os.path.join('~', '.cache', 'advntr-bench')
OUTPUT_NAME = 'accuracy-report.json'
STRATUM_INTERPRETATION_MINIMUM = 20


def _validate_count(name, value):
    if isinstance(value, bool) or not isinstance(value, INTEGER_TYPES):
        raise TypeError('%s must be an integer count' % name)
    if value < 0:
        raise ValueError('%s must be non-negative' % name)


def wilson_ci(successes, total, confidence=0.95):
    """Return a Wilson score interval for integer ``successes`` of ``total``.

    ``total`` must be positive because an interval for an empty denominator is
    undefined. Callers represent that case as JSON null instead.
    """
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
    """Run two-sided exact McNemar on the two discordant-pair counts.

    ``baseline_only`` counts samples classified correctly only by baseline;
    ``candidate_only`` counts samples classified correctly only by candidate.
    Concordant pairs are deliberately absent from this API because exact
    McNemar conditions only on the total number of discordant pairs.
    """
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


def load_records(path):
    """Read non-blank JSONL records; validation belongs to ``build_report``."""
    records = []
    with open(path) as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError as error:
                raise ValueError('%s:%d: invalid JSON: %s'
                                 % (path, line_number, error))
            records.append(record)
    return records


def _repository_boundaries():
    """Return every worktree and the common Git directory, or fail closed."""
    try:
        worktree_output = subprocess.check_output(
            ['git', 'worktree', 'list', '--porcelain', '-z'], cwd=REPO_ROOT)
        common_git = subprocess.check_output(
            ['git', 'rev-parse', '--git-common-dir'], cwd=REPO_ROOT).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError('cannot establish repository output boundaries: %s'
                         % error)
    worktrees = [field[len('worktree '):]
                 for field in worktree_output.split('\0')
                 if field.startswith('worktree ')]
    if not worktrees or not common_git:
        raise ValueError('cannot establish repository output boundaries')
    if not os.path.isabs(common_git):
        common_git = os.path.join(REPO_ROOT, common_git)
    boundaries = worktrees + [common_git]
    return tuple(sorted(set(os.path.realpath(path) for path in boundaries)))


def _path_is_within(path, boundary):
    boundary_prefix = boundary.rstrip(os.sep) + os.sep
    return path == boundary or path.startswith(boundary_prefix)


def _resolved_external_path(path):
    resolved = os.path.realpath(os.path.abspath(os.path.expanduser(path)))
    for boundary in _repository_boundaries():
        if _path_is_within(resolved, boundary):
            raise ValueError('output path must be outside Git storage: %s' % path)
    return resolved


def _file_identity(stat_result):
    return stat_result.st_dev, stat_result.st_ino


def _validate_directory_fd(directory_fd):
    """A descriptor pins the inode, not ancestry, so compare both filesystem views."""
    anchored_dir = '/proc/self/fd/%d' % directory_fd
    resolved_dir = _resolved_external_path(anchored_dir)
    try:
        opened_identity = _file_identity(os.fstat(directory_fd))
        resolved_identity = _file_identity(os.stat(resolved_dir))
    except OSError as error:
        raise ValueError('cannot validate opened output directory: %s' % error)
    if opened_identity != resolved_identity:
        raise ValueError('opened output directory moved during validation')
    return anchored_dir, resolved_dir


def _unlink_matching(directory_path, name, identity):
    """Avoid removing an older leaf absent a concurrent same-UID name swap."""
    path = os.path.join(directory_path, name)
    try:
        if _file_identity(os.stat(path)) == identity:
            os.unlink(path)
    except OSError:
        pass


def _best_effort(action, *args):
    try:
        action(*args)
    except BaseException:
        pass


def publish_report(report, output_dir, records_path=None):
    """Atomically publish in a pre-existing, descriptor-validated directory.

    Boundary placement linearizes at the final descriptor-based validation
    after rename; success also requires the following final-file identity check.
    Moves observed by those checks fail, with best-effort descriptor-anchored
    cleanup before close. An open descriptor pins an inode, not its ancestry,
    and stat-then-unlink cannot lock a leaf name. A same-UID process can move the
    directory or swap a leaf after/between checks, including after return, so
    persistent placement and cleanup cannot be guaranteed in that threat model.
    """
    resolved_dir = _resolved_external_path(output_dir)
    if not os.path.exists(resolved_dir):
        raise ValueError('output directory must already exist: %s' % output_dir)
    if not os.path.isdir(resolved_dir):
        raise ValueError('output path is not a directory: %s' % output_dir)
    directory_fd = None
    temporary_fd = None
    temporary_name = None
    temporary_identity = None
    rename_attempted = False
    failure = None
    published_path = None
    try:
        open_flags = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0)
        directory_fd = os.open(resolved_dir, open_flags)
        anchored_dir, _validated_dir = _validate_directory_fd(directory_fd)
        output_path = os.path.join(anchored_dir, OUTPUT_NAME)
        resolved_output = _resolved_external_path(output_path)
        if records_path is not None:
            resolved_records = os.path.realpath(os.path.abspath(
                os.path.expanduser(records_path)))
            if resolved_records == resolved_output:
                raise ValueError('records path collides with final report destination')

        temporary_fd, temporary_path = tempfile.mkstemp(
            prefix='.accuracy-report-', suffix='.json', dir=anchored_dir)
        temporary_name = os.path.basename(temporary_path)
        temporary_identity = _file_identity(os.fstat(temporary_fd))
        _validate_directory_fd(directory_fd)
        handle = os.fdopen(temporary_fd, 'w')
        temporary_fd = None
        with handle:
            json.dump(report, handle, sort_keys=True, indent=2,
                      separators=(',', ': '))
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        _validate_directory_fd(directory_fd)
        rename_attempted = True
        os.rename(temporary_path, output_path)
        anchored_dir, validated_dir = _validate_directory_fd(directory_fd)
        if _file_identity(os.stat(output_path)) != temporary_identity:
            raise ValueError('published report identity changed during validation')
        published_path = os.path.join(validated_dir, OUTPUT_NAME)
    except BaseException:
        failure = sys.exc_info()
        if temporary_fd is not None:
            _best_effort(os.close, temporary_fd)
        if directory_fd is not None and temporary_identity is not None:
            anchored_dir = '/proc/self/fd/%d' % directory_fd
            _best_effort(
                _unlink_matching, anchored_dir, temporary_name,
                temporary_identity)
            if rename_attempted:
                _best_effort(
                    _unlink_matching, anchored_dir, OUTPUT_NAME,
                    temporary_identity)
    finally:
        if directory_fd is not None:
            try:
                os.close(directory_fd)
            except BaseException:
                if failure is None:
                    raise
    if failure is not None:
        raise failure[0], failure[1], failure[2]
    return published_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', required=True,
                        help='prepared truth/call records in JSONL format')
    parser.add_argument('--mode', choices=('baseline', 'compare'), required=True)
    parser.add_argument(
        '--out', help='external output directory; must already exist')
    args = parser.parse_args(argv)
    output_dir = (args.out or os.environ.get('ADVNTR_BENCH_OUT')
                  or DEFAULT_OUTPUT_DIR)
    try:
        records = load_records(args.records)
        report = build_report(records, compare=args.mode == 'compare')
        publish_report(report, output_dir, records_path=args.records)
    except (IOError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':
    sys.exit(main())
