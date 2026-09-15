"""Finite JSON receipts of the shared native decision result, never rounded calls."""
import math


def _number(value):
    if value is None or math.isnan(value):
        return None
    if math.isinf(value):
        raise ValueError('nonfinite statistic is not representable as a capture number')
    return float(value)


def visit_document(assessment):
    """Project one native visit; dispositions retain why a statistic is absent.

    Exact underflow retains its finite log tail. A genuinely zero exact tail is
    explicitly tagged, while inherited legacy NaN is a null with a noncall
    disposition. Decoder acceptance compares these receipts with a fresh native
    evaluation, so arbitrary nulls or dispositions cannot manufacture decisions.
    """
    statistic, exact = assessment.statistic, assessment.exact_assessment
    numeric = None
    if statistic is not None:
        log_tail = None
        if statistic.log_tail is not None:
            log_tail = ({'kind': 'negative-infinity', 'value': None} if statistic.log_tail == float('-inf')
                        else {'kind': 'finite', 'value': _number(statistic.log_tail)})
        numeric = {'called': bool(statistic.called), 'pvalue': _number(statistic.pvalue),
                   'log_tail': log_tail, 'disposition': statistic.disposition}
    return {'plan': dict(assessment.plan._asdict()), 'disposition': assessment.disposition,
            'mean_coverage': _number(assessment.mean_coverage), 'expected_indels': _number(assessment.expected_indels),
            'sequencing_error_probability': _number(assessment.sequencing_error_probability),
            'frameshift_probability': _number(assessment.frameshift_probability), 'statistic': numeric,
            'exact_evidence': None if exact is None else {
                'support': exact.support, 'opportunities': exact.opportunities, 'probability': _number(exact.probability)}}


def decision_warnings(assessments, ineligible_states):
    """Separate native exact-caller warnings from the stronger calibration audit."""
    result = []
    for assessment in assessments:
        if assessment.exact_assessment is not None and assessment.disposition in (
                'missing-opportunity-row', 'no-trials', 'no-occurrence-support', 'support-exceeds-opportunities'):
            result.append({'origin': 'caller', 'ordinal': assessment.plan.ordinal,
                           'state': assessment.plan.state, 'disposition': assessment.disposition})
    for state in ineligible_states:
        result.append({'origin': 'calibration-audit', 'ordinal': None, 'state': state,
                       'disposition': 'attribution-outside-trials'})
    return result
