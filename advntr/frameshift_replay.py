"""Packaged pure replay of completed captures using the production decision ABI."""
from copy import deepcopy

from advntr.capture_policy import calibrated_v2_domain_errors
from advntr.frameshift_background import BackgroundModel
from advntr.frameshift_capture_record import decode_capture, decode_background, digest
from advntr.frameshift_replay_policy import decode_replay_policy, require_replay_compatible
from advntr.frameshift_replay_decisions import evaluate_visits
from advntr.frameshift_visit_document import visit_document, decision_warnings


def replay_capture(capture_document, policy_document, background=None):
    """Verify original decisions, then evaluate only replay-supported changes.

    Inputs and output are JSON-compatible objects; this function reads no files,
    settings or process globals. The CLI separately binds the installed producer
    identity and complete manifest roster. An exact candidate's background never
    substitutes for the captured semantic background in the baseline parity check.
    """
    policy = decode_replay_policy(policy_document)
    if (policy.capture.caller_mode == 'exact') != (background is not None):
        raise ValueError('replay background presence differs from the requested caller mode')
    semantic = None
    if background is not None:
        if type(background) is not BackgroundModel:
            raise ValueError('replay requires a native loaded background model')
        semantic = {'schema': 'advntr.frameshift.background', 'version': background.version,
                    'default_probability': background.default_probability, 'states': dict(background.states)}
        background = decode_background(semantic)
    captured = decode_capture(capture_document)
    require_replay_compatible(captured.policy.capture, policy.capture)
    boundaries = capture_document['flank_boundaries']
    visits = evaluate_visits(captured.traversal, capture_document['reference_order'],
                              (boundaries['suffix_min_position'], boundaries['prefix_max_position']),
                              captured.geometry, captured.records, policy, background)
    calls = [{'state': visit.plan.state, 'read_support': visit.plan.read_support,
              'mean_coverage': visit.mean_coverage, 'pvalue': float(visit.statistic.pvalue)}
             for visit in visits if visit.statistic is not None and visit.statistic.called]
    return {'schema_version': 'advntr-frameshift-replay-result-v1', 'vntr_id': captured.model.vntr_id,
            'capture_record_sha256': digest(capture_document), 'policy_sha256': digest(policy_document),
            'capture_producer': deepcopy(capture_document['producer']),
            'capture_assets': deepcopy(capture_document['assets']),
            'loaded_background_sha256': None if semantic is None else digest(semantic),
            'baseline_parity': True, 'decision_visits': [visit_document(visit) for visit in visits], 'calls': calls,
            'warnings': decision_warnings(visits, captured.ineligible_states),
            'capture_audit': {'attribution_outside_trials': list(captured.ineligible_states),
                              'calibrated_policy_domain_errors': list(calibrated_v2_domain_errors(policy.capture))}}
