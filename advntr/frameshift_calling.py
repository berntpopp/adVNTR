"""Production logging adapter around the pure frameshift visit decisions.

Traversal is frozen before decisions, but consumed lazily: earlier candidates
still score before a later reached geometry lookup can fail. Repeat decisions
precede flank-boundary construction, as in the original read-loop implementation.
Only completed assessments are published; these receipts are not yet capture v2.
"""
import logging

from advntr import coverage_guard, exact_caller
from advntr.frameshift_traversal import CandidateTraversal, freeze_traversal, iter_visits
from advntr.frameshift_visit_decisions import assess_visit, coverage_basis, flank_boundaries
from advntr.run_context import runtime_value


def _assess_and_log(finder, plan, ru_bp_coverage, hmm_match_count, estimated_ru_count,
                    locus_coverage, background):
    """Keep prescore gates and log ordering without duplicating numeric decisions."""
    basis = None
    state, count, index = plan.state, plan.read_support, plan.repeat_unit_index
    if plan.disposition != 'outside-boundary':
        logging.info('Frameshift Candidate and Occurrence {}: {}'.format(state, count))
        if plan.disposition == 'insufficient-read-support':
            logging.info('Skipped due to too small number of occurrence {}: {}'.format(state, count))
        else:
            unit_length = hmm_match_count[index]
            total_bps = ru_bp_coverage[index]
            logging.info('Observed repeating base pairs in RU: %s' % total_bps)
            basis = coverage_basis(total_bps, unit_length, estimated_ru_count[index],
                                   finder.is_haploid, locus_coverage)
            logging.info('Average coverage for each base pair in RU: %s' % basis.mean_coverage)
    assessment = assess_visit(
        plan, basis, finder.frameshift_policy, finder.last_frameshift_opportunities, background,
        rare_fraction=runtime_value(finder, 'minimum_relative_ru_coverage'),
        legacy_error_rate=runtime_value(finder, 'legacy_error_rate'), legacy_statistic=finder.identify_frameshift)
    if assessment.disposition == 'rare-unit-coverage':
        logging.info('Candidate %s skipped: RU%s coverage %.2f collapsed below relative threshold' %
                     (state, index, assessment.mean_coverage))
    elif assessment.statistic is not None:
        if assessment.exact_assessment is not None:
            exact_caller.log_assessment(state, assessment.exact_assessment)
        else:
            logging.info('Sequencing error prob: %s' % assessment.sequencing_error_probability)
            logging.info('Frame-shift prob: %s' % assessment.frameshift_probability)
        logging.info('P-value: %s' % assessment.statistic.pvalue)
        if assessment.statistic.called:
            log_id = 'ID' if plan.site == 'prefix' else 'VID'
            logging.info(log_id + ':{}, There is a mutation at {}'.format(finder.reference_vntr.id, state))
    return assessment


def call_frameshift_candidates(finder, mutations, flank_mutations, ru_bp_coverage,
                              hmm_match_count, estimated_ru_count, reference_order, background):
    """Score the actual source candidates and publish complete ordered receipts."""
    traversal = freeze_traversal(mutations, flank_mutations)
    finder.last_frameshift_traversal = traversal
    logging.debug('sorted mutations: %s ' % list(traversal.repeat_candidates))
    locus_coverage = coverage_guard.compute_locus_coverage(
        ru_bp_coverage, hmm_match_count, estimated_ru_count, finder.is_haploid)
    assessments = []
    for plan in iter_visits(CandidateTraversal(traversal.repeat_candidates, ()), reference_order,
                            0, 0, finder.frameshift_policy):
        assessments.append(_assess_and_log(finder, plan, ru_bp_coverage, hmm_match_count,
                                           estimated_ru_count, locus_coverage, background))

    reference = finder.reference_vntr
    segments = reference.get_repeat_segments()
    suffix, prefix = flank_boundaries(segments, reference.left_flanking_region, reference.right_flanking_region,
                                      finder.hmm.read_length_used_to_build_model)
    finder.last_frameshift_boundaries = (suffix, prefix)
    logging.debug('TR region: {}*|{}...{}|*{}'.format(reference.left_flanking_region[-10:],
                                                   segments[0], segments[-1], reference.right_flanking_region[:10]))
    logging.debug('Suffix boundary {}'.format(suffix))
    logging.debug('Prefix boundary {}'.format(prefix))
    logging.debug('Prefix and suffix mutations: %s ' % flank_mutations)
    offset = len(assessments)
    for plan in iter_visits(CandidateTraversal((), traversal.flank_candidates), reference_order,
                            suffix, prefix, finder.frameshift_policy):
        plan = plan._replace(ordinal=offset + plan.ordinal)
        assessments.append(_assess_and_log(finder, plan, ru_bp_coverage, hmm_match_count,
                                           estimated_ru_count, locus_coverage, background))
    finder.last_frameshift_visits = tuple(assessments)
    return [(item.plan.state, item.plan.read_support, item.mean_coverage, item.statistic.pvalue)
            for item in assessments if item.statistic is not None and item.statistic.called]
