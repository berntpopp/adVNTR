"""Read quality and repeat-coverage gates with explicit resolved parameters."""
import logging


def is_low_quality_read(read, mapq_cutoff, base_quality_cutoff, maximum_low_quality_fraction):
    if read.mapq <= mapq_cutoff:
        logging.debug('Rejecting read for poor mapping quality')
        return True
    low_quality_base_pairs = [i for i, q in enumerate(read.query_qualities) if q < base_quality_cutoff]
    if len(low_quality_base_pairs) >= maximum_low_quality_fraction * len(read.query_qualities):
        logging.debug('Rejecting read for having so many low quality base pairs')
        return True
    maximum_low_quality_run = int(maximum_low_quality_fraction * len(read.query_qualities) / 4)
    for i in low_quality_base_pairs:
        passed = False
        for j in range(i+1, i+maximum_low_quality_run):
            if j not in low_quality_base_pairs:
                passed = True
                break
        if not passed:
            logging.debug('Rejecting read for having long run of low quality base pairs')
            return True
    return False


def update_repeat_coverage(visited_states, coverage, full_start, full_end, fully_covered, is_matching):
    """Count the same matching-state bases within the explicitly selected bounds."""
    start, end = (full_start, full_end) if fully_covered else (0, len(visited_states))
    for index in range(start, end):
        state = visited_states[index]
        if is_matching(state) and not state.endswith('fix'):
            coverage[state.split('_')[-1]] += 1
