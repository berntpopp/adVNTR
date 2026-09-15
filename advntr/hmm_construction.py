"""Working enhanced HMM construction with an explicit per-build error rate.

State/transition order and arithmetic association are inherited unchanged. Public
compatibility wrappers live in hmm_utils; this module never reads mutable settings.
Alignment reconstruction is supplied by the wrapper when vpaths are provided, so
construction does not import the path-analysis module that delegates to it.

The two legacy from_matrix builders remain in hmm_utils and unsupported. Their
presence is not a claim that the enhanced Model implements that obsolete surface.
"""
import math
from math import log

from advntr.pattern_clustering import get_pattern_clusters
from advntr.profile_hmm import build_profile_hmm_for_repeats, build_profile_hmm_pseudocounts_for_alignment
from advntr.profiler import time_usage
from hmm.hmm import Model
from hmm.base import DiscreteDistribution, State


def _validate_error_rate(value):
    if (not isinstance(value, float) or math.isnan(value) or math.isinf(value)
            or not 0.0 <= value <= 1.0):
        raise ValueError('maximum_error_rate must be a finite float in [0, 1]')


def _alignment(vpaths, alignment_from_reads):
    if not callable(alignment_from_reads):
        raise ValueError('vpaths require an explicit alignment_from_reads function')
    return alignment_from_reads(vpaths)


@time_usage
def get_prefix_matcher_hmm(pattern, maximum_error_rate):
    _validate_error_rate(maximum_error_rate)
    model = Model(name="Prefix Matcher HMM Model")
    insert_distribution = DiscreteDistribution({'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25})
    insert_states = []
    match_states = []
    delete_states = []
    hmm_name = 'prefix'
    for i in range(len(pattern) + 1):
        insert_states.append(State(insert_distribution, name='I%s_%s' % (i, hmm_name)))

    for i in range(len(pattern)):
        distribution_map = dict({'A': 0.01, 'C': 0.01, 'G': 0.01, 'T': 0.01})
        distribution_map[pattern[i]] = 0.97
        match_states.append(State(DiscreteDistribution(distribution_map), name='M%s_%s' % (str(i + 1), hmm_name)))

    for i in range(len(pattern)):
        delete_states.append(State(None, name='D%s_%s' % (str(i + 1), hmm_name)))

    unit_start = State(None, name='prefix_start_%s' % hmm_name)
    unit_end = State(None, name='prefix_end_%s' % hmm_name)
    model.add_states(insert_states + match_states + delete_states + [unit_start, unit_end])
    last = len(delete_states) - 1

    model.add_transition(model.start, unit_start, 1)

    model.add_transition(unit_end, model.end, 1)

    insert_error = maximum_error_rate * 2 / 5
    delete_error = maximum_error_rate * 1 / 5
    model.add_transition(unit_start, match_states[0], 1 - insert_error - delete_error)
    model.add_transition(unit_start, delete_states[0], delete_error)
    model.add_transition(unit_start, insert_states[0], insert_error)

    model.add_transition(insert_states[0], insert_states[0], insert_error)
    model.add_transition(insert_states[0], delete_states[0], delete_error)
    model.add_transition(insert_states[0], match_states[0], 1 - insert_error - delete_error)

    model.add_transition(delete_states[last], unit_end, 1 - insert_error)
    model.add_transition(delete_states[last], insert_states[last + 1], insert_error)

    model.add_transition(match_states[last], unit_end, 1 - insert_error)
    model.add_transition(match_states[last], insert_states[last + 1], insert_error)

    model.add_transition(insert_states[last + 1], insert_states[last + 1], insert_error)
    model.add_transition(insert_states[last + 1], unit_end, 1 - insert_error)

    for i in range(0, len(pattern)):
        model.add_transition(match_states[i], insert_states[i + 1], insert_error)
        model.add_transition(delete_states[i], insert_states[i + 1], insert_error)
        model.add_transition(insert_states[i + 1], insert_states[i + 1], insert_error)
        if i < len(pattern) - 1:
            model.add_transition(insert_states[i + 1], match_states[i + 1], 1 - insert_error - delete_error)
            model.add_transition(insert_states[i + 1], delete_states[i + 1], delete_error)

            model.add_transition(match_states[i], match_states[i + 1], 1 - insert_error - delete_error - 0.01)
            model.add_transition(match_states[i], delete_states[i + 1], delete_error)
            model.add_transition(match_states[i], unit_end, 0.01)

            model.add_transition(delete_states[i], delete_states[i + 1], delete_error)
            model.add_transition(delete_states[i], match_states[i + 1], 1 - insert_error - delete_error)

    model.bake(merge=None)

    return model


@time_usage
def get_suffix_matcher_hmm(pattern, maximum_error_rate):
    _validate_error_rate(maximum_error_rate)
    model = Model(name="Suffix Matcher HMM Model")
    insert_distribution = DiscreteDistribution({'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25})
    insert_states = []
    match_states = []
    delete_states = []
    hmm_name = 'suffix'
    for i in range(len(pattern) + 1):
        insert_states.append(State(insert_distribution, name='I%s_%s' % (i, hmm_name)))

    for i in range(len(pattern)):
        distribution_map = dict({'A': 0.01, 'C': 0.01, 'G': 0.01, 'T': 0.01})
        distribution_map[pattern[i]] = 0.97
        match_states.append(State(DiscreteDistribution(distribution_map), name='M%s_%s' % (str(i + 1), hmm_name)))

    for i in range(len(pattern)):
        delete_states.append(State(None, name='D%s_%s' % (str(i + 1), hmm_name)))

    unit_start = State(None, name='suffix_start_%s' % hmm_name)
    unit_end = State(None, name='suffix_end_%s' % hmm_name)
    model.add_states(insert_states + match_states + delete_states + [unit_start, unit_end])
    last = len(delete_states) - 1

    model.add_transition(model.start, unit_start, 1)

    model.add_transition(unit_end, model.end, 1)

    insert_error = maximum_error_rate * 2 / 5
    delete_error = maximum_error_rate * 1 / 5
    model.add_transition(unit_start, delete_states[0], delete_error)
    model.add_transition(unit_start, insert_states[0], insert_error)
    for i in range(len(pattern)):
        model.add_transition(unit_start, match_states[i], (1 - insert_error - delete_error) / len(pattern))

    model.add_transition(insert_states[0], insert_states[0], insert_error)
    model.add_transition(insert_states[0], delete_states[0], delete_error)
    model.add_transition(insert_states[0], match_states[0], 1 - insert_error - delete_error)

    model.add_transition(delete_states[last], unit_end, 1 - insert_error)
    model.add_transition(delete_states[last], insert_states[last + 1], insert_error)

    model.add_transition(match_states[last], unit_end, 1 - insert_error)
    model.add_transition(match_states[last], insert_states[last + 1], insert_error)

    model.add_transition(insert_states[last + 1], insert_states[last + 1], insert_error)
    model.add_transition(insert_states[last + 1], unit_end, 1 - insert_error)

    for i in range(0, len(pattern)):
        model.add_transition(match_states[i], insert_states[i + 1], insert_error)
        model.add_transition(delete_states[i], insert_states[i + 1], insert_error)
        model.add_transition(insert_states[i + 1], insert_states[i + 1], insert_error)
        if i < len(pattern) - 1:
            model.add_transition(insert_states[i + 1], match_states[i + 1], 1 - insert_error - delete_error)
            model.add_transition(insert_states[i + 1], delete_states[i + 1], delete_error)

            model.add_transition(match_states[i], match_states[i + 1], 1 - insert_error - delete_error)
            model.add_transition(match_states[i], delete_states[i + 1], delete_error)

            model.add_transition(delete_states[i], delete_states[i + 1], delete_error)
            model.add_transition(delete_states[i], match_states[i + 1], 1 - insert_error - delete_error)

    model.bake(merge=None)

    return model


@time_usage
def get_constant_number_of_repeats_matcher_hmm(patterns, copies, vpaths, maximum_error_rate, alignment_from_reads=None):
    _validate_error_rate(maximum_error_rate)
    model = Model(name="Repeating Pattern Matcher HMM Model")

    if vpaths:
        alignment = _alignment(vpaths, alignment_from_reads)
        transitions, emissions = build_profile_hmm_pseudocounts_for_alignment(maximum_error_rate, alignment)
    else:
        transitions, emissions = build_profile_hmm_for_repeats(patterns, maximum_error_rate)
    matches = [m for m in emissions.keys() if m.startswith('M')]

    last_end = None
    for repeat in range(copies):
        insert_states = []
        match_states = []
        delete_states = []
        for i in range(len(matches) + 1):
            insert_distribution = DiscreteDistribution(emissions['I%s' % i])
            insert_states.append(State(insert_distribution, name='I%s_%s' % (i, repeat)))

        for i in range(1, len(matches) + 1):
            match_distribution = DiscreteDistribution(emissions['M%s' % i])
            match_states.append(State(match_distribution, name='M%s_%s' % (str(i), repeat)))

        for i in range(1, len(matches) + 1):
            delete_states.append(State(None, name='D%s_%s' % (str(i), repeat)))

        unit_start = State(None, name='unit_start_%s' % repeat)
        unit_end = State(None, name='unit_end_%s' % repeat)
        model.add_states(insert_states + match_states + delete_states + [unit_start, unit_end])
        n = len(delete_states) - 1

        if repeat > 0:
            model.add_transition(last_end, unit_start, 1)
        else:
            model.add_transition(model.start, unit_start, 1)

        if repeat == copies - 1:
            model.add_transition(unit_end, model.end, 1)

        model.add_transition(unit_start, match_states[0], transitions['unit_start']['M1'])
        model.add_transition(unit_start, delete_states[0], transitions['unit_start']['D1'])
        model.add_transition(unit_start, insert_states[0], transitions['unit_start']['I0'])

        model.add_transition(insert_states[0], insert_states[0], transitions['I0']['I0'])
        model.add_transition(insert_states[0], delete_states[0], transitions['I0']['D1'])
        model.add_transition(insert_states[0], match_states[0], transitions['I0']['M1'])

        model.add_transition(delete_states[n], unit_end, transitions['D%s' % (n + 1)]['unit_end'])
        model.add_transition(delete_states[n], insert_states[n + 1], transitions['D%s' % (n + 1)]['I%s' % (n + 1)])

        model.add_transition(match_states[n], unit_end, transitions['M%s' % (n + 1)]['unit_end'])
        model.add_transition(match_states[n], insert_states[n + 1], transitions['M%s' % (n + 1)]['I%s' % (n + 1)])

        model.add_transition(insert_states[n + 1], insert_states[n + 1], transitions['I%s' % (n + 1)]['I%s' % (n + 1)])
        model.add_transition(insert_states[n + 1], unit_end, transitions['I%s' % (n + 1)]['unit_end'])

        for i in range(1, len(matches) + 1):
            model.add_transition(match_states[i - 1], insert_states[i], transitions['M%s' % i]['I%s' % i])
            model.add_transition(delete_states[i - 1], insert_states[i], transitions['D%s' % i]['I%s' % i])
            model.add_transition(insert_states[i], insert_states[i], transitions['I%s' % i]['I%s' % i])
            if i < len(matches):
                model.add_transition(insert_states[i], match_states[i], transitions['I%s' % i]['M%s' % (i + 1)])
                model.add_transition(insert_states[i], delete_states[i], transitions['I%s' % i]['D%s' % (i + 1)])

                model.add_transition(match_states[i - 1], match_states[i], transitions['M%s' % i]['M%s' % (i + 1)])
                model.add_transition(match_states[i - 1], delete_states[i], transitions['M%s' % i]['D%s' % (i + 1)])

                model.add_transition(delete_states[i - 1], match_states[i], transitions['D%s' % i]['M%s' % (i + 1)])
                model.add_transition(delete_states[i - 1], delete_states[i], transitions['D%s' % i]['D%s' % (i + 1)])

        last_end = unit_end

    model.bake(merge=None)
    return model


def get_repeat_matcher_enhanced_hmm(pattern_clusters, copies, vpaths, maximum_error_rate, alignment_from_reads=None):
    _validate_error_rate(maximum_error_rate)
    model = Model(name="Repeating Pattern Matcher HMM Model")
    # Equally distributed from model start to each unique repeating unit
    repeating_unit_transition_prob = 1.0 / len(pattern_clusters)

    pattern_count = 0
    for pattern_cluster in pattern_clusters:
        pattern_count += 1
        if vpaths:
            alignment = _alignment(vpaths, alignment_from_reads)
            transitions, emissions = build_profile_hmm_pseudocounts_for_alignment(maximum_error_rate, alignment)
        else:
            transitions, emissions = build_profile_hmm_for_repeats(pattern_cluster, maximum_error_rate)
        matches = [m for m in emissions.keys() if m.startswith('M')]

        insert_states = []
        match_states = []
        delete_states = []
        for i in range(len(matches) + 1):
            insert_distribution = DiscreteDistribution(emissions['I%s' % i])
            insert_states.append(State(insert_distribution, name='I%s_%s' % (str(i), pattern_count)))

        for i in range(1, len(matches) + 1):
            match_distribution = DiscreteDistribution(emissions['M%s' % i])
            match_states.append(State(match_distribution, name='M%s_%s' % (str(i), pattern_count)))

        for i in range(1, len(matches) + 1):
            delete_states.append(State(None, name='D%s_%s' % (str(i), pattern_count)))

        unit_start = State(None, name='unit_start_%s' % str(pattern_count))
        unit_end = State(None, name='unit_end_%s' % str(pattern_count))
        model.add_states(insert_states + match_states + delete_states + [unit_start, unit_end])
        n = len(delete_states) - 1

        # From model.start to unit_starts
        model.add_transition(model.start, unit_start, repeating_unit_transition_prob)
        model.add_transition(unit_end, model.end, 1)

        # From unit start to first 3 states
        model.add_transition(unit_start, match_states[0], transitions['unit_start']['M1'])
        model.add_transition(unit_start, delete_states[0], transitions['unit_start']['D1'])
        model.add_transition(unit_start, insert_states[0], transitions['unit_start']['I0'])

        # From I0 to (I0, D1, M1)
        model.add_transition(insert_states[0], insert_states[0], transitions['I0']['I0'])
        model.add_transition(insert_states[0], delete_states[0], transitions['I0']['D1'])
        model.add_transition(insert_states[0], match_states[0], transitions['I0']['M1'])

        # From DN to (unit_end, IN)
        model.add_transition(delete_states[n], unit_end, transitions['D%s' % (n + 1)]['unit_end'])
        model.add_transition(delete_states[n], insert_states[n + 1], transitions['D%s' % (n + 1)]['I%s' % (n + 1)])

        # From MN to (unit_end, IN)
        model.add_transition(match_states[n], unit_end, transitions['M%s' % (n + 1)]['unit_end'])
        model.add_transition(match_states[n], insert_states[n + 1], transitions['M%s' % (n + 1)]['I%s' % (n + 1)])

        # From IN to (IN, unit_end)
        model.add_transition(insert_states[n + 1], insert_states[n + 1], transitions['I%s' % (n + 1)]['I%s' % (n + 1)])
        model.add_transition(insert_states[n + 1], unit_end, transitions['I%s' % (n + 1)]['unit_end'])

        for i in range(1, len(matches) + 1):
            model.add_transition(match_states[i - 1], insert_states[i], transitions['M%s' % i]['I%s' % i])
            model.add_transition(delete_states[i - 1], insert_states[i], transitions['D%s' % i]['I%s' % i])
            model.add_transition(insert_states[i], insert_states[i], transitions['I%s' % i]['I%s' % i])
            if i < len(matches):
                model.add_transition(insert_states[i], match_states[i], transitions['I%s' % i]['M%s' % (i + 1)])
                model.add_transition(insert_states[i], delete_states[i], transitions['I%s' % i]['D%s' % (i + 1)])

                model.add_transition(match_states[i - 1], match_states[i], transitions['M%s' % i]['M%s' % (i + 1)])
                model.add_transition(match_states[i - 1], delete_states[i], transitions['M%s' % i]['D%s' % (i + 1)])

                model.add_transition(delete_states[i - 1], match_states[i], transitions['D%s' % i]['M%s' % (i + 1)])
                model.add_transition(delete_states[i - 1], delete_states[i], transitions['D%s' % i]['D%s' % (i + 1)])

    # The transition probability from Model.end to Model.start (RU loop)
    repeat_prob = len(pattern_clusters) / (1.0 + len(pattern_clusters))
    model.add_transition(model.end, model.start, repeat_prob)

    model.bake(merge=None)
    return model


@time_usage
def get_read_matcher_model_enhanced(left_flanking_region, right_flanking_region, patterns, copies, vpaths,
                                    is_frameshift_mode, maximum_error_rate, alignment_from_reads=None):
    _validate_error_rate(maximum_error_rate)
    model = get_suffix_matcher_hmm(left_flanking_region, maximum_error_rate)
    if is_frameshift_mode:
        pattern_clusters = [[pattern] * patterns.count(pattern) for pattern in sorted(list(set(patterns)))]
    else:
        pattern_clusters = get_pattern_clusters(patterns)
    repeats_matcher = get_repeat_matcher_enhanced_hmm(
        pattern_clusters, copies, vpaths, maximum_error_rate, alignment_from_reads)
    right_flanking_matcher = get_prefix_matcher_hmm(right_flanking_region, maximum_error_rate)

    # Connect suffix matcher with repeat matcher
    model.concatenate(repeats_matcher, transition_probability=1.0)
    # Connect repeat matcher with prefix matcher
    model.concatenate(right_flanking_matcher, transition_probability=1.0 / (1.0 + len(pattern_clusters)))

    # 1. Setting start to matches
    repeats_matcher_model = model.subModels[1]
    repeat_match_states = []
    for state in repeats_matcher_model.states:
        if state.name[0] == 'M':
            repeat_match_states.append(state)

    suffix_matcher_model = model.subModels[0]
    suffix_start = suffix_matcher_model.states[1]

    model.set_transition(model.start, suffix_start, 0.3)  # overwriting
    for repeat_match_state in repeat_match_states:
        model.set_transition(model.start, repeat_match_state, 0.7 / len(repeat_match_states))

    # 2. Setting Match to end
    to_end = 0.7 / (len(repeat_match_states) * copies)
    total = 1 + to_end

    for match_state in repeat_match_states:
        for next_state in repeats_matcher_model.transition_map[match_state]:
            if repeats_matcher_model.transition_map[match_state][next_state] != 0:
                prob = repeats_matcher_model.transition_map[match_state][next_state]
                repeats_matcher_model.set_transition(match_state, next_state, prob / total)

        repeats_matcher_model.set_transition(match_state, right_flanking_matcher.end, to_end / total)

    read_length_used_to_build_model = len(left_flanking_region)

    # Get the lower bound of log probability score
    if is_frameshift_mode:
        dp_score_threshold = 0
        dp_score_threshold += log(0.3)  # model-start to match < to suffix-start
        dp_score_threshold += (log(0.97) + log(0.99)) * read_length_used_to_build_model  # sequence length * match
        dp_score_threshold += (log(0.001) + log(0.01)) * (len(patterns[0])/2)  # allow indels of 50% of pattern length

        # Max end to start loop
        repeat_loop_prob = len(pattern_clusters) / (1.0 + len(pattern_clusters))  # end to start
        dp_score_threshold += log(repeat_loop_prob) * (read_length_used_to_build_model / len(patterns[0]))
        # Max start to unit_n loop
        repeating_unit_transition_prob = 1.0 / len(pattern_clusters)  # ru-start to unit_n
        dp_score_threshold += log(repeating_unit_transition_prob) * (read_length_used_to_build_model/len(patterns[0]))

        dp_score_threshold += log(1.0 / len(set(patterns)))  # transition from pattern to prefix-start
        dp_score_threshold += log(to_end / total)  # match to end

        model.bake(merge=None, read_length=read_length_used_to_build_model, dp_score_threshold=dp_score_threshold)
    else:
        model.bake(merge=None, read_length=read_length_used_to_build_model)

    return model
