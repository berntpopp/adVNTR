import numpy as np

import logging
from pomegranate import DiscreteDistribution, State
from pomegranate import HiddenMarkovModel as Model
from advntr.profile_hmm import build_profile_hmm_for_repeats, build_profile_hmm_pseudocounts_for_alignment
from advntr.profiler import time_usage
from advntr import settings


def path_to_alignment(x, y, path):
    for i, (index, state) in enumerate(path[1:-1]):
        name = state.name

        if name.startswith('D'):
            y = y[:i] + '-' + y[i:]
        elif name.startswith('I'):
            x = x[:i] + '-' + x[i:]

    return x, y


def get_multiple_alignment_of_viterbi_paths(repeats_sequences, repeats_visited_states):
    alignment_states = {}
    multiple_alignment_length = 0
    for repeat_visited_states in repeats_visited_states:
        state_map = {}
        for visited_state in repeat_visited_states:
            visited_state = visited_state.split('_')[0]
            if visited_state not in state_map.keys():
                state_map[visited_state] = 0
            state_map[visited_state] += 1
        for key, value in state_map.items():
            index = int(key.split('_')[0][1:])
            multiple_alignment_length = max(multiple_alignment_length, index)
            if key not in alignment_states.keys():
                alignment_states[key] = value
            alignment_states[key] = max(alignment_states[key], value)

    alignment_visited_states = []
    for i in range(multiple_alignment_length+1):
        key = 'M%s' % i
        if key in alignment_states.keys():
            for j in range(alignment_states[key]):
                alignment_visited_states.append(key)
        key = 'I%s' % i
        if key in alignment_states.keys():
            for j in range(alignment_states[key]):
                alignment_visited_states.append(key)

    alignment = ['' for _ in range(len(repeats_sequences))]
    for i, repeat_sequence in enumerate(repeats_sequences):
        sequence_index = 0
        individual_alignment = [state.split('_')[0] for state in repeats_visited_states[i]]
        for state in alignment_visited_states:
            found = False
            for k, ind_state in enumerate(individual_alignment):
                if state == ind_state:
                    individual_alignment[k] = 'DELETED'
                    found = True
            if found:
                alignment[i] += repeat_sequence[sequence_index]
                sequence_index += 1
            else:
                alignment[i] += '-'

    return alignment


def extract_repeating_segments_from_read(sequence, visited_states):
    repeats = []
    vpaths = []
    prev_start = None
    prev_start_state = None
    sequence_index = 0
    for i in range(len(visited_states)):
        if visited_states[i].startswith('unit_end') and prev_start is not None:
            repeat = ''
            vpath = []
            for j in range(prev_start, sequence_index):
                repeat += sequence[j]
            for j in range(prev_start_state+1, i):
                vpath.append(visited_states[j])
            repeats.append(repeat)
            vpaths.append(vpath)
        if visited_states[i].startswith('unit_start'):
            prev_start = sequence_index
            prev_start_state = i
        if is_emitting_state(visited_states[i]):
            sequence_index += 1
    return repeats, vpaths


def get_multiple_alignment_of_repeats_from_reads(sequence_vpath_list):
    repeats_sequences = []
    repeats_visited_states = []
    for sequence, vpath in sequence_vpath_list:
        visited_states = [state.name for idx, state in vpath[1:-1]]
        repeats, repeats_vstates = extract_repeating_segments_from_read(sequence, visited_states)
        repeats_sequences += repeats
        repeats_visited_states += repeats_vstates

    return get_multiple_alignment_of_viterbi_paths(repeats_sequences, repeats_visited_states)


def get_emitted_basepair_from_visited_states(state, visited_states, sequence):
    base_pair_idx = 0
    for visited_state in visited_states:
        if visited_state == state:
            return sequence[base_pair_idx]
        if is_emitting_state(visited_state):
            base_pair_idx += 1
    return None


def is_match_state(state_name):
    if state_name.startswith('M'):
        return True
    return False


def is_emitting_state(state_name):
    if state_name.startswith('M') or state_name.startswith('I') or state_name.startswith('start_random_matches') \
            or state_name.startswith('end_random_matches'):
        return True
    return False


def get_repeating_pattern_lengths(visited_states):
    lengths = []
    prev_start = None
    for i in range(len(visited_states)):
        if visited_states[i].startswith('unit_end') and prev_start is not None:
            current_len = 0
            for j in range(prev_start, i):
                if is_emitting_state(visited_states[j]):
                    current_len += 1
            lengths.append(current_len)
        if visited_states[i].startswith('unit_start'):
            prev_start = i
    return lengths


def get_repeat_segments_from_visited_states_and_region(visited_states, region):
    lengths = get_repeating_pattern_lengths(visited_states)

    repeat_segments = []
    added = 0
    for l in lengths:
        repeat_segments.append(region[added:added + l])
        added += l
    return repeat_segments


def get_number_of_repeats_in_vpath(vpath):
    starts = 0
    ends = 0
    visited_states = [state.name for idx, state in vpath[1:-1]]

    read_length = 0
    for vs in visited_states:
        if is_emitting_state(vs):
            read_length += 1

    minimum_required_bp_in_repeat = 3
    current_bp = 0
    first_end = None
    last_end = None
    first_start = None
    last_start = None
    for i in range(len(visited_states)):
        if is_emitting_state(visited_states[i]):
            current_bp += 1
        if visited_states[i].startswith('unit_start') and read_length - current_bp >= minimum_required_bp_in_repeat:
            if first_start is None:
                first_start = current_bp
            last_start = current_bp
            starts += 1
        if visited_states[i].startswith('unit_end') and current_bp >= minimum_required_bp_in_repeat:
            if first_end is None:
                first_end = current_bp
            last_end = current_bp
            ends += 1
    delta = 0
    if last_start is not None and first_start is not None and last_end is not None and first_end is not None:
        if first_end < first_start and last_start > last_end:
            delta = 1
    return max(starts, ends) + delta


def get_number_of_matches_in_vpath(vpath):
    visited_states = [state.name for idx, state in vpath[1:-1]]
    result = 0
    for i in range(len(visited_states)):
        if is_match_state(visited_states[i]):
            result += 1
    return result


def get_number_of_repeat_bp_matches_in_vpath(vpath):
    visited_states = [state.name for idx, state in vpath[1:-1]]
    result = 0
    for i in range(len(visited_states)):
        if is_emitting_state(visited_states[i]) and not visited_states[i].endswith('fix'):
            result += 1
    return result


def get_flanking_regions_matching_rate(vpath, sequence, left_flank, right_flank, accuracy_filter=False, verbose=False):
    visited_states = [state.name for idx, state in vpath[1:-1]]
    right_flanking_matches = 0
    right_flanking_basepairs = 0
    left_flanking_matches = 0
    left_flanking_basepairs = 0
    seq_index = 0
    max_hmm_index = -1
    prev_state = visited_states[0]
    for state in visited_states:
        if 'suffix_end_suffix' in state:
            max_hmm_index = int(prev_state.split("_")[0][1:])
            break
        prev_state = state
    if verbose:
        logging.debug("len visited_states {}".format(len(visited_states)))
    for i in range(len(visited_states)):
        if 'start' in visited_states[i] or 'end' in visited_states[i]:
            continue
        hmm_state = int(visited_states[i].split("_")[0][1:])
        if visited_states[i].endswith('prefix'):
            if verbose:
                logging.debug("state {} is matching {} seq_index {} sequence[seq_index] {} right_flank[hmm_state - 1] {} is emitting {}".format(
                          visited_states[i],
                          is_match_state(visited_states[i]),
                          seq_index,
                          sequence[seq_index],
                          right_flank[hmm_state - 1],
                          is_emitting_state(visited_states[i])))
            if is_match_state(visited_states[i]) and sequence[seq_index] == right_flank[hmm_state - 1]:
                right_flanking_matches += 1
            if is_emitting_state(visited_states[i]):
                right_flanking_basepairs += 1
        if visited_states[i].endswith('suffix'):
            if verbose:
                logging.debug("state {} is matching {} seq_index {} sequence[seq_index] {} left_flank[-(max_hmm_index - hmm_state + 1)] {} is emitting {}".format(
                          visited_states[i],
                          is_match_state(visited_states[i]),
                          seq_index,
                          sequence[seq_index],
                          left_flank[-(max_hmm_index - hmm_state + 1)],
                          is_emitting_state(visited_states[i])))
            if is_match_state(visited_states[i]) and sequence[seq_index] == left_flank[-(max_hmm_index - hmm_state + 1)]:
                left_flanking_matches += 1
            if is_emitting_state(visited_states[i]):
                left_flanking_basepairs += 1
        if is_emitting_state(visited_states[i]):
            seq_index += 1
    if accuracy_filter:
        # If accuracy filter is set, we want to be conservative in read recruiting.
        # Therefore, in case of zero bp right (or left) flanking match, read is not confidently spanning the VNTR, so return 0.
        epsilon = 0.00001
        right_rate = float(right_flanking_matches) / right_flanking_basepairs if right_flanking_basepairs != 0 else epsilon
        left_rate = float(left_flanking_matches) / left_flanking_basepairs if left_flanking_basepairs != 0 else epsilon
    else:
        right_rate = float(right_flanking_matches) / right_flanking_basepairs if right_flanking_basepairs != 0 else 1
        left_rate = float(left_flanking_matches) / left_flanking_basepairs if left_flanking_basepairs != 0 else 1

    result = min(right_rate, left_rate)
    return result


def get_left_flanking_region_size_in_vpath(vpath):
    visited_states = [state.name for idx, state in vpath[1:-1]]
    result = 0
    for i in range(len(visited_states)):
        if is_emitting_state(visited_states[i]) and visited_states[i].endswith('suffix'):
            result += 1
    return result


def get_right_flanking_region_size_in_vpath(vpath):
    visited_states = [state.name for idx, state in vpath[1:-1]]
    result = 0
    for i in range(len(visited_states)):
        if is_emitting_state(visited_states[i]) and visited_states[i].endswith('prefix'):
            result += 1
    return result


@time_usage
def get_prefix_matcher_hmm(pattern):
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
    last = len(delete_states)-1

    model.add_transition(model.start, unit_start, 1)

    model.add_transition(unit_end, model.end, 1)

    insert_error = settings.MAX_ERROR_RATE * 2 / 5
    delete_error = settings.MAX_ERROR_RATE * 1 / 5
    model.add_transition(unit_start, match_states[0], 1 - insert_error - delete_error)
    model.add_transition(unit_start, delete_states[0], delete_error)
    model.add_transition(unit_start, insert_states[0], insert_error)

    model.add_transition(insert_states[0], insert_states[0], insert_error)
    model.add_transition(insert_states[0], delete_states[0], delete_error)
    model.add_transition(insert_states[0], match_states[0], 1 - insert_error - delete_error)

    model.add_transition(delete_states[last], unit_end, 1 - insert_error)
    model.add_transition(delete_states[last], insert_states[last+1], insert_error)

    model.add_transition(match_states[last], unit_end, 1 - insert_error)
    model.add_transition(match_states[last], insert_states[last+1], insert_error)

    model.add_transition(insert_states[last+1], insert_states[last+1], insert_error)
    model.add_transition(insert_states[last+1], unit_end, 1 - insert_error)

    for i in range(0, len(pattern)):
        model.add_transition(match_states[i], insert_states[i+1], insert_error)
        model.add_transition(delete_states[i], insert_states[i+1], insert_error)
        model.add_transition(insert_states[i+1], insert_states[i+1], insert_error)
        if i < len(pattern) - 1:
            model.add_transition(insert_states[i+1], match_states[i+1], 1 - insert_error - delete_error)
            model.add_transition(insert_states[i+1], delete_states[i+1], delete_error)

            model.add_transition(match_states[i], match_states[i+1], 1 - insert_error - delete_error - 0.01)
            model.add_transition(match_states[i], delete_states[i+1], delete_error)
            model.add_transition(match_states[i], unit_end, 0.01)

            model.add_transition(delete_states[i], delete_states[i+1], delete_error)
            model.add_transition(delete_states[i], match_states[i+1], 1 - insert_error - delete_error)

    model.bake(merge=None)

    return model


@time_usage
def get_suffix_matcher_hmm(pattern):
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
    last = len(delete_states)-1

    model.add_transition(model.start, unit_start, 1)

    model.add_transition(unit_end, model.end, 1)

    insert_error = settings.MAX_ERROR_RATE * 2 / 5
    delete_error = settings.MAX_ERROR_RATE * 1 / 5
    model.add_transition(unit_start, delete_states[0], delete_error)
    model.add_transition(unit_start, insert_states[0], insert_error)
    for i in range(len(pattern)):
        model.add_transition(unit_start, match_states[i], (1 - insert_error - delete_error) / len(pattern))

    model.add_transition(insert_states[0], insert_states[0], insert_error)
    model.add_transition(insert_states[0], delete_states[0], delete_error)
    model.add_transition(insert_states[0], match_states[0], 1 - insert_error - delete_error)

    model.add_transition(delete_states[last], unit_end, 1 - insert_error)
    model.add_transition(delete_states[last], insert_states[last+1], insert_error)

    model.add_transition(match_states[last], unit_end, 1 - insert_error)
    model.add_transition(match_states[last], insert_states[last+1], insert_error)

    model.add_transition(insert_states[last+1], insert_states[last+1], insert_error)
    model.add_transition(insert_states[last+1], unit_end, 1 - insert_error)

    for i in range(0, len(pattern)):
        model.add_transition(match_states[i], insert_states[i+1], insert_error)
        model.add_transition(delete_states[i], insert_states[i+1], insert_error)
        model.add_transition(insert_states[i+1], insert_states[i+1], insert_error)
        if i < len(pattern) - 1:
            model.add_transition(insert_states[i+1], match_states[i+1], 1 - insert_error - delete_error)
            model.add_transition(insert_states[i+1], delete_states[i+1], delete_error)

            model.add_transition(match_states[i], match_states[i+1], 1 - insert_error - delete_error)
            model.add_transition(match_states[i], delete_states[i+1], delete_error)

            model.add_transition(delete_states[i], delete_states[i+1], delete_error)
            model.add_transition(delete_states[i], match_states[i+1], 1 - insert_error - delete_error)

    model.bake(merge=None)

    return model


@time_usage
def get_constant_number_of_repeats_matcher_hmm(patterns, copies, vpaths):
    model = Model(name="Repeating Pattern Matcher HMM Model")

    if vpaths:
        alignment = get_multiple_alignment_of_repeats_from_reads(vpaths)
        transitions, emissions = build_profile_hmm_pseudocounts_for_alignment(settings.MAX_ERROR_RATE, alignment)
    else:
        transitions, emissions = build_profile_hmm_for_repeats(patterns, settings.MAX_ERROR_RATE)
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
        n = len(delete_states)-1

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

        model.add_transition(delete_states[n], unit_end, transitions['D%s' % (n+1)]['unit_end'])
        model.add_transition(delete_states[n], insert_states[n+1], transitions['D%s' % (n+1)]['I%s' % (n+1)])

        model.add_transition(match_states[n], unit_end, transitions['M%s' % (n+1)]['unit_end'])
        model.add_transition(match_states[n], insert_states[n+1], transitions['M%s' % (n+1)]['I%s' % (n+1)])

        model.add_transition(insert_states[n+1], insert_states[n+1], transitions['I%s' % (n+1)]['I%s' % (n+1)])
        model.add_transition(insert_states[n+1], unit_end, transitions['I%s' % (n+1)]['unit_end'])

        for i in range(1, len(matches)+1):
            model.add_transition(match_states[i-1], insert_states[i], transitions['M%s' % i]['I%s' % i])
            model.add_transition(delete_states[i-1], insert_states[i], transitions['D%s' % i]['I%s' % i])
            model.add_transition(insert_states[i], insert_states[i], transitions['I%s' % i]['I%s' % i])
            if i < len(matches):
                model.add_transition(insert_states[i], match_states[i], transitions['I%s' % i]['M%s' % (i+1)])
                model.add_transition(insert_states[i], delete_states[i], transitions['I%s' % i]['D%s' % (i+1)])

                model.add_transition(match_states[i-1], match_states[i], transitions['M%s' % i]['M%s' % (i+1)])
                model.add_transition(match_states[i-1], delete_states[i], transitions['M%s' % i]['D%s' % (i+1)])

                model.add_transition(delete_states[i-1], match_states[i], transitions['D%s' % i]['M%s' % (i+1)])
                model.add_transition(delete_states[i-1], delete_states[i], transitions['D%s' % i]['D%s' % (i+1)])

        last_end = unit_end

    model.bake(merge=None)
    return model


@time_usage
def get_variable_number_of_repeats_matcher_hmm(patterns, copies=1, vpaths=None):
    model = get_constant_number_of_repeats_matcher_hmm(patterns, copies, vpaths)

    start_repeats_matches = State(None, name='start_repeating_pattern_match')
    end_repeats_matches = State(None, name='end_repeating_pattern_match')
    mat = model.dense_transition_matrix()
    states = model.states
    states.append(start_repeats_matches)
    states.append(end_repeats_matches)
    states_count = len(mat)
    start_repeats_ind = states_count
    end_repeats_ind = states_count + 1
    mat = np.c_[mat, np.zeros(states_count), np.zeros(states_count)]
    mat = np.r_[mat, [np.zeros(states_count + 2)]]
    mat = np.r_[mat, [np.zeros(states_count + 2)]]

    unit_ends = []
    for i, state in enumerate(model.states):
        if state.name.startswith('unit_end'):
            unit_ends.append(i)

    first_unit_start = None
    for i in range(len(mat[model.start_index])):
        if mat[model.start_index][i] != 0:
            first_unit_start = i
    mat[model.start_index][first_unit_start] = 0.0
    mat[model.start_index][start_repeats_ind] = 1
    mat[start_repeats_ind][first_unit_start] = 1

    for unit_end in unit_ends:
        next_state = None
        for j in range(len(mat[unit_end])):
            if mat[unit_end][j] != 0:
                next_state = j
        mat[unit_end][next_state] = 0.5
        mat[unit_end][end_repeats_ind] = 0.5

    mat[end_repeats_ind][model.end_index] = 1

    starts = np.zeros(states_count + 2)
    starts[model.start_index] = 1.0
    ends = np.zeros(states_count + 2)
    ends[model.end_index] = 1.0
    state_names = [state.name for state in states]
    distributions = [state.distribution for state in states]
    name = 'Repeat Matcher HMM Model'
    new_model = Model.from_matrix(mat, distributions, starts, ends, name=name, state_names=state_names, merge=None)
    new_model.bake(merge=None)
    return new_model


@time_usage
def get_read_matcher_model(left_flanking_region, right_flanking_region, patterns, copies=1, vpaths=None):
    model = get_suffix_matcher_hmm(left_flanking_region)
    repeats_matcher = get_variable_number_of_repeats_matcher_hmm(patterns, copies, vpaths)
    right_flanking_matcher = get_prefix_matcher_hmm(right_flanking_region)
    model.concatenate(repeats_matcher)
    model.concatenate(right_flanking_matcher)
    model.bake(merge=None)

    mat = model.dense_transition_matrix()

    first_repeat_matches = []
    repeat_match_states = []
    suffix_start = None
    for i, state in enumerate(model.states):
        if state.name[0] == 'M' and state.name.split('_')[-1] == '0':
            first_repeat_matches.append(i)
        if state.name[0] == 'M' and state.name.split('_')[-1] not in ['prefix', 'suffix']:
            repeat_match_states.append(i)
        if state.name == 'suffix_start_suffix':
            suffix_start = i

    mat[model.start_index][suffix_start] = 0.3
    for first_repeat_match in first_repeat_matches:
        mat[model.start_index][first_repeat_match] = 0.7 / len(first_repeat_matches)

    for match_state in repeat_match_states:
        to_end = 0.7 / len(repeat_match_states)
        total = 1 + to_end
        for next_state in range(len(mat[match_state])):
            if mat[match_state][next_state] != 0:
                mat[match_state][next_state] /= total
        mat[match_state][model.end_index] = to_end / total

    starts = np.zeros(len(model.states))
    starts[model.start_index] = 1.0
    ends = np.zeros(len(model.states))
    ends[model.end_index] = 1.0
    state_names = [state.name for state in model.states]
    distributions = [state.distribution for state in model.states]
    name = 'Read Matcher'
    new_model = Model.from_matrix(mat, distributions, starts, ends, name=name, state_names=state_names, merge=None)
    new_model.bake(merge=None)
    return new_model


def build_reference_repeat_finder_hmm(patterns, copies=1):
    pattern = patterns[0]
    model = Model(name="HMM Model")
    insert_distribution = DiscreteDistribution({'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25})

    last_end = None
    start_random_matches = State(insert_distribution, name='start_random_matches')
    end_random_matches = State(insert_distribution, name='end_random_matches')
    model.add_states([start_random_matches, end_random_matches])
    for repeat in range(copies):
        insert_states = []
        match_states = []
        delete_states = []
        for i in range(len(pattern) + 1):
            insert_states.append(State(insert_distribution, name='I%s_%s' % (i, repeat)))

        for i in range(len(pattern)):
            distribution_map = dict({'A': 0.01, 'C': 0.01, 'G': 0.01, 'T': 0.01})
            distribution_map[pattern[i]] = 0.97
            match_states.append(State(DiscreteDistribution(distribution_map), name='M%s_%s' % (str(i + 1), repeat)))

        for i in range(len(pattern)):
            delete_states.append(State(None, name='D%s_%s' % (str(i + 1), repeat)))

        unit_start = State(None, name='unit_start_%s' % repeat)
        unit_end = State(None, name='unit_end_%s' % repeat)
        model.add_states(insert_states + match_states + delete_states + [unit_start, unit_end])
        last = len(delete_states)-1

        if repeat > 0:
            model.add_transition(last_end, unit_start, 0.5)
        else:
            model.add_transition(model.start, unit_start, 0.5)
            model.add_transition(model.start, start_random_matches, 0.5)
            model.add_transition(start_random_matches, unit_start, 0.5)
            model.add_transition(start_random_matches, start_random_matches, 0.5)

        model.add_transition(unit_end, end_random_matches, 0.5)
        if repeat == copies - 1:
            model.add_transition(unit_end, model.end, 0.5)
            model.add_transition(end_random_matches, end_random_matches, 0.5)
            model.add_transition(end_random_matches, model.end, 0.5)

        model.add_transition(unit_start, match_states[0], 0.98)
        model.add_transition(unit_start, delete_states[0], 0.01)
        model.add_transition(unit_start, insert_states[0], 0.01)

        model.add_transition(insert_states[0], insert_states[0], 0.01)
        model.add_transition(insert_states[0], delete_states[0], 0.01)
        model.add_transition(insert_states[0], match_states[0], 0.98)

        model.add_transition(delete_states[last], unit_end, 0.99)
        model.add_transition(delete_states[last], insert_states[last+1], 0.01)

        model.add_transition(match_states[last], unit_end, 0.99)
        model.add_transition(match_states[last], insert_states[last+1], 0.01)

        model.add_transition(insert_states[last+1], insert_states[last+1], 0.01)
        model.add_transition(insert_states[last+1], unit_end, 0.99)

        for i in range(0, len(pattern)):
            model.add_transition(match_states[i], insert_states[i+1], 0.01)
            model.add_transition(delete_states[i], insert_states[i+1], 0.01)
            model.add_transition(insert_states[i+1], insert_states[i+1], 0.01)
            if i < len(pattern) - 1:
                model.add_transition(insert_states[i+1], match_states[i+1], 0.98)
                model.add_transition(insert_states[i+1], delete_states[i+1], 0.01)

                model.add_transition(match_states[i], match_states[i+1], 0.98)
                model.add_transition(match_states[i], delete_states[i+1], 0.01)

                model.add_transition(delete_states[i], delete_states[i+1], 0.01)
                model.add_transition(delete_states[i], match_states[i+1], 0.98)

        last_end = unit_end

    model.bake()
    if len(patterns) > 1:
        # model.fit(patterns, algorithm='baum-welch', transition_pseudocount=1, use_pseudocount=True)
        fit_patterns = [pattern * copies for pattern in patterns]
        #model.fit(fit_patterns, algorithm='viterbi', transition_pseudocount=1, use_pseudocount=True)

    return model
