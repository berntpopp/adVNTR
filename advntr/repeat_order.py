"""Repeat-order / realignment helpers for frameshift calling.

These four functions are the repeat-order / realignment concern behind issue #5: the
path Task 3's INFO log (`advntr/vntr_finder.py`) reports as inactive whenever the read
length covers fewer than three repeat units. Keeping them as one unit keeps that
disabled path visible as a single surface before #5 changes it, rather than scattered
across `vntr_finder.py` where a future edit could touch one piece without noticing the
others.
"""
from collections import defaultdict


def get_reference_repeat_order(patterns, unique_repeat_units):
    reference_repeat_order = ['L']
    for repeat_unit in patterns:
        for i, unique_repeat_unit in enumerate(unique_repeat_units):
            if repeat_unit == unique_repeat_unit:
                reference_repeat_order.append(str(i+1))
    reference_repeat_order.append('R')

    return reference_repeat_order


def get_repeat_unit_number(read):
    sequence = read.sequence
    visited_states = [state.name for idx, state in read.vpath[1:-1]]

    read_as_repeat_unit_number = []
    annotated_read = defaultdict(str)
    unit_start_points = []

    current_state = None
    visited_repeat_index = -1
    sequence_index = 0
    for si, state in enumerate(visited_states):
        if 'suffix' in state:
            if current_state != 'L':
                read_as_repeat_unit_number.append('L')
                unit_start_points.append(si)
                visited_repeat_index += 1
                current_state = 'L'
        elif 'unit_end' in state:
            if current_state is None:
                repeat_unit_number = state.split("_")[-1]
                read_as_repeat_unit_number.append(repeat_unit_number)
                unit_start_points.append(si)
                visited_repeat_index += 1
                current_state = repeat_unit_number
        elif 'unit_start' in state:
            repeat_unit_number = state.split("_")[-1]
            read_as_repeat_unit_number.append(repeat_unit_number)
            unit_start_points.append(si)
            visited_repeat_index += 1
            current_state = repeat_unit_number
        elif 'prefix' in state:
            if current_state != 'R':
                read_as_repeat_unit_number.append('R')
                unit_start_points.append(si)
                visited_repeat_index += 1
                current_state = 'R'
        else:
            # Starting with match states (e.g. M2_1, M3_1)
            # Same as unit_end
            if current_state is None:
                repeat_unit_number = state.split("_")[-1]
                read_as_repeat_unit_number.append(repeat_unit_number)
                unit_start_points.append(si)
                visited_repeat_index += 1
                current_state = repeat_unit_number

        if state.startswith('M') or state.startswith('I'):
            annotated_read[visited_repeat_index] += sequence[sequence_index]
            sequence_index += 1

    assert len(sequence) == sequence_index
    return read_as_repeat_unit_number, annotated_read, unit_start_points


def find_mutated_repeat_unit(read, reference):
    """
    read, reference
    :param read_as_repeat_unit_number: string1
    :param reference_repeat_order: string2
    :return: the mutated repeat unit number based on local alignment
    """
    n = len(read)
    m = len(reference)
    dynamic_table = [[0] * (m + 1) for _ in range(n + 1)]
    backtrack = [[0] * (m + 1) for _ in range(n + 1)]

    # Initialize the first row and column
    for i in range(1, n + 1):
        dynamic_table[i][0] = 0

    for j in range(1, m + 1):
        dynamic_table[0][j] = 0

    max_value = 0
    max_cell = [0, 0]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match_score = 1 if read[i-1] == reference[j-1] else 0
            dynamic_table[i][j] = dynamic_table[i - 1][j - 1] + match_score

            if dynamic_table[i][j] >= max_value:
                max_value = dynamic_table[i][j]
                max_cell[0] = i
                max_cell[1] = j

            if dynamic_table[i][j] == 0:
                backtrack[i][j] = "source"
            else:
                backtrack[i][j] = "diagonal"

    alignment = [[], []]
    x = max_cell[0]
    y = max_cell[1]

    mutated_repeat_indices = []
    mutated_repeats = []
    correct_repeats = []

    prev_score = len(read) + 1  # max + 1 (impossible to achieve this value)
    while x != 0 and y != 0:
        current_score = dynamic_table[x][y]
        if prev_score == current_score:  # meaning the previous match was wrong
            mutated_repeat_indices.append(x)
            mutated_repeats.append(read[x])
            correct_repeats.append(reference[y])
        prev_score = current_score

        if backtrack[x][y] == "diagonal":
            alignment[0].insert(0, read[x - 1])
            alignment[1].insert(0, reference[y - 1])
            x = x - 1
            y = y - 1
        else:
            x = 0
            y = 0

    match_count = dynamic_table[max_cell[0]][max_cell[1]]

    if match_count == len(reference):
        return [], [], []
    else:
        return mutated_repeat_indices, mutated_repeats, correct_repeats


def get_valid_repeat_orders(repeat_orders):
    min_observed_repeat = 2
    valid_repeat_orders = set()
    for size in range(min_observed_repeat, len(repeat_orders)):
        for i in range(len(repeat_orders) - size + 1):
            valid_repeat_orders.add(''.join(repeat_orders[i:i+size]))

    return valid_repeat_orders
