from collections import OrderedDict, defaultdict, namedtuple
import json


MutationEvent = namedtuple(
    'MutationEvent',
    'type raw_offset inserted_sequence normalized_offset normalized_sequence'
)
RawMutation = namedtuple(
    'RawMutation',
    'visited_index legacy_key repeat_occurrence observed_unit event'
)
FrameshiftEvidence = namedtuple(
    'FrameshiftEvidence',
    'selected_read_index query_name repeat_occurrence observed_unit events'
)


def left_normalise_insertion(ref_unit, offset, inserted_sequence):
    """Canonicalise an insertion without changing its implied edited repeat unit."""
    if ref_unit is None or len(ref_unit) == 0:
        raise ValueError('ref_unit must be non-empty')
    if inserted_sequence is None or len(inserted_sequence) == 0:
        raise ValueError('inserted_sequence must be non-empty')
    if offset < 0 or offset > len(ref_unit):
        raise ValueError('offset must be between 0 and len(ref_unit)')

    normalised_offset = offset
    rotated_inserted_sequence = inserted_sequence
    while normalised_offset > 0:
        previous_base = ref_unit[normalised_offset - 1]
        if rotated_inserted_sequence[-1] != previous_base:
            break
        rotated_inserted_sequence = previous_base + rotated_inserted_sequence[:-1]
        normalised_offset -= 1
    return normalised_offset, rotated_inserted_sequence


def _consumes_read_base(state):
    return (state.startswith('M') or state.startswith('I') or
            state.startswith('start_random_matches') or
            state.startswith('end_random_matches'))


def _partial_end_starts(visited_states):
    open_start = None
    partial_ends = set()
    for index, state in enumerate(visited_states):
        if state.startswith('unit_start'):
            if open_start is not None:
                partial_ends.add(open_start)
            open_start = index
        elif state.startswith('unit_end') and open_start is not None:
            open_start = None
    if open_start is not None:
        partial_ends.add(open_start)
    return partial_ends


def _occurrences_and_bases(visited_states, sequence):
    partial_ends = _partial_end_starts(visited_states)
    occurrences = []
    emitted_bases = []
    bases_by_occurrence = defaultdict(list)
    current_occurrence = 'partial_start'
    complete_repeat_index = -1
    read_index = 0

    for index, state in enumerate(visited_states):
        if state.startswith('unit_start'):
            complete_repeat_index += 1
            current_occurrence = ('partial_end' if index in partial_ends
                                  else complete_repeat_index)
        occurrence = current_occurrence
        if state.endswith('suffix'):
            occurrence = 'suffix_flank'
        elif state.endswith('prefix'):
            occurrence = 'prefix_flank'
        occurrences.append(occurrence)

        emitted_base = None
        if _consumes_read_base(state):
            emitted_base = sequence[read_index]
            read_index += 1
        emitted_bases.append(emitted_base)
        if emitted_base is not None and (state.startswith('M') or state.startswith('I')):
            bases_by_occurrence[occurrence].append(emitted_base)

    observed_units = dict((occurrence, ''.join(bases))
                          for occurrence, bases in bases_by_occurrence.items())
    return occurrences, emitted_bases, observed_units


def _reference_unit_for_state(state, reference_units):
    pattern_index = state.split('_')[-1]
    if not pattern_index.isdigit():
        return None
    index = int(pattern_index) - 1
    if index < 0 or index >= len(reference_units):
        return None
    return reference_units[index]


def extract_raw_mutations(visited_states, sequence, reference_units):
    """Retain vpath evidence before legacy State collapses insertion runs."""
    occurrences, emitted_bases, observed_units = _occurrences_and_bases(visited_states, sequence)
    first_insertion_base = {}
    for index, state in enumerate(visited_states):
        if state.startswith('I') and state not in first_insertion_base:
            first_insertion_base[state] = emitted_bases[index]

    by_index = {}
    index = 0
    while index < len(visited_states):
        state = visited_states[index]
        occurrence = occurrences[index]
        observed_unit = observed_units.get(occurrence, '')
        if state.startswith('D'):
            offset = int(state.split('_')[0][1:])
            event = MutationEvent('D', offset, '', None, None)
            by_index[index] = RawMutation(index, state, occurrence, observed_unit, event)
        elif state.startswith('I'):
            run_end = index + 1
            while (run_end < len(visited_states) and visited_states[run_end] == state and
                   occurrences[run_end] == occurrence):
                run_end += 1
            inserted_sequence = ''.join(emitted_bases[index:run_end])
            offset = int(state.split('_')[0][1:])
            reference_unit = _reference_unit_for_state(state, reference_units)
            if reference_unit is None:
                normalised_offset, normalised_sequence = offset, inserted_sequence
            else:
                normalised_offset, normalised_sequence = left_normalise_insertion(
                    reference_unit, offset, inserted_sequence
                )
            event = MutationEvent('I', offset, inserted_sequence,
                                  normalised_offset, normalised_sequence)
            if state.endswith('suffix') or state.endswith('prefix'):
                legacy_key = state
            else:
                legacy_key = state + '_' + first_insertion_base[state]
            by_index[index] = RawMutation(index, legacy_key, occurrence, observed_unit, event)
            index = run_end - 1
        index += 1
    return by_index


def legacy_mutation_candidates(mutation_counts):
    """Mirror legacy State grouping while exposing each candidate's source keys."""
    items = list(mutation_counts.items())
    if len(items) == 1:
        mutation, count = items[0]
        state = mutation + ('_LEN%d' % count if mutation.startswith('I') else '')
        return [(state, (mutation,))]

    candidates = []
    previous = items[0][0]
    sequence = previous
    sequence_keys = [previous]
    if previous.startswith('I'):
        sequence += '_LEN%d' % items[0][1]

    def flush():
        if sequence is not None:
            candidates.append((sequence, tuple(sequence_keys)))

    for mutation, count in items[1:]:
        current_index = int(mutation.split('_')[0][1:])
        current_hmm = int(mutation.split('_')[1])
        previous_index = int(previous.split('_')[0][1:])
        previous_hmm = int(previous.split('_')[1])

        if mutation.startswith('D'):
            if previous_index + 1 == current_index and previous_hmm == current_hmm:
                if sequence is None:
                    sequence, sequence_keys = mutation, [mutation]
                else:
                    sequence += '&' + mutation
                    sequence_keys.append(mutation)
            else:
                flush()
                sequence, sequence_keys = mutation, [mutation]

        if mutation.startswith('I'):
            if previous_index == current_index and previous_hmm == current_hmm:
                sequence += '&%s_LEN%d' % (mutation, count)
                sequence_keys.append(mutation)
                flush()
                sequence, sequence_keys = None, []
            else:
                flush()
                candidates.append(('%s_LEN%d' % (mutation, count), (mutation,)))
                sequence, sequence_keys = None, []
        previous = mutation

    flush()
    return candidates


def evidence_for_candidate(selected_read_index, query_name, legacy_keys, raw_mutations):
    relevant = [raw for raw in raw_mutations if raw.legacy_key in legacy_keys]
    relevant.sort(key=lambda raw: raw.visited_index)
    grouped = OrderedDict()
    for raw in relevant:
        if raw.repeat_occurrence not in grouped:
            grouped[raw.repeat_occurrence] = [raw.observed_unit, []]
        grouped[raw.repeat_occurrence][1].append(raw.event)
    return [FrameshiftEvidence(selected_read_index, query_name, occurrence,
                               observed_unit, tuple(events))
            for occurrence, (observed_unit, events) in grouped.items()]


def _event_dict(event):
    value = {'type': event.type, 'raw_offset': event.raw_offset,
             'inserted_sequence': event.inserted_sequence}
    if event.type == 'I':
        value['normalized_offset'] = event.normalized_offset
        value['normalized_sequence'] = event.normalized_sequence
    return value


def encode_frameshift_context(evidence):
    """Encode supporting read occurrences; their sum is not the legacy per-read count."""
    grouped = defaultdict(int)
    for record in evidence:
        grouped[(record.repeat_occurrence, record.observed_unit, record.events)] += 1

    encoded_contexts = []
    for (occurrence, observed_unit, events), support in grouped.items():
        context = {'read_occurrence_support': support, 'repeat_occurrence': occurrence,
                   'observed_unit': observed_unit,
                   'events': [_event_dict(event) for event in events]}
        encoded = json.dumps(context, sort_keys=True, separators=(',', ':'))
        encoded_contexts.append(encoded)
    encoded_contexts.sort()
    return '{"v":1,"contexts":[%s]}' % ','.join(encoded_contexts)
