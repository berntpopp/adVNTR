"""Closed anonymous occurrence/span inventory for the proposed capture-v2 ABI.

Dense IDs are local list indices, never read names or hashes of read names.
Production span order is retained; occurrence order is canonical across Python
versions. Masks use canonical lowercase hex (zero is '0', no prefix or leading
zeros), so JSON consumers cannot round bits above the binary64 integer range.

This unused pure leaf neither writes captures nor advertises replay. Stronger
attribution audits belong above it: a counted state can have identities outside
its own trial set even when the production cardinality guard k <= N passes.
"""
import numbers
import re
from collections import namedtuple


DecodedSpans = namedtuple('DecodedSpans', 'occurrences signatures span_occurrence_ids')
_KINDS = ('complete', 'prefix', 'suffix', 'partial_start', 'partial_end')
_LABELS = {'prefix': 'prefix_flank', 'suffix': 'suffix_flank',
           'partial_start': 'partial_start', 'partial_end': 'partial_end'}
_REVERSE_LABELS = dict((value, key) for key, value in _LABELS.items())
_SPAN_FIELDS = set(('pattern_index', 'reached', 'inserted', 'saw_start', 'saw_end', 'occurrence_ids'))


def _integer(value, minimum, label):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError('%s must be an integer >= %s' % (label, minimum))
    return value


def _object(value, fields, label):
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError('%s fields differ from the closed span contract' % label)


def _geometry(unit_lengths, read_length, selected_read_count):
    _integer(read_length, 1, 'read length')
    _integer(selected_read_count, 0, 'selected read count')
    if not isinstance(unit_lengths, dict) or not unit_lengths:
        raise ValueError('unit geometry must be a nonempty mapping')
    for index, length in unit_lengths.items():
        if not isinstance(index, basestring) or re.match(r'^[1-9][0-9]*\Z', index) is None:
            raise ValueError('unit index must be a canonical positive integer string')
        _integer(length, 1, 'unit length')


def _occurrence(value, selected_read_count):
    _object(value, set(('read_index', 'kind', 'ordinal')), 'occurrence')
    index = _integer(value['read_index'], 0, 'read index')
    if index >= selected_read_count:
        raise ValueError('occurrence read index is outside the selected read roster')
    kind, ordinal = value['kind'], value['ordinal']
    if not isinstance(kind, basestring) or kind not in _KINDS:
        raise ValueError('unknown occurrence kind')
    if kind == 'complete':
        _integer(ordinal, 0, 'complete occurrence ordinal')
        label = ordinal
    else:
        if ordinal is not None:
            raise ValueError('non-complete occurrence ordinal must be null')
        label = _LABELS[kind]
    return (index, label), (index, _KINDS.index(kind), -1 if ordinal is None else ordinal)


def _mask(value, limit):
    if not isinstance(value, basestring) or re.match(r'^(0|[1-9a-f][0-9a-f]*)\Z', value) is None:
        raise ValueError('span masks must be canonical lowercase hexadecimal')
    # Bound text before parsing; no unbounded giant integer from a tiny geometry.
    if len(value) > (limit + 4) // 4:
        raise ValueError('span mask exceeds its model geometry')
    result = int(value, 16)
    if result.bit_length() > limit + 1:
        raise ValueError('span mask exceeds its model geometry')
    return result


def decode_spans(document, unit_lengths, read_length, selected_read_count):
    """Validate closed fields, canonical IDs and unique ownership independently.

    Returns immutable native anonymous pairs, source signatures, and per-span ID
    tuples. The original observation names cannot be reconstructed from this ABI.
    """
    _geometry(unit_lengths, read_length, selected_read_count)
    _object(document, set(('occurrences', 'spans')), 'inventory')
    if not isinstance(document['occurrences'], list) or not isinstance(document['spans'], list):
        raise ValueError('occurrences and spans must be ordered lists')
    occurrences, keys = [], []
    for row in document['occurrences']:
        pair, key = _occurrence(row, selected_read_count)
        occurrences.append(pair)
        keys.append(key)
    if keys != sorted(set(keys)):
        raise ValueError('occurrences must be unique and in canonical order')
    signatures, span_ids, owned = [], [], set()
    for row in document['spans']:
        _object(row, _SPAN_FIELDS, 'span')
        index = row['pattern_index']
        if not isinstance(index, basestring) or index not in set(unit_lengths) | set(('prefix', 'suffix')):
            raise ValueError('span pattern is outside the model geometry')
        limit = read_length if index in ('prefix', 'suffix') else unit_lengths[index]
        reached, inserted = _mask(row['reached'], limit), _mask(row['inserted'], limit)
        if type(row['saw_start']) is not bool or type(row['saw_end']) is not bool:
            raise ValueError('span terminator flags must be booleans')
        ids = row['occurrence_ids']
        if not isinstance(ids, list) or not ids:
            raise ValueError('every span must own a nonempty occurrence ID list')
        for identity in ids:
            _integer(identity, 0, 'occurrence ID')
            if identity >= len(occurrences):
                raise ValueError('span occurrence ID is outside the inventory')
            label = occurrences[identity][1]
            expected_flank = _LABELS.get(index)
            if ((expected_flank is not None and label != expected_flank)
                    or (expected_flank is None and label in ('prefix_flank', 'suffix_flank'))):
                raise ValueError('span pattern and occurrence kind differ')
        if ids != sorted(set(ids)) or owned.intersection(ids):
            raise ValueError('occurrence IDs must be sorted, unique, and owned by exactly one span')
        owned.update(ids)
        signature = (index, reached, inserted, row['saw_start'], row['saw_end'])
        if signature in signatures:
            raise ValueError('duplicate span signature')
        signatures.append(signature)
        span_ids.append(tuple(ids))
    if owned != set(range(len(occurrences))):
        raise ValueError('every occurrence must belong to exactly one span')
    return DecodedSpans(tuple(occurrences), tuple(signatures), tuple(span_ids))


def encode_spans(source_spans, unit_lengths, read_length, selected_read_count):
    """Freeze the counter's ordered span map without exporting observation names.

    Repeated identical observations within a signature deduplicate as finalise
    does. A selected-read/occurrence pair with contradictory names or signatures
    is refused, since anonymizing that inconsistency would silently change counts.
    """
    _geometry(unit_lengths, read_length, selected_read_count)
    rows, identities, names = [], {}, {}
    for signature, observations in source_spans.items():
        index, reached, inserted, start, end = signature
        _integer(reached, 0, 'reached mask')
        _integer(inserted, 0, 'inserted mask')
        pairs = set()
        for read_index, name, label in observations:
            kind = 'complete' if isinstance(label, numbers.Integral) and not isinstance(label, bool) else _REVERSE_LABELS.get(label)
            row = {'read_index': read_index, 'kind': kind, 'ordinal': label if kind == 'complete' else None}
            pair, key = _occurrence(row, selected_read_count)
            if pair in names and names[pair] != name:
                raise ValueError('one anonymous observation has conflicting source names')
            names[pair] = name
            identities[pair] = (key, row)
            pairs.add(pair)
        rows.append(({'pattern_index': index, 'reached': '%x' % reached, 'inserted': '%x' % inserted,
                      'saw_start': start, 'saw_end': end}, pairs))
    ordered = sorted(identities, key=lambda pair: identities[pair][0])
    ids = dict((pair, ordinal) for ordinal, pair in enumerate(ordered))
    document = {'occurrences': [identities[pair][1] for pair in ordered], 'spans': []}
    for row, pairs in rows:
        row['occurrence_ids'] = sorted(ids[pair] for pair in pairs)
        document['spans'].append(row)
    decode_spans(document, unit_lengths, read_length, selected_read_count)
    return document
