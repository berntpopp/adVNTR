"""Dense evidence rows with independent trial and attribution-set audits.

Source read support is preserved separately from occurrence support. Native
scoring still compares cardinalities; the stronger aggregate subset failure is
returned as calibration ineligibility, never normalized into a different call.
"""
from collections import OrderedDict

from advntr.frameshift_capture_spans import _integer, _object
from advntr.frameshift_opportunities import _signature_supports, parse_components


_FIELDS = set(('candidate', 'support', 'opportunities', 'support_occurrence_ids', 'legacy_support',
               'state_occurrence_ids', 'pattern_index', 'ru_bp_coverage', 'ru_length',
               'ru_bp_coverage_ratio', 'avg_bp_coverage'))
_SOURCE_FIELDS = (_FIELDS - set(('support_occurrence_ids', 'state_occurrence_ids'))
                  | set(('support_identities', 'state_identities', 'opportunity_spans', 'legacy_states')))


def _ids(values, size):
    if not isinstance(values, list):
        raise ValueError('occurrence IDs must be a list')
    for value in values:
        _integer(value, 0, 'occurrence ID')
        if value >= size:
            raise ValueError('row occurrence ID is outside the inventory')
    if values != sorted(set(values)):
        raise ValueError('row occurrence IDs must be sorted and unique')
    return set(values)


def _state(value):
    if not isinstance(value, basestring) or not value:
        raise ValueError('candidate state must be a nonempty string')
    return value


def _diagnostics(index, geometry, is_haploid):
    unit = geometry.get(index)
    if unit is None:
        return None, None, None, None
    length, total, copies = len(unit['sequence']), unit['ru_bp_coverage'], unit['reference_copies']
    _integer(length, 1, 'unit length')
    _integer(total, 0, 'unit base coverage')
    _integer(copies, 1, 'reference copy count')
    ratio = int(round(total / float(length)))
    average = float(total) / length / (1 if is_haploid else 2) / copies
    return total, length, ratio, average


def _row(row, inventory, geometry, is_haploid):
    _object(row, _FIELDS, 'evidence row')
    state = _state(row['candidate'])
    for name in ('support', 'opportunities', 'legacy_support'):
        _integer(row[name], 0, name)
    own_support = _ids(row['support_occurrence_ids'], len(inventory.occurrences))
    if row['support'] != len(own_support):
        raise ValueError('row support differs from distinct occurrence IDs')
    components = parse_components(state)
    index = components[0][2] if components else None
    spans, trials = [], set()
    for span_id, (signature, ids) in enumerate(zip(inventory.signatures, inventory.span_occurrence_ids)):
        if components is not None and _signature_supports(signature, components):
            spans.append((span_id, len(ids)))
            trials.update(ids)
    if row['opportunities'] != len(trials) or not own_support.issubset(trials):
        raise ValueError('row opportunities or support ownership differ from its independent trial set')
    for name in ('ru_bp_coverage', 'ru_length', 'ru_bp_coverage_ratio'):
        if row[name] is not None:
            _integer(row[name], 0, name)
    diagnostics = _diagnostics(index, geometry, is_haploid)
    actual = tuple(row[name] for name in ('ru_bp_coverage', 'ru_length', 'ru_bp_coverage_ratio', 'avg_bp_coverage'))
    if row['pattern_index'] != index or actual != diagnostics or any(type(value) is bool for value in actual):
        raise ValueError('row unit diagnostics differ from the bound geometry')
    attribution = row['state_occurrence_ids']
    if not isinstance(attribution, dict):
        raise ValueError('state attribution must be a mapping')
    attributed = {}
    for emitted, ids in attribution.items():
        _state(emitted)
        identities = _ids(ids, len(inventory.occurrences))
        if not identities.issubset(own_support):
            raise ValueError('state attribution is outside its source row support')
        attributed[emitted] = tuple(sorted(inventory.occurrences[identity] for identity in ids))
    native = dict((name, value) for name, value in row.items()
                  if name not in ('support_occurrence_ids', 'state_occurrence_ids'))
    native.update(support_identities=tuple(sorted(inventory.occurrences[identity] for identity in row['support_occurrence_ids'])),
                  state_identities=attributed, legacy_states=sorted(attributed), opportunity_spans=tuple(spans))
    return native, trials


def decode_rows(rows, inventory, geometry, is_haploid):
    """Return native exact-caller rows and states failing the stronger subset audit.

    A missing attributed state's own row is retained as ineligible, since no own
    trial set exists to justify it. Empty opportunity-only rows remain inventory;
    neither this decoder nor the audit manufactures production candidate visits.
    """
    if type(is_haploid) is not bool or not isinstance(rows, list):
        raise ValueError('row collection must be a list and ploidy flag a boolean')
    records, trials, attributed = OrderedDict(), {}, {}
    previous = None
    for row in rows:
        native, own_trials = _row(row, inventory, geometry, is_haploid)
        state = native['candidate']
        if previous is not None and state <= previous:
            raise ValueError('evidence rows must have unique sorted candidate states')
        previous = state
        records[state], trials[state] = native, own_trials
        for emitted, ids in row['state_occurrence_ids'].items():
            attributed.setdefault(emitted, set()).update(ids)
    ineligible = tuple(sorted(state for state, identities in attributed.items()
                              if state not in trials or not identities.issubset(trials[state])))
    return records, ineligible


def encode_rows(records, inventory, geometry, is_haploid):
    """Encode source rows, then prove that no original field was silently changed."""
    ids = dict((pair, identity) for identity, pair in enumerate(inventory.occurrences))
    rows = []
    for state in sorted(records):
        source = records[state]
        _object(source, _SOURCE_FIELDS, 'source evidence row')
        if source['candidate'] != state:
            raise ValueError('source candidate key differs from its row')
        row = dict((name, source[name]) for name in _FIELDS
                   if name not in ('support_occurrence_ids', 'state_occurrence_ids'))
        try:
            row['support_occurrence_ids'] = sorted(ids[pair] for pair in source['support_identities'])
            row['state_occurrence_ids'] = dict((emitted, sorted(ids[pair] for pair in pairs))
                                                for emitted, pairs in source['state_identities'].items())
        except KeyError:
            raise ValueError('source row identity is absent from the span inventory')
        rows.append(row)
    decoded, _ineligible = decode_rows(rows, inventory, geometry, is_haploid)
    if dict(decoded) != dict(records):
        raise ValueError('source evidence fields differ from the independently reconstructed rows')
    return rows
