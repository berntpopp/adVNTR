"""Capture v2 completion I/O after native decisions and all binding checks.

The command owns the new sink and private assets. This writer appends only a
validated completed locus, under a lock, refusing duplicate loci and torn or
incompatible predecessors. Process success and full roster checks remain the
controller's responsibility; a partial failed write must never be skipped.
"""
import fcntl
import json
import os
import stat

from advntr.capabilities import canonical_bytes
from advntr.capture_policy import policy_document
from advntr.frameshift_capture_model import model_document, decode_model, geometry_document, decode_geometry
from advntr.frameshift_capture_record import decode_capture, digest
from advntr.frameshift_capture_rows import encode_rows, decode_rows
from advntr.frameshift_capture_spans import encode_spans, decode_spans
from advntr.frameshift_replay_policy import caller_policy_document
from advntr.frameshift_visit_document import visit_document, decision_warnings
from advntr.models import load_unique_vntrs_data
from advntr.mutation_keys import encode_frameshift_context


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('duplicate field in existing capture record')
        result[key] = value
    return result


def _constant(_value):
    raise ValueError('nonfinite number in existing capture record')


def _append_completed(path, document):
    """Append one checked record without repairing or hiding prior evidence loss."""
    encoded = canonical_bytes(document) + '\n'
    descriptor = os.open(path, os.O_RDWR | os.O_APPEND | os.O_NOFOLLOW | os.O_NONBLOCK)
    handle = os.fdopen(descriptor, 'r+b')
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError('capture sink must remain a regular file')
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        for line in handle:
            if not line.endswith('\n'):
                raise ValueError('capture sink contains a torn prior record')
            previous = json.loads(line, object_pairs_hook=_pairs, parse_constant=_constant)
            decoded = decode_capture(previous)
            if decoded.model.vntr_id == document['locus']['vntr_id']:
                raise ValueError('capture sink already contains this completed VNTR')
            for field in ('producer', 'capture_policy', 'caller_policy'):
                if canonical_bytes(previous[field]) != canonical_bytes(document[field]):
                    raise ValueError('capture sink predecessor belongs to an incompatible run')
            for field in ('model_sha256', 'background_sha256', 'loaded_background_sha256'):
                if previous['assets'][field] != document['assets'][field]:
                    raise ValueError('capture sink predecessor uses different run assets')
        handle.seek(0, os.SEEK_END)
        handle.write(encoded)
        handle.flush()
        os.fsync(descriptor)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        handle.close()


def write_completed_capture(finder, counter, ru_bp_coverage, reference_order, selected_read_count):
    """Verify loaded reference fields against the queried snapshot before writing."""
    context = finder.run_context
    assets = context.capture_assets
    binding = assets.document(context.background)
    model = model_document(finder.reference_vntr)
    loaded = load_unique_vntrs_data(db_file=context.model_path, target_vids=[model['vntr_id']])
    if len(loaded) != 1 or canonical_bytes(model_document(loaded[0])) != canonical_bytes(model):
        raise ValueError('capture reference object differs from the queried model snapshot')
    if context.background is None:
        from advntr.vntr_finder import VNTRFinder
        if finder.identify_frameshift is not VNTRFinder.identify_frameshift:
            raise ValueError('capture v2 requires the native legacy statistic implementation')
    read_length = finder.hmm.read_length_used_to_build_model
    unit_geometry = geometry_document(decode_model(model), ru_bp_coverage)
    geometry, _reference_order = decode_geometry(unit_geometry, decode_model(model))
    inventory = encode_spans(counter._spans, dict(counter._hmm_match_count), read_length, selected_read_count)
    spans = decode_spans(inventory, dict(counter._hmm_match_count), read_length, selected_read_count)
    rows = encode_rows(finder.last_frameshift_opportunities, spans, geometry, finder.is_haploid)
    _records, ineligible = decode_rows(rows, spans, geometry, finder.is_haploid)
    binding['assets']['model_locus_sha256'] = digest(model)
    traversal, boundaries, visits = finder.last_frameshift_traversal, finder.last_frameshift_boundaries, finder.last_frameshift_visits
    document = dict(inventory, **{
        'schema_version': 'advntr-frameshift-capture-v2', 'completion': 'completed-vntr',
        'producer': binding['producer'], 'assets': binding['assets'], 'model_locus': model,
        'loaded_background': assets.loaded_background_document(), 'capture_policy': policy_document(context.capture),
        'caller_policy': caller_policy_document(context.capture, context.frameshift),
        'locus': {'vntr_id': model['vntr_id'], 'read_length': read_length, 'is_haploid': finder.is_haploid,
                  'selected_read_count': selected_read_count},
        'unit_geometry': unit_geometry, 'reference_order': list(reference_order),
        'flank_boundaries': {'suffix_min_position': boundaries[0], 'prefix_max_position': boundaries[1]},
        'warnings': decision_warnings(visits, ineligible), 'evidence_rows': rows,
        'candidate_traversal': {'repeat_candidates': [list(pair) for pair in traversal.repeat_candidates],
                                'flank_candidates': [list(pair) for pair in traversal.flank_candidates]},
        'decision_visits': [visit_document(visit) for visit in visits]})
    decode_capture(document)
    assets.document(context.background)
    if canonical_bytes(model_document(finder.reference_vntr)) != canonical_bytes(model):
        raise ValueError('capture reference object changed during completion')
    _append_completed(context.capture_path, document)
    return document


def complete_frameshift_calls(finder, frameshifts, counter, ru_bp_coverage, reference_order, selected_read_count):
    """Keep called-context assertions before any v2 record can claim completion."""
    for state, _count, _coverage, _pval in frameshifts:
        evidence = finder.last_frameshift_evidence[state]
        if not evidence:
            raise AssertionError('called frameshift lacks context evidence: %s' % state)
        finder.last_frameshift_context[state] = encode_frameshift_context(evidence)
    context = getattr(finder, 'run_context', None)
    if context is not None and context.capture_version == 2:
        write_completed_capture(finder, counter, ru_bp_coverage, reference_order, selected_read_count)
    return frameshifts if frameshifts else None
