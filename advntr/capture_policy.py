"""Closed immutable raw-runtime policy, ready for explicit production propagation.

This module performs no I/O and does not read settings. Defaults are the verified
fresh-command defaults, including the command's one thread rather than the
host-dependent settings.CORES. Genotype binds these values through a run context. This raw contract does not
advertise a new capture or JSON policy capability.

This upstream raw contract retains None/zero read-match limits and nonnegative
rare-unit fractions above one. VNtyper generated calibrated-v2 profiles have a
narrower domain; calibrated_v2_domain_errors reports only that domain mismatch,
not eligibility of a profile, model, study, background, or executable.

Background objects, capture destinations, selected model paths/byte digests and
build identity are run-context assets, not fields or tunable leaves of this policy.
The diagnostic cutoff/support policy and recipe-v1 fitter remain independent.
"""
import math
import numbers
from collections import namedtuple


SCHEMA_VERSION = 'advntr-runtime-capture-policy-v1'
_DEFAULTS = (
    ('platform', 'illumina'), ('frameshift_mode', False), ('is_haploid', False),
    ('caller_mode', 'legacy'), ('threads', 1), ('minimum_read_length', None),
    ('prune_reverse', False), ('filter_adapter_readthrough', False),
    ('minimum_read_match_ratio', None), ('minimum_relative_ru_coverage', None),
    ('use_reference_alignment', True), ('fully_covered_ru_only', False),
    ('maximum_error_rate', 0.05), ('legacy_error_rate', 0.01),
    ('mapq_cutoff', 0), ('base_quality_cutoff', 20),
    ('maximum_low_quality_fraction', 0.10),
    ('enhanced_hmm', True), ('trained_hmms', False),
)
_FIELDS = tuple(name for name, _default in _DEFAULTS)
_PolicyTuple = namedtuple('_CapturePolicy', _FIELDS)
_BOOLEAN_FIELDS = (
    'frameshift_mode', 'is_haploid', 'prune_reverse',
    'filter_adapter_readthrough', 'use_reference_alignment',
    'fully_covered_ru_only', 'enhanced_hmm', 'trained_hmms',
)


class CapturePolicy(_PolicyTuple):
    """Complete validated raw-runtime values; use resolve_capture_policy for defaults."""

    __slots__ = ()

    def __new__(cls, *values, **fields):
        result = _PolicyTuple.__new__(cls, *values, **fields)
        validate_capture_policy(result)
        return result

    def __reduce_ex__(self, protocol):
        # Python 2 protocols 0/1 otherwise reconstruct tuple subclasses without
        # invoking __new__, bypassing validation on a forged tuple payload.
        return type(self), tuple(self)

    @classmethod
    def _make(cls, iterable):
        return cls(*iterable)

    def _replace(self, **values):
        unknown = set(values) - set(self._fields)
        if unknown:
            raise ValueError('unknown capture policy fields: %s' % sorted(unknown))
        return type(self)(*(values.get(field, getattr(self, field))
                            for field in self._fields))


def _count(value, field, minimum):
    if (isinstance(value, bool) or not isinstance(value, numbers.Integral)
            or value < minimum):
        raise ValueError('%s must be an integer >= %s' % (field, minimum))


def _fraction(value, field, maximum=1.0):
    if (not isinstance(value, float) or math.isnan(value) or math.isinf(value)
            or value < 0.0 or (maximum is not None and value > maximum)):
        raise ValueError('%s must be a finite nonnegative float%s' % (
            field, '' if maximum is None else ' <= %s' % maximum))


def validate_capture_policy(policy):
    """Revalidate complete values, including direct tuple/pickle forgery boundaries."""
    if not isinstance(policy, CapturePolicy) or len(policy) != len(_FIELDS):
        raise ValueError('capture policy must be a complete CapturePolicy')
    if policy.platform not in ('illumina', 'pacbio', 'nanopore'):
        raise ValueError('capture platform must be illumina, pacbio or nanopore')
    if policy.caller_mode not in ('legacy', 'exact'):
        raise ValueError('capture caller_mode must be legacy or exact')
    for field in _BOOLEAN_FIELDS:
        if type(getattr(policy, field)) is not bool:
            raise ValueError('%s must be a boolean' % field)
    if policy.caller_mode == 'exact' and not policy.frameshift_mode:
        raise ValueError('exact caller_mode requires the frameshift workflow')
    if not policy.enhanced_hmm or policy.trained_hmms:
        raise ValueError('unsupported HMM mode: enhanced_hmm must be true and trained_hmms false')
    _count(policy.threads, 'threads', 1)
    if policy.minimum_read_length is not None:
        _count(policy.minimum_read_length, 'minimum_read_length', 1)
    for field in ('mapq_cutoff', 'base_quality_cutoff'):
        _count(getattr(policy, field), field, 0)
    for field in ('maximum_error_rate', 'legacy_error_rate', 'maximum_low_quality_fraction'):
        _fraction(getattr(policy, field), field)
    if policy.minimum_read_match_ratio is not None:
        _fraction(policy.minimum_read_match_ratio, 'minimum_read_match_ratio')
    if policy.minimum_relative_ru_coverage is not None:
        _fraction(policy.minimum_relative_ru_coverage, 'minimum_relative_ru_coverage', None)
    return policy


def resolve_capture_policy(**values):
    """Resolve a fresh policy without inheriting any prior command or mutable settings.

    maximum_error_rate defaults to 0.05 for Illumina or 0.3 for PacBio/Nanopore.
    An explicit value is retained and validated. Other defaults never depend on
    platform. A None minimum_read_length preserves the existing per-locus derived
    int(read_length * 0.9) threshold; it does not mean an absolute zero cutoff.
    """
    unknown = set(values) - set(_FIELDS)
    if unknown:
        raise ValueError('unknown capture policy fields: %s' % sorted(unknown))
    resolved = dict(_DEFAULTS)
    resolved.update(values)
    if 'maximum_error_rate' not in values:
        resolved['maximum_error_rate'] = 0.05 if resolved['platform'] == 'illumina' else 0.3
    return CapturePolicy(**resolved)


def calibrated_v2_domain_errors(policy):
    """Fields outside generated calibrated-v2 numeric domains, without coercion.

    This is only representability of the ratio/rare-fraction fields. An empty
    result makes no claim about the rest of a calibrated bundle or applicability.
    The read-match ratio must be explicit and positive even with the filter off.
    """
    validate_capture_policy(policy)
    errors = []
    if policy.minimum_read_match_ratio is None or policy.minimum_read_match_ratio <= 0.0:
        errors.append('minimum_read_match_ratio')
    rare = policy.minimum_relative_ru_coverage
    if rare is not None and not 0.0 < rare <= 1.0:
        errors.append('minimum_relative_ru_coverage')
    return tuple(errors)


def policy_document(policy):
    """Return the complete versioned raw policy document, with no paths or assets."""
    validate_capture_policy(policy)
    parameters = dict(zip(_FIELDS, policy))
    for field in ('threads', 'minimum_read_length', 'mapq_cutoff', 'base_quality_cutoff'):
        if parameters[field] is not None:
            parameters[field] = int(parameters[field])
    return {'schema_version': SCHEMA_VERSION, 'parameters': parameters}


def policy_from_document(document):
    """Validate a decoded closed document without supplying missing/default values.

    JSON byte decoding must reject duplicate keys/nonfinite tokens at its outer
    boundary. Here all resolved fields must be present with their correct types.
    """
    if (not isinstance(document, dict)
            or set(document) != set(('schema_version', 'parameters'))
            or document['schema_version'] != SCHEMA_VERSION):
        raise ValueError('capture policy document has an invalid schema or fields')
    parameters = document['parameters']
    if not isinstance(parameters, dict) or set(parameters) != set(_FIELDS):
        raise ValueError('capture policy parameters must contain every field exactly once')
    return CapturePolicy(**parameters)
