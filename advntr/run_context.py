"""Run-local policy and separately loaded assets; no artifact identity claim.

Production commands provide this context. The legacy runtime_value fallback is
only for direct library callers that have not adopted the explicit contract.
"""
from collections import namedtuple

from advntr import settings
from advntr.capture_policy import resolve_capture_policy, validate_capture_policy
from advntr.frameshift_background import BackgroundModel
from advntr.frameshift_decisions import resolve_policy, validate_policy


_ContextTuple = namedtuple('_RunContext', 'capture frameshift background capture_path model_path')
_LEGACY_FIELDS = {
    'maximum_error_rate': 'MAX_ERROR_RATE', 'threads': 'CORES',
    'minimum_read_length': 'MIN_READ_LENGTH', 'prune_reverse': 'PRUNE_REVERSE_DECODE',
    'filter_adapter_readthrough': 'FILTER_ADAPTER_READTHROUGH',
    'minimum_read_match_ratio': 'MIN_READ_MATCH_RATIO',
    'minimum_relative_ru_coverage': 'MIN_RELATIVE_RU_COVERAGE',
    'use_reference_alignment': 'USE_REF_ALIGNMENT',
    'fully_covered_ru_only': 'USE_ONLY_FULLY_COVERED_RU',
    'enhanced_hmm': 'USE_ENHANCED_HMM', 'trained_hmms': 'USE_TRAINED_HMMS',
}


class RunContext(_ContextTuple):
    """Validated immutable container; background is a separately loaded run asset."""
    __slots__ = ()

    def __new__(cls, capture, frameshift, background, capture_path, model_path):
        result = _ContextTuple.__new__(cls, capture, frameshift, background, capture_path, model_path)
        validate_context(result)
        return result

    def __reduce_ex__(self, protocol):
        return type(self), tuple(self)

    @classmethod
    def _make(cls, values):
        return cls(*values)

    def _replace(self, **fields):
        if set(fields) - set(self._fields):
            raise ValueError('unknown run context fields')
        return type(self)(*(fields.get(name, getattr(self, name)) for name in self._fields))

    @property
    def effective_match_ratio(self):
        """The historical adapter default, explicitly passed even when raw None."""
        ratio = self.capture.minimum_read_match_ratio
        return 0.60 if ratio is None else ratio


def validate_context(context):
    """Validate policy types, conditional background and separate path assets."""
    if not isinstance(context, RunContext) or len(context) != len(_ContextTuple._fields):
        raise ValueError('run context must be a complete RunContext')
    validate_capture_policy(context.capture)
    validate_policy(context.frameshift)
    if context.capture.caller_mode == 'exact':
        if not isinstance(context.background, BackgroundModel):
            raise ValueError('exact run context requires a loaded background model')
    elif context.background is not None:
        raise ValueError('legacy run context cannot contain an active background model')
    for name in ('capture_path', 'model_path'):
        value = getattr(context, name)
        if name == 'capture_path' and value is None:
            continue
        if not isinstance(value, basestring) or not value:
            raise ValueError('%s must be a nonempty path' % name)
    return context


def runtime_value(owner, field):
    """Use the explicit run value, retaining direct-library legacy compatibility."""
    context = getattr(owner, 'run_context', None)
    if context is not None:
        if field == 'minimum_read_match_ratio':
            return context.effective_match_ratio
        return getattr(context.capture, field)
    if field == 'legacy_error_rate':
        return 0.01  # The public identify_frameshift default was bound at import.
    return getattr(settings, _LEGACY_FIELDS[field])


def command_policies(args):
    """Resolve before asset I/O, without inheriting mutable prior-command settings."""
    get = lambda name, default=None: getattr(args, name, default)
    if get('pacbio', False) and get('nanopore', False):
        raise ValueError('--pacbio and --nanopore are mutually exclusive')
    if get('frameshift', False) and (get('pacbio', False) or get('nanopore', False)):
        raise ValueError('--frameshift is unsupported with --pacbio or --nanopore; use the short-read workflow')
    frameshift = resolve_policy(get('frameshift_pvalue_cutoff'), get('min_frameshift_read_support'))
    capture = resolve_capture_policy(
        platform='nanopore' if get('nanopore', False) else ('pacbio' if get('pacbio', False) else 'illumina'),
        frameshift_mode=get('frameshift', False), is_haploid=get('haploid', False),
        caller_mode='exact' if get('exact_frameshift_caller', False) else 'legacy',
        threads=get('threads', 1), minimum_read_length=get('min_read_length'),
        prune_reverse=get('prune_reverse', False),
        filter_adapter_readthrough=get('filter_adapter_readthrough', False),
        minimum_read_match_ratio=get('min_read_match_ratio'),
        minimum_relative_ru_coverage=get('rare_unit_coverage_guard'),
        use_reference_alignment=not get('noref_aln', False),
        fully_covered_ru_only=get('fullru', False),
        enhanced_hmm=settings.USE_ENHANCED_HMM, trained_hmms=settings.USE_TRAINED_HMMS)
    return capture, frameshift


def bind_owner(owner, context, frameshift_policy, is_haploid, is_frameshift_mode):
    """Bind identical context through the analyzer/finder; reject conflicting inputs."""
    owner.run_context = None if context is None else validate_context(context)
    if context is not None:
        if (context.capture.is_haploid != is_haploid
                or context.capture.frameshift_mode != is_frameshift_mode
                or (frameshift_policy is not None and frameshift_policy != context.frameshift)):
            raise ValueError('run context conflicts with analyzer/finder arguments')
        return context.frameshift
    return resolve_policy() if frameshift_policy is None else validate_policy(frameshift_policy)
