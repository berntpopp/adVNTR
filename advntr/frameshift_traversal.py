"""Frozen caller traversal and pure boundary/read-support visit plans.

Evidence inventory is deliberately not an input: an occurrence-only candidate
must not manufacture a caller visit. Flank order is captured from production,
not recovered by iterating a newly decoded dict. Plans distinguish an absent
visit, boundary/support suppression and readiness for subsequent scoring; they
are not genotype decisions or a claim that capture/replay v2 is implemented.

The two flank branches remain independent. Crucially, the suffix support
failure executes the source loop's continue, which skips the prefix branch for
that same candidate. A different supporting-read policy can change reachability.
"""
import numbers
from collections import namedtuple

from advntr.frameshift_decisions import passes_support, validate_policy


_TraversalTuple = namedtuple('_CandidateTraversal', 'repeat_candidates flank_candidates')
VisitPlan = namedtuple('VisitPlan', 'ordinal source_index site state read_support repeat_unit_index disposition')


class CandidateTraversal(_TraversalTuple):
    """Closed immutable source candidates, independent of the diagnostic policy."""
    __slots__ = ()

    def __new__(cls, repeat_candidates, flank_candidates):
        result = _TraversalTuple.__new__(cls, repeat_candidates, flank_candidates)
        validate_traversal(result)
        return result

    def __reduce_ex__(self, protocol):
        return type(self), tuple(self)

    @classmethod
    def _make(cls, values):
        return cls(*values)

    def _replace(self, **fields):
        if set(fields) - set(self._fields):
            raise ValueError('unknown candidate traversal fields')
        return type(self)(*(fields.get(name, getattr(self, name)) for name in self._fields))


def _integer(value, name, minimum=None):
    if (isinstance(value, bool) or not isinstance(value, numbers.Integral)
            or (minimum is not None and value < minimum)):
        raise ValueError('%s must be an integer%s' % (name, '' if minimum is None else ' >= %s' % minimum))


def _validate_candidates(candidates):
    if not isinstance(candidates, tuple):
        raise ValueError('candidate collections must be frozen tuples')
    seen = set()
    for pair in candidates:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError('each candidate must be a frozen state/count pair')
        state, count = pair
        if not isinstance(state, basestring) or not state:
            raise ValueError('candidate state must be a nonempty string')
        if state in seen:
            raise ValueError('duplicate candidate within source traversal')
        seen.add(state)
        _integer(count, 'candidate read support', 1)


def validate_traversal(traversal):
    """Validate frozen source candidates without interpreting them as evidence rows."""
    if not isinstance(traversal, CandidateTraversal) or len(traversal) != 2:
        raise ValueError('traversal must be a complete CandidateTraversal')
    _validate_candidates(traversal.repeat_candidates)
    _validate_candidates(traversal.flank_candidates)
    if tuple(sorted(traversal.repeat_candidates, key=lambda pair: (pair[1], pair[0]))) != traversal.repeat_candidates:
        raise ValueError('repeat traversal must retain production count/state order')
    return traversal


def freeze_traversal(mutations, flank_mutations):
    """Freeze the source maps before any policy-dependent continue or decision.

    Repeat candidates use the source's count/state ordering. Flank candidates
    preserve that source map's actual items order, including OrderedDict inputs.
    """
    return CandidateTraversal(tuple(sorted(mutations.items(), key=lambda pair: (pair[1], pair[0]))),
                              tuple(flank_mutations.items()))


def plan_visits(traversal, reference_order, suffix_min_position, prefix_max_position, policy):
    """Return ordered production gate visits, without scoring or reading global state.

    Boundary equality passes in both branches, with opposite comparisons. The
    borrowed repeat-unit index is read only after the boundary passes, but before
    supporting reads are checked, matching the original lookup/error order.
    Missing geometry is not filled in: a reached invalid reference index raises.
    A negative suffix boundary is possible for long matching reference flanks.
    """
    validate_traversal(traversal)
    validate_policy(policy)
    _integer(suffix_min_position, 'suffix boundary')
    _integer(prefix_max_position, 'prefix boundary', 0)
    visits = []

    def append(source_index, site, state, count, index, disposition):
        visits.append(VisitPlan(len(visits), source_index, site, state, count, index, disposition))

    for source_index, (state, count) in enumerate(traversal.repeat_candidates):
        # Keep the first component's index even for a compound candidate.
        index = state.split('&')[0].split('_')[1]
        disposition = 'ready-to-score' if passes_support(count, policy) else 'insufficient-read-support'
        append(source_index, 'repeat', state, count, index, disposition)

    for source_index, (state, count) in enumerate(traversal.flank_candidates):
        position = int(state.split('_')[0][1:])
        if 'suffix' in state:
            if position >= suffix_min_position:
                index = reference_order[1]
                supported = passes_support(count, policy)
                append(source_index, 'suffix', state, count, index,
                       'ready-to-score' if supported else 'insufficient-read-support')
                if not supported:
                    continue
            else:
                append(source_index, 'suffix', state, count, None, 'outside-boundary')
        if 'prefix' in state:
            if position <= prefix_max_position:
                index = reference_order[-2]
                supported = passes_support(count, policy)
                append(source_index, 'prefix', state, count, index,
                       'ready-to-score' if supported else 'insufficient-read-support')
                if not supported:
                    continue
            else:
                append(source_index, 'prefix', state, count, None, 'outside-boundary')
    return tuple(visits)
