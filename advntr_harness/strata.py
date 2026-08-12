"""Tier 1 fixture selection.

Deliberate strata, not a random sample. The point is that a fixture set contains reads
that would actually catch the regressions we care about -- above all the
reverse-complement stratum, which is what upstream PR #57 deletes.

Measured across 29,998 reads, `rev_logp > logp` decides 77 of them (0.26%), and for
example_66bf it fires ZERO times in 12,608 reads. A fixture set drawn from that sample
alone would go green without ever exercising the change. Hence: every stratum must be
non-empty, and the adversarial ones are filled before the general pool.
"""
from advntr_harness.oracle import hex_to_logp

#: Every stratum the selector fills.
STRATUM_NAMES = (
    'reverse_complement_wins',
    'neg_inf_rejected',
    'lowest_surviving_logp',
    'shortest_sequence',
    'longest_sequence',
    'general',
)

#: Strata that are allowed to be empty, with the reason.
#:
#: `neg_inf_rejected` is unreachable on this model, measured rather than assumed: the
#: MUC1 hg19 model at read length 151 has dp_score_threshold = -367.2584, and even
#: deliberate garbage scores far above it -- random ACGT -321.95, poly-A -335.85,
#: poly-GC -328.38. Since production requires len(seq) >= 135, no admissible sequence
#: can drive the DP to -inf. The pruning threshold is effectively inactive.
#:
#: The stratum is kept rather than deleted so that if the threshold is ever tightened
#: (a Tier B change), fixtures start covering the rejection path automatically.
#: tests/test_characterization.py pins the unreachability so a silent tightening is
#: visible.
OPTIONAL_STRATA = {
    'neg_inf_rejected': 'dp_score_threshold -367.26 never rejects a >=135bp sequence',
}

#: Adversarial strata are filled first so a small target cannot crowd them out.
_PRIORITY = STRATUM_NAMES

_RANKED_STRATUM_SIZE = 200
_EXTREME_STRATUM_SIZE = 50


def _pair_attempts(attempts):
    """Group attempts into per-read {orientation: attempt}, preserving stream order."""
    by_read = {}
    order = []
    for attempt in attempts:
        key = (attempt.source_file, attempt.fetch_ordinal, attempt.query_name)
        if key not in by_read:
            by_read[key] = {}
            order.append(key)
        by_read[key][attempt.orientation] = attempt
    return order, by_read


def select_strata(attempts, target=2000):
    """Return (ordered unique sequences, {stratum: count}).

    Deterministic: attempts are consumed in stream order and ties resolve by that order,
    never by set or dict iteration order.
    """
    order, by_read = _pair_attempts(attempts)

    buckets = dict((name, []) for name in STRATUM_NAMES)
    finite = []
    for key in order:
        pair = by_read[key]
        forward = pair.get('fwd')
        if forward is None:
            continue
        reverse = pair.get('rev')
        sequence = forward.sequence

        if reverse is not None and hex_to_logp(forward.logp_hex) < hex_to_logp(reverse.logp_hex):
            buckets['reverse_complement_wins'].append(sequence)
        # Test the VALUE, not the status string. A genuine -inf arriving via the
        # non-ACGT KeyError path is labelled 'exception:KeyError', and testing the
        # status would miss it.
        neg_inf = float('-inf')
        if hex_to_logp(forward.logp_hex) == neg_inf and (
                reverse is None or hex_to_logp(reverse.logp_hex) == neg_inf):
            buckets['neg_inf_rejected'].append(sequence)
        if forward.exit_status == 'ok':
            finite.append(forward)
        buckets['general'].append(sequence)

    if finite:
        by_score = sorted(finite, key=lambda a: hex_to_logp(a.logp_hex))
        buckets['lowest_surviving_logp'] = [a.sequence for a in by_score[:_RANKED_STRATUM_SIZE]]
        by_length = sorted(finite, key=lambda a: len(a.sequence))
        buckets['shortest_sequence'] = [a.sequence for a in by_length[:_EXTREME_STRATUM_SIZE]]
        buckets['longest_sequence'] = [a.sequence for a in by_length[-_EXTREME_STRATUM_SIZE:]]

    selected = []
    seen = set()
    counts = dict((name, 0) for name in STRATUM_NAMES)
    for name in _PRIORITY:
        for sequence in buckets[name]:
            if sequence in seen:
                continue
            if name == 'general' and len(selected) >= target:
                break
            seen.add(sequence)
            selected.append(sequence)
            counts[name] += 1
    return selected, counts


def empty_strata(counts):
    """Names of REQUIRED strata with no members.

    A non-empty result means the fixture set is not adversarial and capture must refuse
    to write it. Strata listed in OPTIONAL_STRATA are excluded, because they are known
    unreachable on this model for a documented reason.
    """
    return sorted(name for name, count in counts.items()
                  if count == 0 and name not in OPTIONAL_STRATA)
