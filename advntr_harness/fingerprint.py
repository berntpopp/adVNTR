"""Model and input fingerprints.

A matching output digest proves nothing unless you also know that both runs read the
same inputs and decoded against the same model. Both failure modes are reachable here:
13 corpus files extract zero reads, and `select_illumina_reads` silently rebuilds the
model rather than using the one it was handed.
"""
import hashlib

from advntr_harness.oracle import EMPTY_STREAM_SENTINEL, logp_to_hex


def input_attestation(source_file, sequences):
    """Count and digest the ordered input sequences for one source file."""
    digest = hashlib.sha256()
    count = 0
    for sequence in sequences:
        digest.update(sequence)
        digest.update('\n')
        count += 1
    return {
        'source_file': source_file,
        'eligible_count': count,
        'input_digest': EMPTY_STREAM_SENTINEL if count == 0 else digest.hexdigest(),
    }


def model_fingerprint(model, reference_vntr):
    """Identify the baked model a run actually decoded against.

    Take this from `finder.hmm` AFTER select_illumina_reads has run, never from the
    model you passed in: advntr/vntr_finder.py:1133 unconditionally overwrites its `hmm`
    argument and rebuilds from a read length derived from `samfile.head(5)`. On the
    corpus BAMs that is 151, giving a 2565-state model -- while a hand-built
    read_length=150 model has 2559 states. Those two are not interchangeable, and a
    harness that reports one while decoding with the other is lying.
    """
    csr = hashlib.sha256()
    if hasattr(model, 'nbr_indptr'):
        for value in model.nbr_indptr:
            csr.update('%d\n' % value)
        for value in model.nbr_indices:
            csr.update('%d\n' % value)
        for value in model.nbr_logp:
            csr.update('%s\n' % logp_to_hex(value))
        csr_digest = csr.hexdigest()
    else:
        # Pristine model, before the CSR tables exist. The remaining fields still
        # identify it; recorded as a distinct value rather than a fake digest so a
        # pristine baseline can never be mistaken for a post-rewrite one.
        csr_digest = 'pre-csr'

    vntr = hashlib.sha256()
    vntr.update('%s\n%s\n%s\n%s\n' % (reference_vntr.id, reference_vntr.chromosome,
                                      reference_vntr.start_point,
                                      reference_vntr.pattern))
    for segment in reference_vntr.get_repeat_segments():
        vntr.update('%s\n' % segment)

    return {
        'n_states': int(model.n_states),
        'read_length': int(model.read_length_used_to_build_model),
        'dp_score_threshold_hex': logp_to_hex(model.dp_score_threshold),
        'csr_digest': csr_digest,
        'vntr_digest': vntr.hexdigest(),
    }


def comparable_fingerprint(fingerprint):
    """The subset of a fingerprint that must match across the rewrite.

    `csr_digest` deliberately changes when the CSR tables appear, so it is excluded:
    comparing it would make every pristine-vs-rewritten check fail for the wrong reason.
    Everything here describes the model's decoding behaviour, not its representation.
    """
    return dict((key, fingerprint[key]) for key in
                ('n_states', 'read_length', 'dp_score_threshold_hex', 'vntr_digest'))
