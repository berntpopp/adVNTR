"""Adapter read-through and short-match detection and filtering (Issue #4).

In short-insert sequencing libraries (e.g. fragment length < read length), sequencing
reads extend beyond the genomic insert into Illumina sequencing adapters.
In MUC1, the first ~20 bp of the read matches the beginning of a repeat unit (such as
RU7), while the remaining ~130 bp consist of Illumina TruSeq adapter sequence:
    GGCCGAGGTGACACCGTGGG AGATCGGAAGAGCACACGTCTGAACTCC
    [-- 20 bp RU match --][------ Illumina TruSeq adapter ------]

Because MAPQ > 0, the read bypassed recruit_read() in vntr_finder.py:948.
Furthermore, is_matching_state() historically treated insertion states ('I') as match
states, so matches >= 0.9 * read_length was a tautology (M + I == read_length).
The DP aligns the adapter sequence into insertion states of a partially observed
repeat unit (partial_end), producing confident false-positive frameshift calls
(e.g. D20_7&D21_7&D22_7&I22_7_A_LEN9, I22_7_A_LEN7 on example_6c28).

This module provides detection and filtering for adapter read-through reads and
adapter-driven candidate mutations.
"""

import re


ADAPTER_KMERS = (
    'AGATCGGAAGAGC',  # Universal TruSeq 13-mer
    'GCTCTTCCGATCT',  # TruSeq reverse complement 13-mer
    'CTGTCTCTTATACACATCT',  # Nextera transposase
    'AGATGTGTATAAGAGACAG',  # Nextera reverse complement
    'AGATCGGAA',      # 9-mer core
    'TTCCGATCT',      # 9-mer core RC
    'AGATCGGA',       # 8-mer core
    'TCCGATCT',       # 8-mer core RC
)

# Long full-length adapter sequences
LONG_ADAPTER_KMERS = (
    'AGATCGGAAGAGC',        # Universal TruSeq 13-mer
    'GCTCTTCCGATCT',        # TruSeq reverse complement 13-mer
    'CTGTCTCTTATACACATCT',  # Nextera transposase
    'AGATGTGTATAAGAGACAG',  # Nextera reverse complement
)

DEFAULT_MIN_GENUINE_MATCH_RATIO = 0.60


def contains_adapter_kmer(sequence, kmers=None):
    """Check if sequence contains any known adapter k-mer."""
    if not sequence:
        return False
    if kmers is None:
        kmers = ADAPTER_KMERS
    seq_upper = sequence.upper()
    for kmer in kmers:
        if kmer in seq_upper:
            return True
    return False


def count_genuine_matches(vpath):
    """Count only pure match states ('M') in vpath, strictly excluding insertion ('I')."""
    if not vpath:
        return 0
    count = 0
    for idx, state in vpath[1:-1]:
        name = state.name
        if name.startswith('M') and not name.startswith('M_random'):
            count += 1
    return count


def genuine_match_ratio(vpath, read_length):
    """Return the fraction of read bases that aligned to genuine match states ('M')."""
    if read_length <= 0 or not vpath:
        return 0.0
    return float(count_genuine_matches(vpath)) / float(read_length)


def is_adapter_readthrough(sequence, vpath=None, min_match_ratio=DEFAULT_MIN_GENUINE_MATCH_RATIO):
    """Return True if read sequence contains adapter k-mers or fails genuine match ratio."""
    if not sequence:
        return False

    if min_match_ratio is None:
        min_match_ratio = DEFAULT_MIN_GENUINE_MATCH_RATIO

    # If vpath is available, evaluate genuine match ratio
    if vpath is not None:
        ratio = genuine_match_ratio(vpath, len(sequence))
        # Reads failing genuine match ratio are poorly matching / adapter read-through
        if min_match_ratio > 0 and ratio < min_match_ratio:
            return True
        # Reads with high genuine match ratio (>= 0.75) are genuine VNTR reads.
        # Only reject if they contain a full-length adapter (>= 13-mer).
        if ratio >= 0.75:
            seq_upper = sequence.upper()
            return any(kmer in seq_upper for kmer in LONG_ADAPTER_KMERS)

    # Check for adapter sequence
    if contains_adapter_kmer(sequence):
        return True

    return False


def is_adapter_at_insertion(observed_unit, ins_start, ins_len, kmers=None):
    """Check if an adapter k-mer overlaps or spans an insertion boundary."""
    if not observed_unit or ins_len <= 0 or ins_start is None:
        return False
    if kmers is None:
        kmers = ADAPTER_KMERS
    obs_upper = observed_unit.upper()
    ins_end = ins_start + ins_len
    for kmer in kmers:
        klen = len(kmer)
        pos = obs_upper.find(kmer)
        while pos != -1:
            k_end = pos + klen
            if max(pos, ins_start) < min(k_end, ins_end):
                return True
            pos = obs_upper.find(kmer, pos + 1)
    return False


def is_adapter_driven_mutation(candidate_state, inserted_seq='', observed_unit='', ins_start=None):
    """Check if a candidate mutation in a partial repeat unit is driven by adapter sequence."""
    if inserted_seq and contains_adapter_kmer(inserted_seq):
        return True
    if observed_unit and ins_start is not None and inserted_seq:
        return is_adapter_at_insertion(observed_unit, ins_start, len(inserted_seq))
    if observed_unit and contains_adapter_kmer(observed_unit) and ins_start is None:
        return True
    return False
