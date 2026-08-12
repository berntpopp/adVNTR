"""Decode-eligible read extraction, mirroring select_illumina_reads' filters exactly.

If these filters drift from advntr/vntr_finder.py:1130-1146 the harness compares a
different population than the decoder processes, and the gate becomes meaningless. Every
predicate below cites the production line it mirrors.
"""
import pysam

from advntr.utils import is_low_quality_read


def resolve_contig(samfile, chromosome):
    """Return the contig name this BAM uses for `chromosome`, or None.

    Handles UCSC ('chr1') and Ensembl ('1') naming. Returns None for RefSeq accessions
    ('NC_000001.11'): adVNTR cannot fetch those, and 13 files in the corpus are in that
    state and legitimately yield zero reads. Callers must treat None as "no data here",
    not as an error, but must also not let it pass silently as a matching empty digest.
    """
    references = set(samfile.references)
    if chromosome in references:
        return chromosome
    if chromosome.startswith('chr') and chromosome[3:] in references:
        return chromosome[3:]
    return None


def derive_read_length(samfile):
    """Reproduce advntr/vntr_finder.py:1123-1126 exactly.

    Production reads the first five records and takes `sorted(read_lengths)[3]`. That
    indexes out of range on a BAM whose head yields fewer than four records -- a real
    latent crash, mirrored here rather than papered over so the harness fails the same
    way production would.
    """
    read_lengths = []
    for read in samfile.head(5):
        read_lengths.append(len(read.seq))
    return sorted(read_lengths)[3]


def eligible_reads(alignment_path, reference_vntr, read_length=None,
                   min_read_length=None):
    """Yield (ordinal, query_name, sequence, mapq, reference_start, is_low_quality).

    `ordinal` is a dense 0-based index over yielded reads, so a later stage can assert
    that a threaded run produced the same occurrences in the same order.

    Filters mirror production:
      - is_unmapped / is_duplicate      -> advntr/vntr_finder.py:1134-1137
      - len(seq) < MIN_READ_LENGTH      -> :1138-1140, MIN_READ_LENGTH = int(rl * 0.9)
      - the span predicate              -> :1141-1142, which uses the model's
                                           `read_length`, NOT len(read.seq)
      - seq.count('N') <= 0             -> :1143
    """
    vntr_start = reference_vntr.start_point
    vntr_end = vntr_start + reference_vntr.get_length()

    samfile = pysam.AlignmentFile(alignment_path, 'rb')
    try:
        contig = resolve_contig(samfile, reference_vntr.chromosome)
        if contig is None:
            return

        if read_length is None:
            read_length = derive_read_length(samfile)
        if min_read_length is None:
            min_read_length = int(read_length * 0.9)

        ordinal = 0
        for read in samfile.fetch(contig, vntr_start, vntr_end):
            if read.is_unmapped or read.is_duplicate or read.seq is None:
                continue
            if len(read.seq) < min_read_length:
                continue
            read_end = read.reference_end
            if not read_end:
                read_end = read.reference_start + len(read.seq)
            spans = (vntr_start - read_length < read.reference_start < vntr_end
                     or vntr_start < read_end < vntr_end)
            if not spans:
                continue
            if read.seq.count('N') > 0:
                continue
            yield (ordinal, read.query_name, str(read.seq).upper(), read.mapq,
                   read.reference_start, is_low_quality_read(read))
            ordinal += 1
    finally:
        samfile.close()
