"""Build tests/data/synthetic_revcomp.bam, the one fixture that forces the reverse
decode in advntr/read_selection.py:_decode_one to win.

Task 7 exists because the public corpus provably cannot exercise that branch: every
reverse-wins read in the 8 example BAMs is rejected before selection (TOO_SHORT or
OUT_OF_SPAN), so the Tier 3 digest is identical whether `_decode_one` skips the reverse
`model.viterbi` call or not -- demonstrated by injection, and the defect passed a full
green gate. See AGENTS.md's "reverse_complement_wins" stratum note: it fires for 77 of
29,998 corpus reads and zero times in example_66bf's 12,608. This script builds the one
read `select_illumina_reads` cannot select without decoding both orientations.

The construction is a swap, not a copy. The read's actual good-matching bases are pulled
from the committed public model, tests/golden/models/hg19_muc1.db (VID 25561), at build
time: the last 30bp of its left flank plus its first two repeat segments (30+60+60 =
150bp -- also the read length this fixture makes `select_illumina_reads` derive, since
`sorted(samfile.head(5) lengths)[3]` == 150 with 3 shorter padding records ahead of it).
That 150bp string is what `model.viterbi` scores well (measured logp -21.109968289314292
against read_length=150). This script stores its REVERSE COMPLEMENT as the BAM record's
SEQ (FLAG 16, MAPQ 60), so the forward decode -- the one `pending.sequence` receives --
scores badly instead (measured logp -311.2674450314392). Only a decoder that runs both
orientations and keeps the winner (`_decode_one`) selects this read at all; a forward-only
decoder rejects it as low likelihood, and there is no other read in this fixture to select.

No byte of this file, or of the BAM it writes, is derived from any simulated or screening
cohort. Every sequence -- target and padding alike -- is sliced from the public golden
model's own left_flanking_region / right_flanking_region / repeat_segments; this repo's
data rule (AGENTS.md) allows nothing else in.

A committed artefact needs a committed producer -- see advntr_harness/tier3.py:157-170
for why: without one, the only record of how the fixture was made is a shell snippet in
someone's history.

Run from the repo root, in the envadvntr environment:
    python scripts/make_revcomp_fixture.py
"""
import os

import pysam
from Bio.Seq import Seq

from advntr_harness.capture import build_finder

VID = 25561
DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  'tests', 'golden', 'models', 'hg19_muc1.db')
OUT_BAM = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'tests', 'data', 'synthetic_revcomp.bam')

#: hg19 chr1 length. advntr/sam_utils.py:31-38's get_reference_genome_of_alignment_file
#: classifies a BAM as 'HG19' whenever ANY reference name starts with 'chr'; this header
#: must therefore use 'chr1', not '1', or select_illumina_reads strips the prefix off the
#: model's chromosome and samfile.fetch('1', ...) raises against a 'chr1'-named contig.
CHR1_LENGTH = 249250621

#: What this fixture is built to make select_illumina_reads derive as read_length: it is
#: `sorted(samfile.head(5) lengths)[3]` (advntr/vntr_finder.py:1071-1073, mirrored at
#: advntr_harness/extract.py:34-45), so 3 padding records well under it plus this one
#: target puts the target's own length at that index.
TARGET_LENGTH = 150
#: MIN_READ_LENGTH at TARGET_LENGTH is int(150 * 0.9) == 135; padding must stay well
#: under that so it is rejected as TOO_SHORT before decoding, not merely out of span.
PADDING_LENGTH = 50

TARGET_NAME = 'synthetic_target_revcomp'
PADDING_NAMES = ('synthetic_padding_1', 'synthetic_padding_2', 'synthetic_padding_3')


def reference_vntr():
    """Load VID 25561 from the committed public model, the way build_finder does."""
    _finder, reference = build_finder(DB)
    return reference


def build_records(ref):
    """Return the target and padding records as plain dicts, sorted by position.

    `target`'s stored SEQ is the reverse complement of the only 150bp string that scores
    well against the model -- see the module docstring for the measured logp values that
    prove it. All four records sit inside (vntr_start - TARGET_LENGTH, vntr_end), so all
    four reach phase 1's filters; the padding records are excluded by MIN_READ_LENGTH,
    not by falling outside the fetched span -- tests/test_orientation.py asserts that
    directly so a future edit that stops exercising TOO_SHORT fails loudly.
    """
    vntr_start = ref.start_point
    good_match = ref.left_flanking_region[-30:] + ref.repeat_segments[0] + ref.repeat_segments[1]
    if len(good_match) != TARGET_LENGTH:
        raise AssertionError('good_match is %d bp, expected %d' % (len(good_match), TARGET_LENGTH))
    stored_seq = str(Seq(good_match).reverse_complement())

    target = {
        'name': TARGET_NAME,
        'seq': stored_seq,
        'reference_start': vntr_start - 30,
        'flag': 16,  # reverse strand -- the orientation this fixture exists to guard
        'mapq': 60,  # > settings.MAPQ_CUTOFF (0), so is_low_quality turns on quality alone
    }

    padding_sources = (
        ref.repeat_segments[2][:PADDING_LENGTH],
        ref.right_flanking_region[:PADDING_LENGTH],
        ref.left_flanking_region[:PADDING_LENGTH],
    )
    padding_offsets = (10, 60, 110)  # bp past vntr_start -- keeps all three inside the span
    padding = []
    for name, seq, offset in zip(PADDING_NAMES, padding_sources, padding_offsets):
        padding.append({
            'name': name,
            'seq': seq,
            'reference_start': vntr_start + offset,
            'flag': 0,
            'mapq': 60,
        })

    return sorted([target] + padding, key=lambda record: record['reference_start'])


def write_bam(path, ref):
    """Write a coordinate-sorted, indexed BAM. pysam.fetch() requires both."""
    records = build_records(ref)
    header = {'HD': {'VN': '1.6', 'SO': 'coordinate'},
             'SQ': [{'SN': ref.chromosome, 'LN': CHR1_LENGTH}]}

    with pysam.AlignmentFile(path, 'wb', header=header) as outfile:
        for record in records:
            segment = pysam.AlignedSegment(outfile.header)
            segment.query_name = record['name']
            segment.query_sequence = record['seq']
            segment.flag = record['flag']
            segment.reference_id = 0
            segment.reference_start = record['reference_start']
            segment.mapping_quality = record['mapq']
            segment.cigarstring = '%dM' % len(record['seq'])
            # Every base high quality ('I' = Phred 40), so is_low_quality_read
            # (advntr/utils.py:20-38) rejects nothing here on quality grounds either --
            # the only thing this fixture tests is the orientation decode.
            segment.query_qualities = pysam.qualitystring_to_array('I' * len(record['seq']))
            outfile.write(segment)

    pysam.index(path)


def main():
    ref = reference_vntr()
    out_dir = os.path.dirname(OUT_BAM)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    write_bam(OUT_BAM, ref)
    print('wrote %s (+ .bai)' % OUT_BAM)


if __name__ == '__main__':
    main()
