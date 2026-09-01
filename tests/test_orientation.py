"""Guards the reverse decode in advntr/read_selection.py:_decode_one.

The public corpus provably cannot catch its removal: every reverse-wins read in the 8
example BAMs is rejected before selection (TOO_SHORT or OUT_OF_SPAN), so the Tier 3
digest is identical whether the reverse `model.viterbi` call runs or not -- demonstrated
by injection, and the defect passed a full green gate. See AGENTS.md's
"reverse_complement_wins" stratum note (77 of 29,998 corpus reads; zero in example_66bf's
12,608). tests/data/synthetic_revcomp.bam (built by scripts/make_revcomp_fixture.py) is
the one fixture that forces the branch.

Measured against tests/golden/models/hg19_muc1.db (VID 25561) at read_length=150:
    forward decode of the stored SEQ:                      logp = -311.2674450314392
    reverse decode (decode of the SEQ's reverse complement): logp = -21.109968289314292
Only a decoder that runs both orientations and keeps the winner selects this read at all
-- a forward-only decoder rejects it as low likelihood, and it is the only decodable read
in the fixture.

Runs the real `select_illumina_reads` (via advntr_harness.capture.build_finder, the way
tests/test_tier3_occurrence.py does), not a reimplementation.
"""
import os
import unittest

from Bio.Seq import Seq

from advntr import settings
from advntr_harness.capture import build_finder
from advntr_harness.extract import derive_read_length

GOLDEN = os.path.join(os.path.dirname(__file__), 'golden')
DB = os.path.join(GOLDEN, 'models', 'hg19_muc1.db')
BAM = os.path.join(os.path.dirname(__file__), 'data', 'synthetic_revcomp.bam')

TARGET_NAME = 'synthetic_target_revcomp'

has_fixture = unittest.skipUnless(
    os.path.isfile(BAM),
    'tests/data/synthetic_revcomp.bam missing -- run scripts/make_revcomp_fixture.py')


class TestFixtureFileExists(unittest.TestCase):
    """Not `@has_fixture` -- deliberately unconditional (Task 8 controller ruling).

    Every class below it is `skipUnless(os.path.isfile(BAM), ...)`, which mirrors the
    Tier 1/Tier 3 idiom for an EXTERNAL file the corpus may or may not supply -- a
    reasonable thing to skip on. This fixture is different: it is COMMITTED
    (tests/data/synthetic_revcomp.bam, added by 6f51ba8). A skip for a committed file
    means only one thing -- someone deleted it -- and `skipUnless` alone turns that into
    a silent green run instead of a loud failure, which is exactly the "a green skip is
    not evidence" trap this plan's Task 8 exists to close. This test has no skip
    decorator, so its absence fails here, not just quietly downgrades every other class
    in this module to skipped.
    """

    def test_the_committed_fixture_is_present(self):
        self.assertTrue(
            os.path.isfile(BAM),
            '%s is missing. This is a COMMITTED fixture (6f51ba8), not an external '
            'corpus file -- its absence means it was deleted, not merely unavailable. '
            'Regenerate with `python scripts/make_revcomp_fixture.py` and `git add` it '
            'back; every other class in this module silently skips without it.' % BAM)


def _read_records(path):
    """(query_name -> uppercase sequence) for every record in the fixture BAM."""
    import pysam
    samfile = pysam.AlignmentFile(path, 'rb')
    try:
        records = dict((read.query_name, str(read.seq).upper())
                       for read in samfile.fetch())
    finally:
        samfile.close()
    return records


@has_fixture
class TestReverseDecodeIsGuarded(unittest.TestCase):
    """The real `select_illumina_reads`, exactly as production runs it."""

    @classmethod
    def setUpClass(cls):
        finder, _reference = build_finder(DB)
        cls.selected = finder.select_illumina_reads(BAM, [])

    def test_exactly_one_read_is_selected(self):
        """The 3 padding records are shorter than MIN_READ_LENGTH and are rejected as
        TOO_SHORT before decoding -- see TestPaddingIsRejectedForTheIntendedReason,
        which pins that down directly so this count alone never has to carry the whole
        claim."""
        self.assertEqual(len(self.selected), 1)

    def test_the_selected_read_is_the_target(self):
        self.assertEqual(self.selected[0].query_name, TARGET_NAME)

    def test_the_selected_sequence_is_the_reverse_complement_of_the_stored_seq(self):
        """The BAM stores SEQ reverse-complemented (FLAG 16) relative to the sequence
        that actually scores well. select_illumina_reads must hand back the reverse
        decode's sequence -- the complement of what is on disk -- not the raw SEQ."""
        stored_seq = _read_records(BAM)[TARGET_NAME]
        expected = str(Seq(stored_seq).reverse_complement()).upper()
        self.assertEqual(self.selected[0].sequence, expected)

    def test_the_selected_logp_beats_the_length_floor(self):
        """settings.MAPQ_CUTOFF is 0, so MAPQ 60 makes is_low_quality False and
        `recruit_read` is never consulted -- selection turns only on `logp != -inf`
        (advntr/vntr_finder.py:1143-1148). A forward-only decoder could still return a
        poor-but-finite logp (measured -311.27 for this fixture's forward orientation)
        and get the read selected on that alone. Comparing against -len(sequence) is
        what catches that: the real winning decode measures -21.11, comfortably above
        -150, while the forward-only value does not clear it."""
        selected = self.selected[0]
        self.assertGreater(selected.logp, -len(selected.sequence))


@has_fixture
class TestReverseDecodeIsGuardedWithPruningOn(unittest.TestCase):
    """Task 8's Tier B fixture-stratum demonstration: this is the ONE fixture in the
    repo where the reverse decode actually wins (AGENTS.md's "reverse_complement_wins"
    note; the corpus itself never exercises it -- see the module docstring), so it is
    also the one case that forces `--prune-reverse`'s safety valve to fire. `_decode_one`
    prunes the reverse decode to threshold=max(dp_score_threshold, fwd_logp) = fwd_logp
    (-311.27, comfortably above dp_score_threshold), which is BELOW the true reverse
    score (-21.11) -- so the pruned result still clears fwd_logp, the valve re-runs the
    reverse decode unpruned, and the outcome must be byte-identical to
    TestReverseDecodeIsGuarded above. Every assertion here is a literal copy of that
    class's, run with the flag on instead of off.
    """

    @classmethod
    def setUpClass(cls):
        cls._original_flag = settings.PRUNE_REVERSE_DECODE
        settings.PRUNE_REVERSE_DECODE = True
        finder, _reference = build_finder(DB)
        cls.selected = finder.select_illumina_reads(BAM, [])

    @classmethod
    def tearDownClass(cls):
        settings.PRUNE_REVERSE_DECODE = cls._original_flag

    def test_exactly_one_read_is_selected(self):
        self.assertEqual(len(self.selected), 1)

    def test_the_selected_read_is_the_target(self):
        self.assertEqual(self.selected[0].query_name, TARGET_NAME)

    def test_the_selected_sequence_is_the_reverse_complement_of_the_stored_seq(self):
        stored_seq = _read_records(BAM)[TARGET_NAME]
        expected = str(Seq(stored_seq).reverse_complement()).upper()
        self.assertEqual(self.selected[0].sequence, expected)

    def test_the_selected_logp_beats_the_length_floor(self):
        selected = self.selected[0]
        self.assertGreater(selected.logp, -len(selected.sequence))

    def test_the_selected_logp_matches_the_unpruned_decode_exactly(self):
        """Not just "close" -- the safety valve exists to make pruning invisible to
        phase 3, so this must be the identical IEEE-754 value TestReverseDecodeIsGuarded
        measures with the flag off, not merely a read with the same query_name."""
        self.assertEqual(self.selected[0].logp, -21.109968289314292)


class TestPaddingIsRejectedForTheIntendedReason(unittest.TestCase):
    """Guards the guard: if a future edit to the fixture let padding clear
    MIN_READ_LENGTH, `test_exactly_one_read_is_selected` above would still fail, but for
    an unrelated reason (padding got selected too) that would be easy to misdiagnose as
    this test itself being broken. This pins the actual mechanism down directly, using
    the same derive_read_length the harness proves matches production
    (advntr_harness/extract.py:34-45, advntr/vntr_finder.py:1071-1076)."""

    @has_fixture
    def test_padding_reads_are_too_short_to_survive_min_read_length(self):
        import pysam
        samfile = pysam.AlignmentFile(BAM, 'rb')
        try:
            read_length = derive_read_length(samfile)
        finally:
            samfile.close()
        min_read_length = (settings.MIN_READ_LENGTH if settings.MIN_READ_LENGTH is not None
                           else int(read_length * 0.9))

        lengths = dict((name, len(seq)) for name, seq in _read_records(BAM).items())
        target_length = lengths.pop(TARGET_NAME)
        self.assertEqual(target_length, read_length,
                         'the target read no longer determines the derived read_length; '
                         'the MIN_READ_LENGTH check below would no longer mean what it says')
        self.assertTrue(lengths, 'no padding records found in the fixture')
        for name, length in lengths.items():
            self.assertLess(
                length, min_read_length,
                '%s is %dbp, which clears MIN_READ_LENGTH (%dbp) and would reach the '
                'decoder -- defeating the point of padding' % (name, length, min_read_length))


if __name__ == '__main__':
    unittest.main()
