"""Tests for adapter read-through detection and filtering (Issue #4)."""
import os
import shutil
import tempfile
import unittest

import pysam

from advntr import settings
from advntr.adapter_filter import (
    contains_adapter_kmer,
    count_genuine_matches,
    genuine_match_ratio,
    is_adapter_readthrough,
    is_adapter_driven_mutation
)
from advntr.models import load_unique_vntrs_data
from advntr.vntr_finder import VNTRFinder


class DummyState(object):
    def __init__(self, name):
        self.name = name


class TestAdapterFilter(unittest.TestCase):

    def test_contains_adapter_kmer(self):
        # TruSeq forward
        seq1 = 'GGCCGAGGTGACACCGTGGGAGATCGGAAGAGCACACGTCTGAACTCC'
        self.assertTrue(contains_adapter_kmer(seq1))

        # TruSeq reverse complement
        seq2 = 'GCTCTTCCGATCTGACACCGTGGG'
        self.assertTrue(contains_adapter_kmer(seq2))

        # Core 9-mer
        seq3 = 'ACGTACGTACAGATCGGAATTT'
        self.assertTrue(contains_adapter_kmer(seq3))

        # Clean VNTR repeat unit
        seq_clean = 'GGCCGAGGTGACACCGTGGGCTGGGGGGCGGTGGAGCCCGGGGCCGGCCTGGTGTCCGG'
        self.assertFalse(contains_adapter_kmer(seq_clean))

    def test_count_genuine_matches(self):
        # 20 M states, 40 I states, 5 D states
        vpath = [(-1, DummyState('start'))]
        for _ in range(20):
            vpath.append((0, DummyState('M1_1')))
        for _ in range(40):
            vpath.append((0, DummyState('I1_1')))
        for _ in range(5):
            vpath.append((0, DummyState('D1_1')))
        vpath.append((-1, DummyState('end')))

        self.assertEqual(count_genuine_matches(vpath), 20)
        self.assertAlmostEqual(genuine_match_ratio(vpath, 60), 20.0 / 60.0)

    def test_is_adapter_readthrough(self):
        # Read with adapter sequence
        read_with_adapter = 'GGCCGAGGTGACACCGTGGGAGATCGGAAGAGCACACG'
        self.assertTrue(is_adapter_readthrough(read_with_adapter))

        # Read without adapter but low match ratio (e.g. 20 matches out of 150)
        vpath_low = [(-1, DummyState('start'))]
        for _ in range(20):
            vpath_low.append((0, DummyState('M1_1')))
        for _ in range(130):
            vpath_low.append((0, DummyState('I1_1')))
        vpath_low.append((-1, DummyState('end')))

        clean_seq = 'A' * 150
        self.assertTrue(is_adapter_readthrough(clean_seq, vpath=vpath_low, min_match_ratio=0.60))

        # Clean read with high match ratio (140 matches out of 150)
        vpath_high = [(-1, DummyState('start'))]
        for _ in range(140):
            vpath_high.append((0, DummyState('M1_1')))
        for _ in range(10):
            vpath_high.append((0, DummyState('I1_1')))
        vpath_high.append((-1, DummyState('end')))

        self.assertFalse(is_adapter_readthrough(clean_seq, vpath=vpath_high, min_match_ratio=0.60))

        # When min_match_ratio is None, default 0.60 applies and catches low match ratio
        self.assertTrue(is_adapter_readthrough(clean_seq, vpath=vpath_low, min_match_ratio=None))

    def test_adapter_orientations(self):
        # Forward core 8-mer
        seq_fwd = 'ACGTACGT' + 'AGATCGGA' + 'TGCAT'
        self.assertTrue(contains_adapter_kmer(seq_fwd))

        # Reverse complement core 8-mer
        seq_rev = 'ACGTACGT' + 'TCCGATCT' + 'TGCAT'
        self.assertTrue(contains_adapter_kmer(seq_rev))

    def test_high_match_read_preservation(self):
        # A read with 150 matches out of 151 bases should NOT be rejected even if
        # a short sub-sequence resembles an 8-mer
        vpath_intact = [(-1, DummyState('start'))]
        for _ in range(150):
            vpath_intact.append((0, DummyState('M1_1')))
        vpath_intact.append((-1, DummyState('end')))

        seq = 'ACGTACGT' + 'AGATCGGA' + 'C' * 135
        # With vpath indicating high match ratio (150/151 >= 0.75), not rejected without long adapter
        self.assertFalse(is_adapter_readthrough(seq, vpath=vpath_intact))

    def test_is_adapter_driven_mutation(self):
        # True adapter insertion
        obs_unit = 'GGCCGAGGTGACACCGTGGGAGATCGGAAGAGCACACG'
        self.assertTrue(is_adapter_driven_mutation('I22_7_A_LEN7', inserted_seq='AGATCGGAAGAGC', observed_unit=obs_unit))
        self.assertTrue(is_adapter_driven_mutation('D27_7&I27_7_A_LEN3', observed_unit=obs_unit))

        # Genuine variant
        genuine_obs = 'GGCCGAGGTGAACACCGTGGGCTTGGGGGGCGGTGGAGCCCGGGGCCGGCCTGGTGTCCGG'
        self.assertFalse(is_adapter_driven_mutation('I10_6_A_LEN1', inserted_seq='A', observed_unit=genuine_obs))

    def test_select_illumina_reads_adapter_filter_branch(self):
        """Integration test verifying production filtering branch in select_illumina_reads."""
        ref = [v for v in load_unique_vntrs_data('tests/golden/models/hg19_muc1.db') if v.id == 25561][0]
        ru = sorted(set(ref.get_repeat_segments()))[1]

        seq_genuine = (ru * 3)[:151]
        seq_adapter = ru[:20] + ('AGATCGGAAGAGCACACGTCTGAACTCCAGTCAC' * 4)[:131]

        orig_cores = settings.CORES
        orig_filter = settings.FILTER_ADAPTER_READTHROUGH
        tmp = tempfile.mkdtemp(prefix='advntr-test-')
        try:
            bam_path = os.path.join(tmp, 'test.bam')
            reads_data = [('genuine_%d' % i, seq_genuine, ref.start_point + 10 + i) for i in range(4)]
            reads_data += [('adapter_%d' % i, seq_adapter, ref.start_point + 20 + i) for i in range(2)]
            reads_data.sort(key=lambda x: x[2])
            with pysam.AlignmentFile(bam_path, 'wb', header={'HD': {'VN': '1.0', 'SO': 'coordinate'},
                                                             'SQ': [{'SN': 'chr1', 'LN': 249250621}]}) as out:
                for name, s, pos in reads_data:
                    r = pysam.AlignedSegment()
                    r.query_name = name
                    r.query_sequence = s
                    r.flag = 0
                    r.reference_id = 0
                    r.reference_start = pos
                    r.mapping_quality = 60
                    r.cigar = [(0, len(s))]
                    r.query_qualities = pysam.qualitystring_to_array('I' * len(s))
                    out.write(r)
            pysam.index(bam_path)

            settings.CORES = 1
            finder = VNTRFinder(ref, is_frameshift_mode=True)

            settings.FILTER_ADAPTER_READTHROUGH = False
            sel_unfiltered = finder.select_illumina_reads(bam_path, [])
            self.assertEqual(len(sel_unfiltered), 6)

            settings.FILTER_ADAPTER_READTHROUGH = True
            sel_filtered = finder.select_illumina_reads(bam_path, [])
            self.assertEqual(len(sel_filtered), 4)
            for r in sel_filtered:
                self.assertTrue(r.query_name.startswith('genuine_'))
        finally:
            settings.CORES = orig_cores
            settings.FILTER_ADAPTER_READTHROUGH = orig_filter
            shutil.rmtree(tmp)

    def test_partial_unit_adapter_insertion_rejection(self):
        """Test candidate-level rejection of adapter insertion in partial_end."""
        from advntr.reference_vntr import ReferenceVNTR
        from tests.test_frameshift_context import _vpath
        from advntr.vntr_finder import SelectedRead
        unit = 'ACGT' * 15
        ref = ReferenceVNTR(1, unit, 100, 'chr1', None, None)
        ref.init_from_xml([unit, unit], 'TTTTTTTT', 'GGGGGGGG')
        finder = VNTRFinder(ref, is_frameshift_mode=True)
        class HMM(object):
            read_length_used_to_build_model = 151
        finder.hmm = HMM()

        states = []
        for _ in range(2):
            states += ['unit_start_1'] + ['M%d_1' % p for p in range(1, 61)] + ['unit_end_1']
        states += ['unit_start_1'] + ['M%d_1' % p for p in range(1, 21)] + ['I20_1'] * 8
        read = SelectedRead(unit * 2 + unit[:20] + 'AGATCGGA', -1.0, _vpath(states), query_name='adapter')

        orig_ref_aln = settings.USE_REF_ALIGNMENT
        orig_fully_covered = settings.USE_ONLY_FULLY_COVERED_RU
        orig_min_support = settings.MIN_SUPPORTING_READ_COUNT
        orig_filter = settings.FILTER_ADAPTER_READTHROUGH
        try:
            settings.USE_REF_ALIGNMENT = False
            settings.USE_ONLY_FULLY_COVERED_RU = False
            settings.MIN_SUPPORTING_READ_COUNT = 1

            settings.FILTER_ADAPTER_READTHROUGH = False
            unfiltered_calls = finder.find_frameshift_from_selected_reads([read])
            self.assertEqual(len(unfiltered_calls), 1)
            self.assertEqual(unfiltered_calls[0][0], 'I20_1_A_LEN8')

            settings.FILTER_ADAPTER_READTHROUGH = True
            filtered_calls = finder.find_frameshift_from_selected_reads([read])
            self.assertIsNone(filtered_calls)
        finally:
            settings.USE_REF_ALIGNMENT = orig_ref_aln
            settings.USE_ONLY_FULLY_COVERED_RU = orig_fully_covered
            settings.MIN_SUPPORTING_READ_COUNT = orig_min_support
            settings.FILTER_ADAPTER_READTHROUGH = orig_filter

    def test_partial_unit_compound_adapter_event_rejection(self):
        """Test that compound adapter evidence (e.g. deletion + adapter insertion) is rejected as a whole."""
        from advntr.reference_vntr import ReferenceVNTR
        from tests.test_frameshift_context import _vpath
        from advntr.vntr_finder import SelectedRead
        unit = 'ACGT' * 15
        ref = ReferenceVNTR(1, unit, 100, 'chr1', None, None)
        ref.init_from_xml([unit, unit], 'TTTTTTTT', 'GGGGGGGG')
        finder = VNTRFinder(ref, is_frameshift_mode=True)
        class HMM(object):
            read_length_used_to_build_model = 151
        finder.hmm = HMM()

        # Partial end has: M1..M20, D21, D22, I22 (8bp adapter AGATCGGA)
        states = []
        for _ in range(2):
            states += ['unit_start_1'] + ['M%d_1' % p for p in range(1, 61)] + ['unit_end_1']
        states += ['unit_start_1'] + ['M%d_1' % p for p in range(1, 21)] + ['D21_1', 'D22_1'] + ['I22_1'] * 8
        read_seq = unit * 2 + unit[:20] + 'AGATCGGA'
        reads = [
            SelectedRead(read_seq, -1.0, _vpath(states), query_name='adapter_%d' % i)
            for i in range(3)
        ]

        orig_ref_aln = settings.USE_REF_ALIGNMENT
        orig_fully_covered = settings.USE_ONLY_FULLY_COVERED_RU
        orig_min_support = settings.MIN_SUPPORTING_READ_COUNT
        orig_filter = settings.FILTER_ADAPTER_READTHROUGH
        try:
            settings.USE_REF_ALIGNMENT = False
            settings.USE_ONLY_FULLY_COVERED_RU = False
            settings.MIN_SUPPORTING_READ_COUNT = 3

            # With filter OFF: compound event is processed
            settings.FILTER_ADAPTER_READTHROUGH = False
            unfiltered_calls = finder.find_frameshift_from_selected_reads(reads)
            self.assertIsNotNone(unfiltered_calls)
            self.assertTrue(any('D21_1' in c[0] and 'I22_1' in c[0] for c in unfiltered_calls))

            # With filter ON: the compound event must NOT leave behind an artifactual D21_1&D22_1 deletion!
            settings.FILTER_ADAPTER_READTHROUGH = True
            filtered_calls = finder.find_frameshift_from_selected_reads(reads)
            self.assertIsNone(filtered_calls)
        finally:
            settings.USE_REF_ALIGNMENT = orig_ref_aln
            settings.USE_ONLY_FULLY_COVERED_RU = orig_fully_covered
            settings.MIN_SUPPORTING_READ_COUNT = orig_min_support
            settings.FILTER_ADAPTER_READTHROUGH = orig_filter

    def test_adapter_occurrence_excluded_from_opportunity_counts(self):
        """Test that adapter-rejected partial occurrences do not inflate opportunity denominators N."""
        from advntr.reference_vntr import ReferenceVNTR
        from tests.test_frameshift_context import _vpath
        from advntr.vntr_finder import SelectedRead
        unit = 'ACGT' * 15
        ref = ReferenceVNTR(1, unit, 100, 'chr1', None, None)
        ref.init_from_xml([unit, unit], 'TTTTTTTT', 'GGGGGGGG')
        finder = VNTRFinder(ref, is_frameshift_mode=True)
        class HMM(object):
            read_length_used_to_build_model = 151
        finder.hmm = HMM()

        full = ['unit_start_1'] + ['M%d_1' % p for p in range(1, 61)] + ['unit_end_1']
        adapter_states = full * 2 + ['unit_start_1'] + ['M%d_1' % p for p in range(1, 21)] + ['I20_1'] * 8
        adapter = SelectedRead(unit * 2 + unit[:20] + 'AGATCGGA', -1.0, _vpath(adapter_states), query_name='adapter')
        variant_states = ['unit_start_1'] + ['M%d_1' % p for p in range(1, 21)] + ['I20_1'] + ['M%d_1' % p for p in range(21, 61)] + ['unit_end_1'] + full
        variant = SelectedRead(unit[:20] + 'A' + unit[20:] + unit, -1.0, _vpath(variant_states), query_name='variant')

        orig_ref_aln = settings.USE_REF_ALIGNMENT
        orig_fully_covered = settings.USE_ONLY_FULLY_COVERED_RU
        orig_min_support = settings.MIN_SUPPORTING_READ_COUNT
        orig_filter = settings.FILTER_ADAPTER_READTHROUGH
        try:
            settings.USE_REF_ALIGNMENT = False
            settings.USE_ONLY_FULLY_COVERED_RU = False
            settings.MIN_SUPPORTING_READ_COUNT = 1
            settings.FILTER_ADAPTER_READTHROUGH = True

            finder.find_frameshift_from_selected_reads([variant, adapter])
            row = finder.last_frameshift_opportunities['I20_1_A_LEN1']
            # variant read has 2 complete units, adapter read has 2 complete units.
            # adapter's partial_end occurrence must be excluded, so N = 4, NOT 5.
            self.assertEqual(row['support'], 1)
            self.assertEqual(row['opportunities'], 4)
            span_counts = dict(row['opportunity_spans'])
            self.assertEqual(sum(span_counts.values()), 4)
        finally:
            settings.USE_REF_ALIGNMENT = orig_ref_aln
            settings.USE_ONLY_FULLY_COVERED_RU = orig_fully_covered
            settings.MIN_SUPPORTING_READ_COUNT = orig_min_support
            settings.FILTER_ADAPTER_READTHROUGH = orig_filter

    def test_preserve_high_match_reads_with_adapter_like_reference(self):
        """Test that high-match complete units are not rejected even if reference contains adapter-like k-mer."""
        from advntr.reference_vntr import ReferenceVNTR
        from tests.test_frameshift_context import _vpath
        from advntr.vntr_finder import SelectedRead
        unit = 'ACGT' * 5 + 'AGATCGGA' + 'ACGT' * 8
        ref = ReferenceVNTR(1, unit, 100, 'chr1', None, None)
        ref.init_from_xml([unit, unit], 'TTTTTTTT', 'GGGGGGGG')
        finder = VNTRFinder(ref, is_frameshift_mode=True)
        class HMM(object):
            read_length_used_to_build_model = 151
        finder.hmm = HMM()

        full = ['unit_start_1'] + ['M%d_1' % p for p in range(1, 61)] + ['unit_end_1']
        states = ['unit_start_1'] + ['M%d_1' % p for p in range(1, 11)] + ['I10_1'] + ['M%d_1' % p for p in range(11, 61)] + ['unit_end_1'] + full
        read = SelectedRead(unit[:10] + 'C' + unit[10:] + unit, -1.0, _vpath(states), query_name='genuine')

        orig_ref_aln = settings.USE_REF_ALIGNMENT
        orig_fully_covered = settings.USE_ONLY_FULLY_COVERED_RU
        orig_min_support = settings.MIN_SUPPORTING_READ_COUNT
        orig_filter = settings.FILTER_ADAPTER_READTHROUGH
        try:
            settings.USE_REF_ALIGNMENT = False
            settings.USE_ONLY_FULLY_COVERED_RU = False
            settings.MIN_SUPPORTING_READ_COUNT = 1

            for flag in [False, True]:
                settings.FILTER_ADAPTER_READTHROUGH = flag
                calls = finder.find_frameshift_from_selected_reads([read])
                self.assertIsNotNone(calls)
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0][0], 'I10_1_C_LEN1')
        finally:
            settings.USE_REF_ALIGNMENT = orig_ref_aln
            settings.USE_ONLY_FULLY_COVERED_RU = orig_fully_covered
            settings.MIN_SUPPORTING_READ_COUNT = orig_min_support
            settings.FILTER_ADAPTER_READTHROUGH = orig_filter

    def test_exclude_rejected_occurrences_from_insertion_base_attribution(self):
        """Test that adapter-rejected occurrences do not contaminate the emitted base of retained insertions."""
        from advntr.reference_vntr import ReferenceVNTR
        from tests.test_frameshift_context import _vpath
        from advntr.vntr_finder import SelectedRead
        unit = 'ACGT' * 15
        ref = ReferenceVNTR(1, unit, 100, 'chr1', None, None)
        ref.init_from_xml([unit, unit], 'TTTTTTTT', 'GGGGGGGG')
        finder = VNTRFinder(ref, is_frameshift_mode=True)
        class HMM(object):
            read_length_used_to_build_model = 151
        finder.hmm = HMM()

        part_start = ['M%d_1' % p for p in range(1, 21)] + ['I20_1'] * 8 + ['M%d_1' % p for p in range(21, 61)] + ['unit_end_1']
        unit_c = ['unit_start_1'] + ['M%d_1' % p for p in range(1, 21)] + ['I20_1'] + ['M%d_1' % p for p in range(21, 61)] + ['unit_end_1']
        states = part_start + unit_c
        seq = unit[:20] + 'AGATCGGA' + unit[20:] + unit[:20] + 'C' + unit[20:]
        read = SelectedRead(seq, -1.0, _vpath(states), query_name='adapter_and_var')

        orig_ref_aln = settings.USE_REF_ALIGNMENT
        orig_fully_covered = settings.USE_ONLY_FULLY_COVERED_RU
        orig_min_support = settings.MIN_SUPPORTING_READ_COUNT
        orig_filter = settings.FILTER_ADAPTER_READTHROUGH
        try:
            settings.USE_REF_ALIGNMENT = False
            settings.USE_ONLY_FULLY_COVERED_RU = False
            settings.MIN_SUPPORTING_READ_COUNT = 1
            settings.FILTER_ADAPTER_READTHROUGH = True

            calls = finder.find_frameshift_from_selected_reads([read])
            self.assertIsNotNone(calls)
            self.assertEqual(len(calls), 1)
            # Must be I20_1_C_LEN1, NOT I20_1_A_LEN1!
            self.assertEqual(calls[0][0], 'I20_1_C_LEN1')
        finally:
            settings.USE_REF_ALIGNMENT = orig_ref_aln
            settings.USE_ONLY_FULLY_COVERED_RU = orig_fully_covered
            settings.MIN_SUPPORTING_READ_COUNT = orig_min_support
            settings.FILTER_ADAPTER_READTHROUGH = orig_filter

    def test_split_alignment_adapter_motif_rejection(self):
        """Test rejection of split adapter motif where 1st base occupies M and remaining occupy I."""
        from advntr.reference_vntr import ReferenceVNTR
        from tests.test_frameshift_context import _vpath
        from advntr.vntr_finder import SelectedRead
        unit = 'ACGT' * 15
        ref = ReferenceVNTR(1, unit, 100, 'chr1', None, None)
        ref.init_from_xml([unit, unit], 'TTTTTTTT', 'GGGGGGGG')
        finder = VNTRFinder(ref, is_frameshift_mode=True)
        class HMM(object):
            read_length_used_to_build_model = 151
        finder.hmm = HMM()

        full = ['unit_start_1'] + ['M%d_1' % p for p in range(1, 61)] + ['unit_end_1']
        split_part = ['unit_start_1'] + ['M%d_1' % p for p in range(1, 22)] + ['I21_1'] * 7
        states = full * 2 + split_part
        seq = unit * 2 + unit[:21] + 'GATCGGA'
        read = SelectedRead(seq, -1.0, _vpath(states), query_name='split_adapter')

        orig_ref_aln = settings.USE_REF_ALIGNMENT
        orig_fully_covered = settings.USE_ONLY_FULLY_COVERED_RU
        orig_min_support = settings.MIN_SUPPORTING_READ_COUNT
        orig_filter = settings.FILTER_ADAPTER_READTHROUGH
        try:
            settings.USE_REF_ALIGNMENT = False
            settings.USE_ONLY_FULLY_COVERED_RU = False
            settings.MIN_SUPPORTING_READ_COUNT = 1

            settings.FILTER_ADAPTER_READTHROUGH = False
            unfiltered_calls = finder.find_frameshift_from_selected_reads([read])
            self.assertIsNotNone(unfiltered_calls)
            self.assertEqual(unfiltered_calls[0][0], 'I21_1_G_LEN7')

            settings.FILTER_ADAPTER_READTHROUGH = True
            filtered_calls = finder.find_frameshift_from_selected_reads([read])
            self.assertIsNone(filtered_calls)
        finally:
            settings.USE_REF_ALIGNMENT = orig_ref_aln
            settings.USE_ONLY_FULLY_COVERED_RU = orig_fully_covered
            settings.MIN_SUPPORTING_READ_COUNT = orig_min_support
            settings.FILTER_ADAPTER_READTHROUGH = orig_filter


if __name__ == '__main__':
    unittest.main()
