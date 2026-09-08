"""Tests for adapter read-through detection and filtering (Issue #4)."""
import unittest

from advntr.adapter_filter import (
    contains_adapter_kmer,
    count_genuine_matches,
    genuine_match_ratio,
    is_adapter_readthrough,
    is_adapter_driven_mutation
)


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

    def test_is_adapter_driven_mutation(self):
        # True adapter insertion
        obs_unit = 'GGCCGAGGTGACACCGTGGGAGATCGGAAGAGCACACG'
        self.assertTrue(is_adapter_driven_mutation('I22_7_A_LEN7', inserted_seq='AGATCGGAAGAGC', observed_unit=obs_unit))
        self.assertTrue(is_adapter_driven_mutation('D27_7&I27_7_A_LEN3', observed_unit=obs_unit))

        # Genuine variant
        genuine_obs = 'GGCCGAGGTGAACACCGTGGGCTTGGGGGGCGGTGGAGCCCGGGGCCGGCCTGGTGTCCGG'
        self.assertFalse(is_adapter_driven_mutation('I10_6_A_LEN1', inserted_seq='A', observed_unit=genuine_obs))


if __name__ == '__main__':
    unittest.main()
