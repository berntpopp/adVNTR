import unittest

from advntr.mutation_keys import left_normalise_insertion


RU2_SEQUENCE = (
    'GGCCGAGGTGACACCATGGGCTGGGGGGGCGGTGGAGCCCGGGGCCGGCCTGGTGTCCGG'
)


class TestLeftNormaliseInsertion(unittest.TestCase):
    def test_every_placement_in_a_homopolymer_maps_to_the_leftmost(self):
        """The G run is ambiguous; the call key must not inherit DP visit order."""
        for offset in range(22, 30):
            self.assertEqual(left_normalise_insertion(RU2_SEQUENCE, offset, 'G'), (22, 'G'))

    def test_a_non_homopolymer_insertion_is_unmoved(self):
        self.assertEqual(left_normalise_insertion(RU2_SEQUENCE, 40, 'T'), (40, 'T'))

    def test_offset_is_zero_based_insertion_slot(self):
        self.assertEqual(left_normalise_insertion('ACGT', 2, 'C'), (1, 'C'))

    def test_multi_base_insertions_rotate_as_full_sequences(self):
        self.assertEqual(left_normalise_insertion('AT', 2, 'CT'), (1, 'TC'))

    def test_right_repeat_unit_boundary_can_normalise_to_left_boundary(self):
        self.assertEqual(left_normalise_insertion('ATAT', 4, 'AT'), (0, 'AT'))


if __name__ == '__main__':
    unittest.main()
