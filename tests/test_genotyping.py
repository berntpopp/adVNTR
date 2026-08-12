"""Copy-number genotyping from observed repeat counts.

These tests were written when `find_genotype_based_on_observed_repeats` returned the
genotype pair alone. It now returns `(genotype_pair, max_probability)`. The assertions
below unpack it; every expected pair is unchanged, so this was API drift in the tests,
not a behaviour regression.
"""
import unittest

from advntr.reference_vntr import ReferenceVNTR
from advntr.vntr_finder import VNTRFinder


class TestGenotyping(unittest.TestCase):

    def get_reference_vntr(self):
        return ReferenceVNTR(1, 'CACA', 1000, 'chr1', None, None)

    def genotype_for(self, observed, is_haploid=False):
        """Return the genotype pair in ascending order, discarding the probability."""
        vntr_finder = VNTRFinder(self.get_reference_vntr(), is_haploid=is_haploid)
        genotype, _max_probability = vntr_finder.find_genotype_based_on_observed_repeats(observed)
        return tuple(sorted(genotype))

    def test_statistical_model_for_haploid_case(self):
        self.assertEqual(self.genotype_for([3, 3, 3, 3, 3]), (3, 3))

    def test_statistical_model_for_haploid_organism(self):
        self.assertEqual(self.genotype_for([2, 3, 3, 3, 3], is_haploid=True), (3, 3))

    def test_statistical_model_for_diploid_case(self):
        self.assertEqual(self.genotype_for([2, 2, 3, 3, 3]), (2, 3))

    def test_statistical_model_for_erroneous_diploid_case(self):
        self.assertEqual(self.genotype_for([4, 5, 5, 5, 7, 8, 8, 8, 9]), (5, 8))

    def test_a_probability_is_returned_alongside_the_genotype(self):
        vntr_finder = VNTRFinder(self.get_reference_vntr())
        _genotype, probability = vntr_finder.find_genotype_based_on_observed_repeats(
            [3, 3, 3, 3, 3])
        self.assertGreater(probability, 0.0)
        self.assertLessEqual(probability, 1.0)

    def test_recruit_read_for_positive_read(self):
        vntr_finder = VNTRFinder(self.get_reference_vntr())
        results = vntr_finder.recruit_read(logp=-20, vpath=[],
                                           min_score_to_count_read=-50, read_length=100)
        self.assertEqual(results, True)


if __name__ == '__main__':
    unittest.main()
