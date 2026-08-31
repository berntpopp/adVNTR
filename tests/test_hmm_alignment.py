"""Mutation-token joining in the `-aln` sidecar.

`generate_aln` builds a mutation string by joining tokens with '&'. Two branches do
that join: Case 1 (a deletion following a deletion) and Case 3 (an insertion following
a deletion at the same index). Case 1 guarded against an unopened sequence; Case 3 did
not, so an insertion arriving with no sequence open raised

    TypeError: unsupported operand type(s) for +=: 'NoneType' and 'str'

which exited 1 on example_6c28_hg19_subset.bam *after* the genotype table had already
been written -- so a correct result was produced and then discarded by any caller that
checks the exit code. VNtyper works around it by not passing -aln at all.
"""
import unittest

from advntr.hmm_alignment import _extend_mutation_sequence


class TestExtendMutationSequence(unittest.TestCase):

    def test_an_open_sequence_is_joined_with_an_ampersand(self):
        self.assertEqual(_extend_mutation_sequence('D11_2', 'D12_2'), 'D11_2&D12_2')

    def test_an_unopened_sequence_starts_with_the_addition(self):
        """The Case 3 crash: no preceding deletion is open, so the insertion is the
        start of a new sequence rather than a continuation of nothing."""
        self.assertEqual(_extend_mutation_sequence(None, 'I22_2_G_LEN1'), 'I22_2_G_LEN1')

    def test_an_unopened_sequence_never_gains_a_leading_separator(self):
        """A leading '&' would produce a token that can never match a reported state."""
        self.assertFalse(_extend_mutation_sequence(None, 'I22_2_G_LEN1').startswith('&'))

    def test_joining_is_associative_across_three_tokens(self):
        first = _extend_mutation_sequence(None, 'D17_2')
        second = _extend_mutation_sequence(first, 'D18_2')
        self.assertEqual(_extend_mutation_sequence(second, 'D19_2'),
                         'D17_2&D18_2&D19_2')

    def test_an_empty_string_is_treated_as_open_not_absent(self):
        """Only None means 'nothing open'. An empty string is a real, if odd, value and
        must not silently swallow the separator -- that distinction is what keeps the
        guard from masking a different bug."""
        self.assertEqual(_extend_mutation_sequence('', 'D12_2'), '&D12_2')


if __name__ == '__main__':
    unittest.main()
