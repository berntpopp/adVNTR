"""Mutation-token joining in the `-aln` sidecar.

`generate_aln` builds a mutation string by joining tokens with '&'. A completed D/I event
closes that accumulator. The cross-HMM branch then advanced `prev_mutation` without opening
the accumulator for the first token in the new HMM, so the following join either dropped that
token or raised

    TypeError: unsupported operand type(s) for +=: 'NoneType' and 'str'

which exited 1 on example_6c28_hg19_subset.bam *after* the genotype table had already
been written. The regression below constrains the state transition, not only the generic join.
"""
import os
import shutil
import tempfile
import unittest

import advntr.hmm_alignment as hmm_alignment
from advntr.hmm_alignment import _extend_mutation_sequence


class _ReferenceVNTR(object):
    repeat_segments = ['ACGTACGTAA', 'ACGTACGTAC', 'ACGTACGTAG',
                       'ACGTACGTAT', 'ACGTACGTCA', 'ACGTACGTCC',
                       'ACGTACGTCG', 'ACGTACGTCT', 'ACGTACGTGA']

    def get_repeat_segments(self):
        return self.repeat_segments


class _InPlaceMutationSequence(object):
    """Expose the semantic difference between the old ``+=`` and ordinary ``+``."""

    def __init__(self, value):
        self.value = value

    def __iadd__(self, addition):
        self.value += addition
        return self

    def __add__(self, _addition):
        raise AssertionError('the previous join used __iadd__, not __add__')


class TestExtendMutationSequence(unittest.TestCase):

    def test_an_open_sequence_is_joined_with_an_ampersand(self):
        self.assertEqual(_extend_mutation_sequence('D11_2', 'D12_2'), 'D11_2&D12_2')

    def test_an_unopened_sequence_starts_with_the_addition(self):
        """The defensive join fallback never prefixes an addition with `None&`."""
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

    def test_an_open_sequence_retains_the_previous_in_place_addition_semantics(self):
        sequence = _InPlaceMutationSequence('D11_2')
        result = _extend_mutation_sequence(sequence, 'D12_2')
        self.assertIs(result, sequence)
        self.assertEqual(result.value, 'D11_2&D12_2')


class TestGenerateAlnMutationSequences(unittest.TestCase):

    def test_a_new_hmm_opens_with_its_first_mutation(self):
        """Closing a D/I event sets the accumulator to None. If the following event
        belongs to another HMM, that first token must open the new sequence; otherwise
        Case 3 drops its deletion and emits only the insertion."""
        work = tempfile.mkdtemp(prefix='advntr-hmm-alignment-')
        logfile = os.path.join(work, 'input.log')
        target = 'D2_9&I2_9_A_LEN2'
        insertion_target = 'I3_8_A_LEN2'
        prefix = 'X' * 24
        lines = [
            prefix + 'INFO:VID:25561, There is a mutation at ' + target + '\n',
            prefix + 'INFO:VID:25561, There is a mutation at ' + insertion_target + '\n',
            prefix + 'DEBUG:finding repeat count from alignment file for 25561\n',
            prefix + 'DEBUG:ReadName:synthetic-read\n',
            prefix + 'DEBUG:Read:AAAAAA\n',
            prefix + "DEBUG:VisitedStates:['unit_start_2', 'D1_2', 'I1_2', "
                     "'I1_2', 'unit_end_2', 'unit_start_9', 'D2_9', 'I2_9', "
                     "'I2_9', 'unit_end_9', 'unit_start_8', 'I3_8', 'I3_8', "
                     "'unit_end_8']\n",
        ]
        with open(logfile, 'w') as handle:
            handle.writelines(lines)

        original = hmm_alignment.get_repeating_unit_state_count

        def repeat_counts(_states, _sequence, _clusters):
            counts = {'M': 9, 'I': 2, 'D': 1, 'S': 0}
            insertion_counts = {'M': 8, 'I': 2, 'D': 0, 'S': 0}
            return {0: counts.copy(), 1: counts.copy(),
                    2: insertion_counts}, None, None

        hmm_alignment.get_repeating_unit_state_count = repeat_counts
        try:
            hmm_alignment.generate_aln(
                logfile, ref_vntr_dict={25561: _ReferenceVNTR()})
            with open(logfile + '.aln') as handle:
                output = handle.read()
            self.assertIn('#Mutation ' + target, output)
            self.assertIn('#Mutation ' + insertion_target, output)
            self.assertIn('synthetic-read', output)
        finally:
            hmm_alignment.get_repeating_unit_state_count = original
            shutil.rmtree(work)


if __name__ == '__main__':
    unittest.main()
