"""Test clean refusal of --update / -u (Issue #9).

Model refinement via -u relied on Model.from_matrix, which was removed with the
enhanced Cython HMM backend. -u must be refused at CLI parse time with an actionable
error message, and iteratively_update_model must raise NotImplementedError if called
programmatically.
"""
import argparse
import unittest

from advntr.advntr_commands import genotype
from advntr.reference_vntr import ReferenceVNTR
from advntr.vntr_finder import VNTRFinder


class TestModelUpdateRefusal(unittest.TestCase):

    def test_cli_refuses_update_flag_with_informative_error(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        genotype_parser = subparsers.add_parser('genotype')
        genotype_parser.add_argument('-u', '--update', action='store_true', default=True)
        genotype_parser.add_argument('-a', '--alignment_file', default='test.bam')

        args = argparse.Namespace(
            update=True,
            alignment_file='test.bam',
            fasta=None,
            nanopore=False,
            pacbio=False,
            threads=1,
            prune_reverse=False,
            exact_frameshift_caller=False,
            frameshift_background=None,
            frameshift_calibration_out=None,
            rare_unit_coverage_guard=None,
            expansion=False,
            coverage=None,
            working_directory='.'
        )

        with self.assertRaises(SystemExit) as cm:
            genotype(args, genotype_parser)

        self.assertIn('--update / -u is unsupported on this fork', str(cm.exception))
        self.assertIn('enhanced HMM backend', str(cm.exception))

    def test_iteratively_update_model_raises_not_implemented_error(self):
        ref_vntr = ReferenceVNTR(1, 'ACGT', 100, 'chr1', 'TEST', 'Coding')
        finder = VNTRFinder(ref_vntr)
        with self.assertRaises(NotImplementedError) as cm:
            finder.iteratively_update_model('fake.bam', [], [], None)

        self.assertIn('iteratively_update_model is unsupported on this fork', str(cm.exception))


if __name__ == '__main__':
    unittest.main()
