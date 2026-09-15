"""Diagnostic operating points never redefine the frozen null estimator."""
import json
import os
import shutil
import tempfile
import unittest

from advntr.background_estimator import FitterError, HYPERPARAMETERS
from advntr.background_fit_command import parse_args


class TestBackgroundFitPolicy(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix='advntr-fit-policy-')
        self.base = ['--capture-root', self.root, '--labels', 'invented.json',
                     '--partition', 'training', '--out-dir', os.path.join(self.root, 'out'), '--profile', 'invented']

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_explicit_recipe_and_default_diagnostic_are_independent_of_settings(self):
        from advntr import settings
        from advntr.background_fit_policy import resolve_diagnostic_policy
        original = settings.MIN_SUPPORTING_READ_COUNT, settings.INDEL_MUTATION_MIN_PVALUE
        try:
            settings.MIN_SUPPORTING_READ_COUNT, settings.INDEL_MUTATION_MIN_PVALUE = 19, 0.4
            args = parse_args(self.base + ['--background-recipe', 'recipe-v1'])
            result = resolve_diagnostic_policy(args)
            self.assertEqual('exact', result['mode'])
            self.assertEqual(3, result['minimum_read_support'])
            self.assertEqual(0.001, result['cutoff'])
            self.assertEqual(4, HYPERPARAMETERS['kprot'])
            self.assertEqual(0.001, HYPERPARAMETERS['floor_target'])
        finally:
            settings.MIN_SUPPORTING_READ_COUNT, settings.INDEL_MUTATION_MIN_PVALUE = original

    def test_policy_is_closed_and_preserves_explicit_values(self):
        from advntr.background_fit_policy import resolve_diagnostic_policy
        path = os.path.join(self.root, 'policy.json')
        policy = {'schema_version': 'advntr-frameshift-policy-v1', 'mode': 'exact',
                  'cutoff': 0.002, 'minimum_read_support': 5}
        with open(path, 'w') as handle:
            json.dump(policy, handle)
        args = parse_args(self.base + ['--diagnostic-policy', path])
        self.assertEqual(policy, resolve_diagnostic_policy(args))
        with open(path, 'w') as handle:
            handle.write('{"mode":"exact","mode":"legacy"}')
        with self.assertRaises(FitterError):
            resolve_diagnostic_policy(args)

    def test_v1_evidence_refuses_unreconstructable_support_or_mode(self):
        from advntr.background_fit_policy import require_diagnostic_capture
        from advntr.background_capture import Capture
        capture = Capture('invented', 'unused', [])
        baseline = {'schema_version': 'advntr-frameshift-policy-v1', 'mode': 'exact',
                    'cutoff': 0.002, 'minimum_read_support': 3}
        require_diagnostic_capture(capture, baseline)
        for policy in (dict(baseline, mode='legacy'), dict(baseline, minimum_read_support=2)):
            with self.assertRaises(FitterError):
                require_diagnostic_capture(capture, policy)

    def test_cli_rejects_unknown_recipe_duplicates_and_abbreviations(self):
        for extra in (['--background-recipe', 'recipe-v2'], ['--background-rec', 'recipe-v1'],
                      ['--diagnostic-policy', 'one', '--diagnostic-policy', 'two']):
            with self.assertRaises(SystemExit):
                parse_args(self.base + extra)
        self.assertEqual(['one', 'two'], parse_args(self.base + ['--note', 'one', '--note', 'two']).note)


if __name__ == '__main__':
    unittest.main()
