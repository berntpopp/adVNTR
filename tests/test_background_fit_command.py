"""Tests for advntr fit-background command."""
import shutil
import subprocess
import sys
import tempfile
import unittest

from advntr import background_fit_command


class TestBackgroundFitCommand(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='task8c-cmd-')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_parse_args_help(self):
        with self.assertRaises(SystemExit) as cm:
            background_fit_command.parse_args(['--help'])
        self.assertEqual(cm.exception.code, 0)

    def test_parse_args_missing_required(self):
        with self.assertRaises(SystemExit) as cm:
            background_fit_command.parse_args([])
        self.assertNotEqual(cm.exception.code, 0)

    def test_effective_hyperparameters_default(self):
        class DummyArgs(object):
            screen_min_samples = None

        hp, banner, overrides = background_fit_command.effective_hyperparameters(DummyArgs())
        self.assertEqual(banner, '')
        self.assertEqual(overrides, {})
        self.assertEqual(hp['dispersion_min_samples'], 15)

    def test_effective_hyperparameters_override(self):
        class DummyArgs(object):
            screen_min_samples = 5

        hp, banner, overrides = background_fit_command.effective_hyperparameters(DummyArgs())
        self.assertIn('PRE-REGISTRATION OVERRIDDEN', banner)
        self.assertEqual(overrides, {'dispersion_min_samples': 5})
        self.assertEqual(hp['dispersion_min_samples'], 5)

    def test_cli_invocation_via_module(self):
        cmd = [sys.executable, '-m', 'advntr', 'fit-background', '--help']
        output = subprocess.check_output(cmd)
        self.assertIn('fit-background', output)
        self.assertIn('--capture-root', output)
        self.assertIn('--profile', output)


if __name__ == '__main__':
    unittest.main()
