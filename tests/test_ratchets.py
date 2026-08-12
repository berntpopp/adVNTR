"""The ratchets must FAIL on violation, not merely pass when satisfied.

A gate nobody has seen fail is a gate nobody knows works.
"""
import contextlib
import os
import sys
import unittest

class _Sink(object):
    """Accepts anything. io.StringIO would reject Python 2 byte strings."""

    def write(self, _text):
        pass

    def flush(self):
        pass


@contextlib.contextmanager
def _quiet():
    """Swallow the ratchet's diagnostic output.

    Two tests below deliberately trip the ratchet. Letting them print makes a passing
    run look like a failing one, which is how real failures get ignored.
    """
    saved, sys.stderr = sys.stderr, _Sink()
    try:
        yield
    finally:
        sys.stderr = saved

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'scripts'))

import loc_ratchet  # noqa: E402


class TestLocRatchet(unittest.TestCase):
    def setUp(self):
        self._limit = loc_ratchet.NEW_FILE_LIMIT
        self._grandfathered = dict(loc_ratchet.GRANDFATHERED)
        self._cwd = os.getcwd()
        os.chdir(REPO)

    def tearDown(self):
        loc_ratchet.NEW_FILE_LIMIT = self._limit
        loc_ratchet.GRANDFATHERED = self._grandfathered
        os.chdir(self._cwd)

    def test_passes_on_the_current_tree(self):
        self.assertEqual(loc_ratchet.main(), 0)

    def test_fails_when_a_file_exceeds_the_new_file_limit(self):
        loc_ratchet.NEW_FILE_LIMIT = 10
        loc_ratchet.GRANDFATHERED = {}
        with _quiet():
            self.assertEqual(loc_ratchet.main(), 1)

    def test_fails_when_a_grandfathered_file_grows(self):
        """The ceiling may only come down. Lowering one below its real size must fail."""
        loc_ratchet.GRANDFATHERED = {'hmm/hmm.pyx': 1}
        with _quiet():
            self.assertEqual(loc_ratchet.main(), 1)

    def test_pomegranate_is_excluded(self):
        """It is not compiled and not maintained; holding it to the limit would make the
        ratchet permanently red for code nobody may touch."""
        self.assertIn('pomegranate/', loc_ratchet.EXCLUDED_PREFIXES)

    def test_grandfathered_entries_all_still_exist(self):
        """A stale entry silently stops enforcing anything for that path."""
        for path in loc_ratchet.GRANDFATHERED:
            self.assertTrue(os.path.isfile(os.path.join(REPO, path)),
                            'grandfathered path no longer exists: %s' % path)

    def test_grandfathered_ceilings_are_not_below_reality(self):
        """If a ceiling is below the real size the ratchet is already failing, which
        means someone lowered it without shrinking the file."""
        for path, ceiling in loc_ratchet.GRANDFATHERED.items():
            actual = sum(1 for _ in open(os.path.join(REPO, path)))
            self.assertLessEqual(actual, ceiling,
                                 '%s is %d lines, ceiling %d' % (path, actual, ceiling))


class TestCoverageBaseline(unittest.TestCase):
    def test_baseline_file_exists_and_parses(self):
        path = os.path.join(REPO, '.coverage-baseline')
        self.assertTrue(os.path.isfile(path))
        with open(path) as handle:
            value = float(handle.read().strip())
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 100.0)


class TestNoUpstreamRemote(unittest.TestCase):
    def test_this_is_a_hard_fork(self):
        """FORK.md says no upstream remote; assert the repo agrees with its own docs."""
        import subprocess
        os.chdir(REPO)
        remotes = subprocess.check_output(['git', 'remote']).split()
        self.assertNotIn('upstream', remotes)


if __name__ == '__main__':
    unittest.main()
