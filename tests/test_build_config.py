"""The build configuration is load-bearing, so it is pinned rather than trusted.

Three of these settings are not style choices: one prevents a silent semantic change on
a Cython upgrade, one segfaults if flipped, and one keeps a Python error path out of code
that runs without the GIL.
"""
import os
import unittest

from build_config import CYTHON_DIRECTIVES, EXTENSION_SOURCES

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestCythonDirectives(unittest.TestCase):
    def test_language_level_is_explicit(self):
        """Unset, Cython 0.29 defaults to 2 with a FutureWarning and a 3.x upgrade
        silently switches every module's semantics."""
        self.assertEqual(CYTHON_DIRECTIVES['language_level'], '2')

    def test_wraparound_stays_enabled(self):
        """Turning this off segfaults: the code relies on negative indexing and
        boundscheck is already disabled, so there is nothing left to catch it.
        Verified empirically, not theorised."""
        self.assertTrue(CYTHON_DIRECTIVES['wraparound'])

    def test_boundscheck_is_disabled(self):
        self.assertFalse(CYTHON_DIRECTIVES['boundscheck'])

    def test_initializedcheck_is_disabled(self):
        """Keeps a PyErr_SetString branch out of the nogil DP."""
        self.assertFalse(CYTHON_DIRECTIVES['initializedcheck'])

    def test_no_fast_math_anywhere(self):
        """-ffast-math reassociates floating point and breaks the -inf propagation the
        decoder relies on for its rejection test.

        Checks non-comment lines only: build_config.py deliberately mentions the flag in
        a comment warning against it, and a naive substring search would flag that.
        """
        for name in ('setup.py', 'setup_hmm.py', 'build_config.py', 'Makefile'):
            path = os.path.join(REPO, name)
            if not os.path.isfile(path):
                continue
            for number, line in enumerate(open(path), 1):
                if line.lstrip().startswith('#'):
                    continue
                self.assertNotIn('ffast-math', line,
                                 '%s:%d enables -ffast-math' % (name, number))


class TestExtensionSources(unittest.TestCase):
    def test_pomegranate_is_not_compiled(self):
        """It is unreachable at runtime and does not build on gcc >= 14, which made
        `setup.py build_ext` fail outright. See FORK.md."""
        for pattern in EXTENSION_SOURCES:
            self.assertNotIn('pomegranate', pattern)

    def test_hmm_is_compiled(self):
        self.assertIn('hmm/*.pyx', EXTENSION_SOURCES)


class TestBuildFilesAgree(unittest.TestCase):
    def test_setup_and_setup_hmm_share_one_source_of_truth(self):
        """Divergent directives between the packaging build and the dev build would mean
        testing a different binary than you ship."""
        for name in ('setup.py', 'setup_hmm.py'):
            with open(os.path.join(REPO, name)) as handle:
                source = handle.read()
            self.assertIn('from build_config import', source,
                          '%s does not use the shared build config' % name)

    def test_setuptools_is_imported_before_cython_in_setup(self):
        """Cython picks its Extension base class depending on whether setuptools is
        already in sys.modules. Import it second and cythonize() returns Extensions that
        setuptools' setup() rejects outright."""
        with open(os.path.join(REPO, 'setup.py')) as handle:
            source = handle.read()
        self.assertLess(source.index('from setuptools import'),
                        source.index('from Cython.Build import'))


class TestCleanDoesNotDeleteSources(unittest.TestCase):
    def test_clean_target_does_not_glob_hmm_c_files(self):
        """hmm/queue.c is a hand-written tracked source, not Cython output. A
        `rm -f hmm/*.c` would delete it and git would not flag it as missing."""
        for number, line in enumerate(open(os.path.join(REPO, 'Makefile')), 1):
            if line.lstrip().startswith('#'):
                continue
            self.assertNotIn('hmm/*.c', line,
                             'Makefile:%d globs hmm/*.c' % number)

    def test_gitignore_does_not_ignore_queue_c(self):
        for number, line in enumerate(open(os.path.join(REPO, '.gitignore')), 1):
            if line.lstrip().startswith('#'):
                continue
            self.assertNotIn('hmm/*.c', line,
                             '.gitignore:%d ignores hmm/*.c' % number)

    def test_queue_c_still_exists(self):
        self.assertTrue(os.path.isfile(os.path.join(REPO, 'hmm', 'queue.c')))


if __name__ == '__main__':
    unittest.main()
