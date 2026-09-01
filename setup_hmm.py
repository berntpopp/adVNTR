"""Build only the hmm/ extension in place.

setup.py builds the installable package; this is the fast path for iterating on
hmm/hmm.pyx without reinstalling. CI uses it too.

    python setup_hmm.py build_ext --inplace

Builds DEV_EXTENSION_SOURCES, not PRODUCTION_EXTENSION_SOURCES: this is the only build
that also needs hmm/hmm_instrumented.pyx (test-only; tests/test_decoder_workload.py
loads it via advntr_harness/workload.py). setup.py -- the installable package --
deliberately does not build it. See build_config.py.
"""
from distutils.core import setup

import numpy
from Cython.Build import cythonize

from build_config import CYTHON_DIRECTIVES, DEV_EXTENSION_SOURCES

setup(
    name='hmm-only',
    ext_modules=cythonize(DEV_EXTENSION_SOURCES, compiler_directives=CYTHON_DIRECTIVES),
    include_dirs=[numpy.get_include()],
)
