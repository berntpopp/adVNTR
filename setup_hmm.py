"""Build only the hmm/ extension in place.

setup.py builds the installable package; this is the fast path for iterating on
hmm/hmm.pyx without reinstalling. CI uses it too.

    python setup_hmm.py build_ext --inplace
"""
from distutils.core import setup

import numpy
from Cython.Build import cythonize

from build_config import CYTHON_DIRECTIVES, EXTENSION_SOURCES

setup(
    name='hmm-only',
    ext_modules=cythonize(EXTENSION_SOURCES, compiler_directives=CYTHON_DIRECTIVES),
    include_dirs=[numpy.get_include()],
)
