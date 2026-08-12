# setuptools MUST be imported before Cython.Build. Cython picks which Extension base
# class to emit depending on whether setuptools is already in sys.modules; import it
# second and cythonize() returns distutils Extensions that setuptools' setup() then
# rejects with "each element of 'ext_modules' option must be an Extension instance".
from setuptools import find_packages, setup  # isort: skip

import numpy
from Cython.Build import cythonize

from advntr import __version__
from build_config import CYTHON_DIRECTIVES, EXTENSION_SOURCES

setup(name='advntr',
      version=__version__,
      description='A tool for genotyping Variable Number Tandem Repeats (VNTR) from sequence data',
      author='Mehrdad Bakhtiari',
      author_email='mbakhtia@ucsd.edu',
      license='BSD-3-Clause',
      url='https://github.com/berntpopp/adVNTR',
      test_suite='tests',
      # advntr_harness and scripts are development tooling, not part of the tool.
      packages=find_packages(exclude=['tests', 'tests.*', 'pomegranate', 'pomegranate.*',
                                      'advntr_harness', 'advntr_harness.*',
                                      'scripts', 'scripts.*']),
      package_dir={'advntr': 'advntr'},
      install_requires=['scipy', 'biopython', 'cython', 'scikit-learn'],
      provides=["advntr"],
      entry_points={
            'console_scripts': ['advntr=advntr.__main__:main']
      },
      ext_modules=cythonize(
            EXTENSION_SOURCES,
            compiler_directives=CYTHON_DIRECTIVES,
            nthreads=4,
      ),
      include_dirs=[numpy.get_include()],
      classifiers=["Environment :: Console",
                   "Intended Audience :: Developers",
                   "Intended Audience :: Science/Research",
                   "Operating System :: Unix",
                   "Programming Language :: Python",
                   "Programming Language :: Python :: 2",
                   "Topic :: Scientific/Engineering :: Bio-Informatics"],
      )
