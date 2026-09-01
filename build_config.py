"""Build configuration shared by setup.py and setup_hmm.py.

Lives in its own module because setup.py calls setup() at import time, so importing
directives from it would run a build as a side effect.
"""

# Cython compiler directives, set explicitly rather than inherited.
#
# `language_level` is the important one: leaving it unset makes Cython 0.29 default to 2
# with a FutureWarning, and a Cython 3.x upgrade would silently switch the semantics of
# every module. Pin it.
#
# `wraparound` MUST stay True. hmm/hmm.pyx relies on negative indexing; turning it off
# with boundscheck already disabled segfaults (verified empirically, not theorised).
#
# `-O3` is NOT set and does not need to be: conda's Python 2.7 CFLAGS already append it
# after -O2, and gcc takes the last -O. Never add -ffast-math -- it reassociates floating
# point and would break the -inf propagation the decoder relies on.
CYTHON_DIRECTIVES = {
    'language_level': '2',
    'boundscheck': False,
    'wraparound': True,
    'cdivision': True,
    'initializedcheck': False,
}

# pomegranate/ is NOT compiled.
#
# It is dead code on every supported path: advntr/settings.py sets USE_ENHANCED_HMM = True,
# which routes advntr/hmm_utils.py and advntr/vntr_finder.py to hmm/hmm.pyx instead. And it
# does not build on gcc >= 14 -- pomegranate/distributions.pyx uses old scipy cython_blas
# bindings that pass long* where int* is expected, which made `setup.py build_ext` fail
# outright and so blocked building this package from source at all. The source tree stays
# for reference. See FORK.md.
#
# The glob compiles every .pyx under hmm/ as its own extension module, which is what
# produces TWO modules from the Viterbi DP source rather than one: hmm/hmm.pyx (module
# hmm.hmm, production) and hmm/hmm_instrumented.pyx (module hmm.hmm_instrumented,
# test-only) each `include` hmm/_viterbi_fill_core.pxi -- a .pxi, so this glob does not
# also try to compile it standalone -- with a different `DEF INSTRUMENTED`, so `make
# build` produces both from one hand-maintained DP source without any extra wiring here.
# See _viterbi_fill_core.pxi's docstring for why two compiled modules exist instead of
# one runtime flag (Task 3 fix round 1; task-3-report.md).
EXTENSION_SOURCES = ["hmm/*.pyx"]
