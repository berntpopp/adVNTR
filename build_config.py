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

# What ships to end users via `setup.py` (`pip install .`), enumerated rather than
# globbed. hmm/hmm.pyx (module hmm.hmm, production) `include`s hmm/_viterbi_fill_core.pxi
# -- a .pxi, so it is never compiled as an extension of its own -- for the actual
# Viterbi DP fill (Task 3 fix round 1; task-3-report.md).
#
# NOT `hmm/*.pyx`: that glob also matches hmm/hmm_instrumented.pyx (module
# hmm.hmm_instrumented, test-only counters + a skip_enabled toggle, `include`-ing the
# identical .pxi with a different compile-time DEF), and this list is shared with
# setup_hmm.py's dev build, so a glob here would ship it to every `pip install` too.
# FORK.md's 2.0.1 entry records exactly this failure mode already happening once:
# `find_packages()` shipped `advntr_harness` and `scripts/` -- development tooling --
# into the installed egg, caught only by installing in Docker and importing from
# outside the repo. An enumerated list can't silently absorb a new test-only file the
# way a glob can; anything added to hmm/ ships only if it is added here too.
PRODUCTION_EXTENSION_SOURCES = ["hmm/base.pyx", "hmm/hmm.pyx"]

# The dev/test build (setup_hmm.py, `make build`) additionally compiles the test-only
# instrumented module, so `tests/test_decoder_workload.py` (via
# advntr_harness/workload.py) has something to load. Never used by setup.py.
DEV_EXTENSION_SOURCES = PRODUCTION_EXTENSION_SOURCES + ["hmm/hmm_instrumented.pyx"]
