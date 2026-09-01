
#cython: boundscheck=False
#cython: cdivision=True

from collections import defaultdict
from operator import attrgetter
import threading

from .base cimport DiscreteDistribution
from .base cimport State

import numpy as np
cimport numpy as np
from libc.math cimport log, INFINITY
from libc.stdlib cimport malloc, realloc, free

cimport cython

# `_viterbi_fill` itself lives in _viterbi_fill_core.pxi, included below and again
# (with a different DEF) from hmm_instrumented.pyx. See that file's docstring for why:
# a runtime `counters != NULL` guard in the hot loop, however written, still measured
# 4.2-4.5% on this build (Task 3 fix round 1) -- over the ~1% budget no matter how the
# branch is shaped -- so production compiles the guards out entirely instead of hiding
# them behind one.
DEF INSTRUMENTED = False
include "_viterbi_fill_core.pxi"

#: 256-entry base encoder, indexed by ord(); -1 is the sentinel get_encoded_sequence()
#: raises KeyError on -- an array index replacing a dict lookup, same failure on an
#: undeclared symbol (bake() already refuses a model whose emitting states do not
#: declare all of A/C/G/T, precisely so that failure stays loud, not a silent 0).
_BASE_CODE = [-1] * 256
_BASE_CODE[ord('A')], _BASE_CODE[ord('C')] = 0, 1
_BASE_CODE[ord('G')], _BASE_CODE[ord('T')] = 2, 3

#: Per-thread vpath scratch ONLY (threading.local(): genuinely per-thread, never on
#: self -- advntr/read_selection.py's docstring is why that matters). The score
#: table is deliberately NOT amortised here -- see AGENTS.md Traps ("Reusing
#: viterbi()'s small score table..."). Re-keyed on n_states, never aliased: a numpy
#: row-slice narrower than the buffer's own shape is not Fortran-contiguous (only a
#: column slice is), so a differently sized model gets a wholly fresh buffer rather
#: than a sub-view of a bigger one. Column count (read length) alone grows in place.
_dp_scratch = threading.local()

def _thread_scratch(n_states, n_cols):
    cached = getattr(_dp_scratch, 'buffers', None)
    if cached is None or cached[1] != n_states or cached[2] < n_cols:
        cols = max(n_cols, cached[2]) if cached is not None and cached[1] == n_states else n_cols
        cached = (np.zeros((n_states, cols), dtype=np.intc, order='F'), n_states, cols)
        _dp_scratch.buffers = cached
    return cached[0][:, :n_cols]


@cython.wraparound(False)
@cython.boundscheck(False)
cdef int _traceback(int[::1, :] vpath_table_row, unsigned char[::1] silent,
                     int row, int col, int** out_rows) nogil:
    """Walk vpath_table_row to the DP's origin into a malloc'd growable array (caller
    frees, fresh every call, not amortised -- AGENTS.md Traps explains why) instead
    of the old `vpath.insert(0, ...)`, O(n^2) under the GIL for a ~156-entry path
    (task-6-report.md). Runs nogil; the caller appends once and reverses, O(n).
    """
    cdef int cap = 256, n = 0, pred_row, pred_col
    cdef int* buf = <int*> malloc(cap * sizeof(int))
    cdef int* grown
    if buf == NULL:
        return -1
    while True:
        if n == cap:
            grown = <int*> realloc(buf, 2 * cap * sizeof(int))
            if grown == NULL:
                free(buf)
                return -1
            buf, cap = grown, 2 * cap
        buf[n] = row
        n += 1
        if row == 0 and col == 0:
            out_rows[0] = buf
            return n
        pred_row = vpath_table_row[row, col]
        pred_col = col if silent[pred_row] else col - 1
        row, col = pred_row, pred_col


cdef class Model(object):
    """ Hidden Markov Model
        start: a State representing the model start
        end:   a State representing the model end
        states: a list of states
        edges:  a list of edges represented by tuples of (from-state, to-state)

    """
    cdef public char* name
    cdef public char* model

    # states
    cdef public list states
    cdef public int n_states
    cdef public object start # State object
    cdef public object end

    # store transitions as a map
    cdef public dict transition_map
    cdef public dict neighbors
    # 2D matrix (After conforming the topology, create one matrix for visualization)
    cdef double[:, ::1] transition_matrix
    # cdef int[:,:] neighboring_state_indices
    # public: hmm_instrumented.pyx's decode_instrumented() reads it off an already-baked
    # Model to find start_index, without duplicating any construction/bake() logic.
    cdef public dict state_to_index

    # Flat, index-addressed mirrors of the graph, built in bake() and read by the
    # Viterbi DP. Same content as `neighbors` / `State.is_silent()` /
    # `DiscreteDistribution.log_emission`, but reachable without touching a Python
    # object, so the inner loop can run without the GIL.
    cdef public int[::1] nbr_indptr        # CSR row pointers, length n_states + 1
    cdef public int[::1] nbr_indices       # CSR column indices, length n_edges
    cdef public double[::1] nbr_logp       # CSR edge weights, PARALLEL to nbr_indices
    cdef public unsigned char[::1] silent  # 1 if state emits nothing
    cdef public double[:, ::1] emissions   # log emission per (state, base in 0..3)

    # edges
    cdef list edges
    cdef int n_edges

    cdef public int start_index  # will be set in bake
    cdef public int end_index # will be set in bake

    cdef public np.ndarray dynamic_table

    cdef public list subModels
    cdef public int n_subModels

    cdef public bint is_baked

    cdef public int read_length_used_to_build_model
    cdef public double dp_score_threshold

    def __init__(self, name=None, start=None, end=None):
        # Save the name or make up a name.
        self.name = str(name) or str(id(self))
        self.model = "HiddenMarkovModel"

        # states
        self.states = []
        self.n_states = 0
        self.start = start or State(None, name=self.name + "-start")
        self.end = end or State(None, name=self.name + "-end")

        # store transitions as a map
        self.transition_map = dict()
        self.neighbors = dict()
        self.state_to_index = dict()

        # Put start and end in the states
        self.add_states(self.start, self.end)

        # edges
        self.edges = []
        self.n_edges = 0

        self.start_index = -1     # will be set in bake
        self.end_index = -1       # will be set in bake

        self.dynamic_table = None

        self.subModels = [self]
        self.n_subModels = 1

        self.is_baked = False

        self.read_length_used_to_build_model = 0
        self.dp_score_threshold = -np.inf

    def append_subModel(self, other):
        self.subModels.append(other)

    def add_model(self, model):
        pass

    def add_state(self, state):
        self.states.append(state)
        self.n_states += 1
        # initialize transition map
        self.transition_map[state] = defaultdict(lambda: 0)

    def add_states(self, *states):
        for state in states:
            if isinstance( state, list ):
                for s in state:
                    self.add_state( s )
            else:
                self.add_state( state )

    def state_count(self):
        return self.n_states

    def set_transition(self, from_state, to_state, probability):
        self.transition_map[from_state][to_state] = probability

    def add_transition(self, from_state, to_state, probability, pseudocount=None):
        if from_state not in self.states:
            print("ERROR: No such state named {}".format(from_state.name))
            raise Exception("No such state")
        elif to_state not in self.states:
            print("ERROR: No such state named {}".format(to_state.name))
            raise Exception("No such state")
        else:
            self.transition_map[from_state][to_state] = probability

    def add_transitions(self, transitions):
        pass

    def bake(self, read_length=None, dp_score_threshold=None, merge=None, sort_by_name=False):
        """
        In a model, start state comes the first and end state comes the last.
        Other states are in the middle, and they are sorted by their name.
        e.g.)
        start - I0 - D1 - M1, I1 - D2 - M2 - I2, ... D10 - M10 - I10 - end

        setting start_index and end_index

        setting connections between subModels
        """
        if dp_score_threshold is not None:
            self.dp_score_threshold = dp_score_threshold
        if read_length is not None:
            self.read_length_used_to_build_model = read_length

        # Bake all the subModels
        for subModel in self.subModels:
            if subModel == 0:
                continue
            # Ordering states
            if sort_by_name:
                states_without_start_and_end = [state for state in subModel.states if state is not subModel.start and state is not subModel.end ]
                sorted_states = list(sorted(states_without_start_and_end, key=attrgetter('name')))
                subModel.states = [subModel.start] + sorted_states + [subModel.end]
            else:
                subModel._sort_states()

        # Start is the start state of the very fist sub-model
        self.start = self.subModels[0].start
        # End is the end state of the very last sub-model
        self.end = self.subModels[self.n_subModels-1].end

        # Build aggregate states and transition map from subModels
        n_states = 0
        states = []
        transition_map = dict()
        for subModel in self.subModels:
            if subModel == 0:
                continue
            self.state_to_index.update(dict(zip(subModel.states, range(n_states, n_states+subModel.n_states))))
            n_states += subModel.n_states
            for state in subModel.states:
                states.append(state)
            transition_map.update(subModel.transition_map)

        self.states = states
        self.n_states = n_states
        self.transition_map = transition_map

        # viterbi()'s traceback uses a row it already has as its own index instead of
        # re-deriving it through state_to_index[states[row]] -- sound only because
        # state_to_index is the plain positional inverse of states, built above by
        # zip()ing each subModel's states against a range(). Verified equal for all
        # 2,565 states of the shipped MUC1 model; asserted here so a future subModel
        # topology that breaks the invariant (e.g. a duplicated state object) fails
        # loudly instead of silently addressing the wrong state.
        for index, state in enumerate(self.states):
            if self.state_to_index[state] != index:
                raise ValueError('state_to_index[states[%d]] != %d: viterbi() '
                                 'addresses traceback rows directly and would '
                                 'silently corrupt the path' % (index, index))

        self.transition_matrix = np.zeros((self.n_states, self.n_states), dtype=np.double, order='C')
        cdef int from_index = 0
        cdef int to_index = 0
        for from_state in transition_map.keys():
            outgoing_states = transition_map[from_state]
            self.neighbors[from_state] = sorted([self.state_to_index[ot] for ot in outgoing_states])
            for to_state in outgoing_states.keys():
                from_index = self.state_to_index[from_state]
                to_index = self.state_to_index[to_state]
                self.transition_matrix[from_index][to_index] = log(outgoing_states[to_state])

        # Flatten the graph into CSR + per-state emission/silence tables. Built from
        # exactly the structures the DP used to walk at runtime, so the DP reads the
        # same numbers in the same order -- it just stops paying for a dict lookup
        # keyed by a Python State object on every (state, column) visit.
        indptr = np.zeros(self.n_states + 1, dtype=np.intc)
        silent = np.zeros(self.n_states, dtype=np.uint8)
        # -inf, NOT zero. A zero-filled table turns a symbol the distribution never
        # declared into log(1.0) -- certainty -- whereas DiscreteDistribution.__getitem__
        # (hmm/base.pyx:12) raised KeyError. That would convert a loud failure into a
        # silent wrong answer. The assertion below makes sure the question never arises.
        emissions = np.full((self.n_states, 4), -np.inf, dtype=np.double)
        for state, index in self.state_to_index.items():
            nbrs = self.neighbors.get(state, [])
            indptr[index + 1] = len(nbrs)
            if state.distribution is None:
                silent[index] = 1
            else:
                for base, logp in state.distribution.log_emission.items():
                    emissions[index][base] = logp
                if np.any(np.isneginf(emissions[index])):
                    raise ValueError(
                        'emitting state %d does not declare all of A/C/G/T; a baked '
                        'model must be complete, and the flat emission cache has no '
                        'way to reproduce the KeyError the dict lookup used to raise'
                        % index)
        if not silent[self.n_states - 2]:  # viterbi()'s final relaxation derives, not stores, its column
            raise ValueError('states[n_states-2] (index %d) is not silent, but viterbi()\'s final relaxation derives its target column assuming it is, and would silently corrupt the path' % (self.n_states - 2))
        indptr = np.cumsum(indptr).astype(np.intc)
        flat_indices = np.zeros(indptr[self.n_states], dtype=np.intc)
        flat_logp = np.zeros(indptr[self.n_states], dtype=np.double)
        for state, index in self.state_to_index.items():
            nbrs = self.neighbors.get(state, [])
            flat_indices[indptr[index]:indptr[index] + len(nbrs)] = nbrs
            # COPY the double already in transition_matrix. Do NOT re-evaluate
            # log(p): libm's log is not correctly rounded and can differ in the
            # last ulp across versions, which would break bit-equivalence.
            for offset, nb in enumerate(nbrs):
                flat_logp[indptr[index] + offset] = self.transition_matrix[index][nb]
            # The DP hoists `dynamic_table[row, col]` out of the neighbour loop,
            # which is only sound if a silent state can never relax itself.
            if silent[index] and index in nbrs:
                raise ValueError(
                    'silent self-loop on state %d: the Viterbi DP hoists the '
                    'source cell out of the neighbour loop and would diverge' % index)
        self.nbr_indptr = indptr
        self.nbr_indices = flat_indices
        self.nbr_logp = flat_logp
        self.silent = silent
        self.emissions = emissions

        self.is_baked = True

    def transition_matrix_view(self):
        """Read-only view of the dense transition matrix. For tests only.

        Lets the CSR invariant tests prove `nbr_logp` was COPIED from this matrix rather
        than recomputed with log(). `writeable = False` is load-bearing, not decoration:
        np.asarray on the cdef memoryview ALIASES the model's storage, and the decoder
        reads these edges twice from two copies -- the main DP from `nbr_logp` (:920),
        the hardcoded final relaxation from this matrix (:943). Measured: dropping the
        final edge by 1.0 through a writable view moved a 151-base score from
        -335.85084206362586 to -336.85084206362586 while `nbr_logp[9015]` stayed 0.0.
        """
        view = np.asarray(self.transition_matrix)
        view.flags.writeable = False
        return view

    def _sort_states(self):
        """Sort states into topology order: start, then per repeat-unit
        [unit_start, I0, (D,M,I)*, unit_end], then end -- names are
        I/M/D[index]_[repeat_unit_index], with the flanking unit_start/unit_end
        markers being neither. bake() (Task 6) relies on the resulting self.states
        always ending in the model's own end state, and beginning with start.

        :return: None
        """
        if self.n_states == 2:
            return

        sorted_states = []

        insert_states = defaultdict(list)
        match_states = defaultdict(list)
        delete_states = defaultdict(list)

        dummy_start_states = defaultdict(list)
        dummy_end_states = defaultdict(list)

        for state in self.states:
            repeat_unit_id = state.name.split("_")[-1]

            if state.name.startswith("I"):
                insert_states[repeat_unit_id].append(state)
            elif state.name.startswith("M"):
                match_states[repeat_unit_id].append(state)
            elif state.name.startswith("D"):
                delete_states[repeat_unit_id].append(state)
            else:
                if "_start_" in state.name:
                    dummy_start_states[repeat_unit_id].append(state)
                if "_end_" in state.name:
                    dummy_end_states[repeat_unit_id].append(state)

        for repeat_unit_id, states in insert_states.items():
            states.sort(key=lambda x: int(x.name[1:x.name.find("_")]))

        for repeat_unit_id, states in match_states.items():
            states.sort(key=lambda x: int(x.name[1:x.name.find("_")]))

        for repeat_unit_id, states in delete_states.items():
            states.sort(key=lambda x: int(x.name[1:x.name.find("_")]))

        # 1. Model-start state
        sorted_states.append(self.start)

        # TODO: iterate
        for repeat_unit_id in sorted(dummy_start_states.keys()):
            # unit start or suffix, prefix start
            sorted_states.extend(dummy_start_states[repeat_unit_id])

            # Insert 0
            sorted_states.append(insert_states[repeat_unit_id].pop(0))

            # Delete, Match, Insert
            for i in range(len(match_states[repeat_unit_id])):
                sorted_states.append(delete_states[repeat_unit_id][i])
                sorted_states.append(match_states[repeat_unit_id][i])
                sorted_states.append(insert_states[repeat_unit_id][i])

            # unit end or suffix, prefix end
            sorted_states.extend(dummy_end_states[repeat_unit_id])

        # Model-end state
        sorted_states.append(self.end)
        self.states = sorted_states

    def dense_transition_matrix( self ):
        """The dense (n_states, n_states) transition PROBABILITY matrix (not the log
        matrix bake() builds as self.transition_matrix)."""

        m = len(self.states)
        transition_probabilities = np.zeros( (m, m) )

        for i in range(m):
            state1 = self.states[i]
            for n in range(m):
                state2 = self.states[n]
                if (self.transition_map[state1][state2] > 0.0):
                    transition_probabilities[i, n] = self.transition_map[state1][state2]

        return transition_probabilities

    def concatenate(self, other, suffix='', prefix='', transition_probability=1.0):
        """Append `other` as a new subModel, wiring self.end -> other.start at
        `transition_probability`. `suffix`/`prefix` are accepted for the upstream
        signature but unused here -- no caller on the supported path renames states.
        """
        self.append_subModel(other)
        self.n_subModels += 1

        # setting connections between subModels if they exist
        subModel = self.subModels[self.n_subModels - 1]
        subModel_prev = self.subModels[self.n_subModels - 2]
        subModel_prev.transition_map[subModel_prev.end][subModel.start] = transition_probability

        # Once concatenation happened, it should be baked again
        self.is_baked = False

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cpdef tuple subseq_viterbi(self, sequence, repeat_unit_number):

        cdef Model repeat_matcher_model = self.subModels[1]
        cdef int repeat_start_index = 0
        cdef int repeat_end_index = 0
        for state in repeat_matcher_model.states:
            if state.name == 'unit_start_{}'.format(repeat_unit_number):
                repeat_start_index = self.state_to_index[state]
            if state.name == 'unit_end_{}'.format(repeat_unit_number):
                repeat_end_index = self.state_to_index[state]
                break

        # Initialize dynamic programming table
        # Rows represent states and Columns represent sequence
        cdef int sequence_length = len(sequence)
        cdef int[::1] encoded_sequence = self.get_encoded_sequence(sequence)
        cdef int state_count = repeat_end_index - repeat_start_index + 1

        cdef double[::1,:] dynamic_table = np.full((state_count, sequence_length + 1), -np.inf, dtype=np.double, order='F')
        dynamic_table[0][0] = log(1)

        # Storing previous states row and column separately (Naive version)
        cdef int[::1,:] vpath_table_row = np.zeros((state_count, sequence_length + 1), dtype=np.intc, order='F')
        cdef int[::1,:] vpath_table_col = np.zeros((state_count, sequence_length + 1), dtype=np.intc, order='F')

        cdef int row, col
        cdef char ch
        for col in range(sequence_length):
            for row in range(state_count-1):
                row_index = repeat_start_index + row
                if col != 0 and dynamic_table[row][col] < self.dp_score_threshold:
                    continue
                state = self.states[row_index]
                ch = encoded_sequence[col]
                self._update_tables_for_subseq(row, col, repeat_start_index, repeat_end_index, ch, state, vpath_table_row, vpath_table_col, dynamic_table)

        # For the last update
        col = sequence_length
        for row in range(state_count-1):
            row_index = repeat_start_index + row
            if col != 0 and dynamic_table[row][col] == -np.inf:
                continue
            state = self.states[row_index]

            if state.is_silent():  # Silent state: Stay in the same column
                neighbor_states = self.transition_map[state]
                neighbor_state_index = 0
                log_prob = 0
                for neighbor_state in neighbor_states:
                    neighbor_state_index = self.state_to_index[neighbor_state]
                    if neighbor_state_index > repeat_end_index:
                        continue
                    log_prob = dynamic_table[row][col] + self.transition_matrix[row_index][neighbor_state_index]
                    zero_based_neighbor_index = self.state_to_index[neighbor_state] - repeat_start_index

                    if log_prob - dynamic_table[zero_based_neighbor_index][col] > 1e-10:
                        dynamic_table[zero_based_neighbor_index][col] = log_prob
                        vpath_table_row[zero_based_neighbor_index][col] = row
                        vpath_table_col[zero_based_neighbor_index][col] = col

        # Back tracking viterbi path from the Prefix Matcher End
        cdef list vpath = []
        vpath.insert(0, (state_count - 1,  self.states[repeat_end_index]))
        row, col = vpath_table_row[state_count-1][sequence_length], vpath_table_col[state_count-1][sequence_length]
        row_index = row + repeat_start_index

        while row != 0 or col != 0:
            vpath.insert(0, (self.state_to_index[self.states[row_index]], self.states[row_index]))
            row, col = vpath_table_row[row][col], vpath_table_col[row][col]
            row_index = row + repeat_start_index

        vpath.insert(0, (self.state_to_index[self.states[row_index]], self.states[row_index]))
        ## TODO
        # Try to match with the previous one and get the score, and update the score
        # cdef double logp = dynamic_table[self.state_to_index[self.subModels[self.n_subModels-1].end]][sequence_length]

        return 0, vpath

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _update_tables_for_subseq(self,
                               int row,
                               int col,
                               int repeat_start_index,
                               int repeat_end_index,
                               char ch,
                               State state,
                               int[::1,:] vpath_table_row,
                               int[::1,:] vpath_table_col,
                               double[::1,:] dynamic_table):

        neighbor_states = self.transition_map[state]
        cdef int neighbor_state_index = 0
        cdef int zero_based_neighbor_index = 0
        cdef double log_prob = 0
        cdef int row_index = repeat_start_index + row

        if state.is_silent():  # Silent state: Stay in the same column
            for neighbor_state in neighbor_states:
                neighbor_state_index = self.state_to_index[neighbor_state]
                if neighbor_state_index > repeat_end_index:
                    continue
                log_prob = dynamic_table[row][col] + self.transition_matrix[row_index][neighbor_state_index]
                zero_based_neighbor_index = neighbor_state_index - repeat_start_index

                if log_prob - dynamic_table[zero_based_neighbor_index][col] > 1e-10:
                    dynamic_table[zero_based_neighbor_index][col] = log_prob
                    vpath_table_row[zero_based_neighbor_index][col] = row
                    vpath_table_col[zero_based_neighbor_index][col] = col
        else:  # Not a silent state: Emit a character and move to the next column
            for neighbor_state in neighbor_states:
                neighbor_state_index = self.state_to_index[neighbor_state]
                if neighbor_state_index > repeat_end_index:
                    continue
                log_prob = dynamic_table[row][col] + self.transition_matrix[row_index][neighbor_state_index] + state.distribution[ch]
                zero_based_neighbor_index = neighbor_state_index - repeat_start_index

                if log_prob - dynamic_table[zero_based_neighbor_index][col + 1] > 1e-10:
                    dynamic_table[zero_based_neighbor_index][col + 1] = log_prob
                    vpath_table_row[zero_based_neighbor_index][col + 1] = row
                    vpath_table_col[zero_based_neighbor_index][col + 1] = col

    cpdef get_encoded_sequence(self, sequence):
        cdef int i, n = len(sequence)
        encoded_seq = [0] * n
        for i in range(n):
            encoded_seq[i] = _BASE_CODE[ord(sequence[i])]
            if encoded_seq[i] < 0:
                raise KeyError(sequence[i])
        return np.array(encoded_seq, dtype=np.intc)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cpdef tuple viterbi(self, sequence, min_threshold=None):
        """
        :param sequence: a sequence
        :param min_threshold: per-call floor OR'd with self.dp_score_threshold via max() --
            never assigned to self (thread-unsafe; see read_selection.py:_decode_one).
        :return: log probability and viterbi path

        Production only -- no counters, no dp_tables, no skip_enabled toggle. That
        instrumentation lives entirely in hmm.hmm_instrumented (test-only,
        decode_instrumented()), a SEPARATE extension built from the same
        _viterbi_fill_core.pxi with DEF INSTRUMENTED = True, so it costs this module
        nothing (Task 3 fix round 1; task-3-report.md).
        """
        if not self.is_baked:
            raise ValueError("ERROR: To call viterbi, the model must have been baked")

        # Initialize dynamic programming table
        # Rows represent states and Columns represent sequence
        cdef int sequence_length = len(sequence)
        cdef int[::1] encoded_sequence = self.get_encoded_sequence(sequence)
        # Score table: fresh every call -- see _thread_scratch's docstring.
        cdef double[::1,:] dynamic_table = np.empty((self.n_states, 2), dtype=np.double, order='F')
        # Per-thread (never self -- read_selection.py) scratch from _thread_scratch,
        # amortising the ~1.56 MB vpath allocation across calls on this thread.
        cdef int[::1,:] vpath_table_row = _thread_scratch(self.n_states, sequence_length + 1)
        cdef int row, col, ch

        # Hoist the flat tables into locals so the DP can run with the GIL released:
        # attribute access on self would need it back.
        cdef int[::1] indptr = self.nbr_indptr
        cdef int[::1] indices = self.nbr_indices
        cdef unsigned char[::1] silent = self.silent
        cdef double[:, ::1] emissions = self.emissions
        cdef double[::1] weights = self.nbr_logp
        cdef double threshold = self.dp_score_threshold if min_threshold is None else max(self.dp_score_threshold, min_threshold)
        cdef int start_index = self.state_to_index[self.start]

        cdef int fill_status = 0
        with nogil:
            # Reused scratch is not guaranteed -inf like a fresh np.full was; reset
            # here, under nogil, before _viterbi_fill's own per-column reset takes over.
            for row in range(self.n_states):
                dynamic_table[row, 0] = -INFINITY
                dynamic_table[row, 1] = -INFINITY
            dynamic_table[start_index, 0] = log(1)
            fill_status = _viterbi_fill(encoded_sequence, dynamic_table, vpath_table_row,
                          indptr, indices, silent, emissions,
                          weights, threshold, sequence_length,
                          start_index, NULL, True)
        if fill_status != 0:
            raise MemoryError('Viterbi work queue could not be grown')

        # For the last update
        col = sequence_length
        cdef int col_phys = col & 1  # physical index into the rolled table, not the logical column
        state = self.states[self.n_states-2]
        row = self.n_states - 2  # state_to_index[states[row]] == row (bake() asserts it)

        cdef int neighbor_state_index = 0
        cdef double log_prob = 0
        neighbor_indices = self.neighbors[state]
        for neighbor_state_index in neighbor_indices:
            log_prob = dynamic_table[row][col_phys] + self.transition_matrix[row][neighbor_state_index]

            if log_prob - dynamic_table[neighbor_state_index][col_phys] > 1e-10:
                dynamic_table[neighbor_state_index][col_phys] = log_prob
                vpath_table_row[neighbor_state_index][col] = row

        # Back tracking viterbi path from the Prefix Matcher End
        cdef int end_index = self.n_states - 1  # == state_to_index[subModels[-1].end]
        cdef double logp = dynamic_table[end_index][col_phys]
        if logp == -np.inf:  # no path satisfying the threshold
            return logp, [(end_index, self.subModels[self.n_subModels-1].end)]

        # _traceback walks nogil into a malloc'd array; append once + reverse
        # replaces the old O(n^2) front-of-list insertion under the GIL.
        cdef int* path_rows
        cdef int path_len
        with nogil:
            path_len = _traceback(vpath_table_row, silent, end_index, sequence_length, &path_rows)
        if path_len < 0:
            raise MemoryError('Viterbi traceback could not be grown')

        cdef list vpath = []
        for row in range(path_len):
            vpath.append((path_rows[row], self.states[path_rows[row]]))
        free(path_rows)
        vpath.reverse()
        return logp, vpath

    def check_sanity_of_transition_prob(self, verbose):
        for subModel in self.subModels:
            for state in subModel.states:
                print("State {}".format(state.name))
                if abs(sum(subModel.transition_map[state].values()) - 1) > 0.0001:
                    if verbose:
                        print([(key.name, value) for key, value in subModel.transition_map[state].items()])
                    print("Transition prob of {} is not sum up to 1".format(state.name))
                    print("Sum: {}".format(sum(subModel.transition_map[state].values())))


if __name__ == "__main__":
    pass
