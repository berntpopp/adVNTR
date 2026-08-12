
#cython: boundscheck=False
#cython: cdivision=True

from collections import defaultdict
from operator import attrgetter

from .base cimport DiscreteDistribution
from .base cimport State

import numpy as np
cimport numpy as np
from libc.math cimport log
from libc.stdlib cimport malloc, realloc, free

cimport cython



@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _viterbi_fill(int[::1] encoded_sequence,
                        double[::1,:] dynamic_table,
                        int[::1,:] vpath_table_row,
                        int[::1,:] vpath_table_col,
                        int[::1] indptr,
                        int[::1] indices,
                        unsigned char[::1] silent,
                        double[:, ::1] emissions,
                        double[::1] weights,
                        double threshold,
                        int sequence_length,
                        int start_index) nogil:
    """Fill the Viterbi DP table. Pure C -- runs with the GIL released.

    This is the body that used to live inline in Model.viterbi plus
    Model.__update_dynamic_table. It touches no Python object: the graph arrives as
    CSR (`indptr`/`indices`), emissions as a flat (state, base) table, and silence as
    a byte flag, all built once in bake(). Relaxation order is unchanged -- neighbours
    are still visited in the ascending order bake() sorted them into -- so the
    resulting logp and path are bit-identical to the pre-nogil version.
    """
    cdef int row, col, k, ch, neighbor_state_index, next_col, start, end
    cdef double log_prob, emission, current

    # Two array-backed FIFOs replacing the linked-list Queue, whose queue_push_tail
    # malloc'd a 24-byte node per relaxation (~1.5M malloc/free pairs per call).
    # Push at tail, pop at head, never wrap: byte-for-byte the same visit ORDER the
    # linked list produced. Order is load-bearing -- the relaxation guard is
    # `> 1e-10`, not `> 0`, so this is not an order-independent fixpoint and a LIFO
    # would silently change vpath.
    cdef int cur_cap = 4096, nxt_cap = 4096
    cdef int cur_head = 0, cur_tail = 0, nxt_head = 0, nxt_tail = 0
    cdef int* cur = <int*> malloc(cur_cap * sizeof(int))
    cdef int* nxt = <int*> malloc(nxt_cap * sizeof(int))
    cdef int* swap_buf
    cdef int* grown
    cdef int swap_int

    if cur == NULL or nxt == NULL:
        free(cur)
        free(nxt)
        return -1

    nxt[0] = start_index
    nxt_tail = 1

    for col in range(sequence_length):
        swap_buf = cur; cur = nxt; nxt = swap_buf
        swap_int = cur_cap; cur_cap = nxt_cap; nxt_cap = swap_int
        cur_head = 0
        cur_tail = nxt_tail
        nxt_head = 0
        nxt_tail = 0

        if cur_head == cur_tail:
            break

        ch = encoded_sequence[col]
        next_col = col + 1

        while cur_head < cur_tail:
            row = cur[cur_head]
            cur_head += 1
            start = indptr[row]
            end = indptr[row + 1]
            current = dynamic_table[row, col]

            if silent[row]:  # Silent state: stay in the same column
                for k in range(start, end):
                    neighbor_state_index = indices[k]
                    log_prob = current + weights[k]

                    if log_prob - dynamic_table[neighbor_state_index, col] > 1e-10 and log_prob >= threshold:
                        if cur_tail == cur_cap:
                            grown = <int*> realloc(cur, 2 * cur_cap * sizeof(int))
                            if grown == NULL:
                                free(cur)
                                free(nxt)
                                return -1
                            cur = grown
                            cur_cap *= 2
                        cur[cur_tail] = neighbor_state_index
                        cur_tail += 1
                        dynamic_table[neighbor_state_index, col] = log_prob
                        vpath_table_row[neighbor_state_index, col] = row
                        vpath_table_col[neighbor_state_index, col] = col
            else:  # Emitting state: consume a character and advance a column
                emission = emissions[row, ch]
                for k in range(start, end):
                    neighbor_state_index = indices[k]
                    log_prob = current + weights[k] + emission

                    if log_prob - dynamic_table[neighbor_state_index, next_col] > 1e-10 and log_prob >= threshold:
                        if nxt_tail == nxt_cap:
                            grown = <int*> realloc(nxt, 2 * nxt_cap * sizeof(int))
                            if grown == NULL:
                                free(cur)
                                free(nxt)
                                return -1
                            nxt = grown
                            nxt_cap *= 2
                        nxt[nxt_tail] = neighbor_state_index
                        nxt_tail += 1
                        dynamic_table[neighbor_state_index, next_col] = log_prob
                        vpath_table_row[neighbor_state_index, next_col] = row
                        vpath_table_col[neighbor_state_index, next_col] = col

    free(cur)
    free(nxt)
    return 0


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
    cdef dict state_to_index

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
        # 2D matrix (After conforming the topology, create one matrix for visualization)
        # self.transition_matrix = NULL
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

    def log_probability(self, seq):
        T = len(seq)
        N = self.state_count() - 2 # exclude the first two states (model-start and model-end)
        prob_mat = np.zeros((N,2))
        for n in range(N):
            state = self.states[n]
            prob_mat[n,0] = self.transition_map[self.start][state] * state.distribution[seq[0]]
        for t in range(1,T):
            prob_mat[:,t%2] = 0.0
            for n in range(N):
                state = self.states[n]
                for n_prev in range(N):
                    state_prev = self.states[n_prev]
                    prob_mat[n,t%2] += prob_mat[n_prev,(t-1)%2] * self.transition_map[state_prev][state]
                prob_mat[n,t%2] *= state.distribution[seq[t]]
        for n in range(N):
            state = self.states[n]
            prob_mat[n,(T-1)%2] *= self.transition_map[state][self.end]
        prob = sum(prob_mat[:,(T-1)%2])
        return np.log(prob)

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

            # indices = {subModel.states[i]: i for i in range(subModel.n_states)}
            # subModel.start_index = indices[subModel.start]
            # subModel.end_index = indices[subModel.end]

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

        # self.neighboring_state_indices = np.zeros((self.n_states, self.n_states), dtype=np.int)
        # for from_state in transition_map.keys():
        #     outgoing_states = transition_map[from_state]
        #     for to_state in outgoing_states:
        #         from_index = self.state_to_index[from_state]
        #         to_index = self.state_to_index[to_state]
        #         self.neighboring_state_indices[from_index][to_index] = 1


        # Find start and end index of repeats matcher
        # if len(self.subModels) > 1:
        #     repeat_matcher_model = self.subModels[1]
        #     self.repeat_start_index = self.state_to_index[repeat_matcher_model.start]
        #     self.repeat_end_index = self.state_to_index[repeat_matcher_model.end]

        self.is_baked = True

    def transition_matrix_view(self):
        """Read-only view of the dense transition matrix. For tests only.

        `transition_matrix` is a private cdef attribute; exposing a numpy view lets the
        CSR invariant tests prove that `nbr_logp` was COPIED from it rather than
        recomputed with log(), without making the matrix itself writable from Python.
        """
        return np.asarray(self.transition_matrix)

    def _sort_states(self):
        """
        Sort states in pre-defined (topology of our hmm model) order.

        State naming rule:
        There should be three types of state except start and end
        1. Insert
        2. Match
        3. Delete
        Insertion states start with I
        Match states start with M
        Delete states start with D

        Format:
        I/M/D[index]_[repeating_unit_index]

        Example:

        total_hmm_start
        --------------------------
        suffix_matcher_hmm_start
        ...
        suffix_matcher_hmm_end
        --------------------------
        Repeating Pattern Matcher HMM Model-start
        --------------------------
        unit_start_1
        I0_1
        D1_1
        M1_1
        I1_1
        ...
        unit_end_1
        --------------------------
        unit_start_2
        I0_2
        D1_2
        M1_2
        I1_2
        ...
        unit_end_2
        ...
        --------------------------
        Repeating Pattern Matcher HMM Model-end
        --------------------------
        prefix_matcher_hmm_start
        ...
        prefix_matcher_hmm_end
        --------------------------
        total_hmm_end

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
                # insert_states.append(state)
            elif state.name.startswith("M"):
                match_states[repeat_unit_id].append(state)
                # match_states.append(state)
            elif state.name.startswith("D"):
                delete_states[repeat_unit_id].append(state)
                # delete_states.append(state)
            else:
                if "_start_" in state.name:
                    dummy_start_states[repeat_unit_id].append(state)
                    # dummy_start_states.append(state)
                if "_end_" in state.name:
                    dummy_end_states[repeat_unit_id].append(state)
                    # dummy_end_states.append(state)
                # assert ("start" in state.name or "end" in state.name), "State type should be in (I, M, D, start, end)"

        for repeat_unit_id, states in insert_states.items():
            states.sort(key=lambda x: int(x.name[1:x.name.find("_")]))

        for repeat_unit_id, states in match_states.items():
            states.sort(key=lambda x: int(x.name[1:x.name.find("_")]))

        for repeat_unit_id, states in delete_states.items():
            states.sort(key=lambda x: int(x.name[1:x.name.find("_")]))
        # insert_states.sort(key=lambda x: int(x.name[1:x.name.find("_")]))
        # match_states.sort(key=lambda x: int(x.name[1:x.name.find("_")]))
        # delete_states.sort(key=lambda x: int(x.name[1:x.name.find("_")]))

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

        # # 1.1 Dummy start state
        # for dummy_start in dummy_start_states:
        #     sorted_states.append(dummy_start)
        #
        # # 2. Insert 0 state (number of repeating units)
        # for i in range(len(dummy_start_states)):
        #     sorted_states.append(insert_states.pop(0))
        #
        # # 3. Delete, Match, Insert states
        # assert (len(match_states) == len(delete_states))
        #
        # for i in range(len(match_states)):
        #     sorted_states.append(delete_states[i])
        #     sorted_states.append(match_states[i])
        #     sorted_states.append(insert_states[i])
        #
        # # 4.0 Dummy end state
        # for dummy_end in dummy_end_states:
        #     sorted_states.append(dummy_end)
        #
        # # 4. End state
        # sorted_states.append(self.end)
        #
        # self.states = sorted_states

    def dense_transition_matrix( self ):
        """Returns the dense transition matrix.

        Parameters
        ----------
        None

        Returns
        -------
        matrix : numpy.ndarray, shape (n_states, n_states)
            A dense transition matrix, containing the probability
            of transitioning from each state to each other state.
        """

        m = len(self.states)
        transition_probabilities = np.zeros( (m, m) )

        for i in range(m):
            state1 = self.states[i]
            for n in range(m):
                state2 = self.states[n]
                if (self.transition_map[state1][state2] > 0.0):
                    transition_probabilities[i, n] = self.transition_map[state1][state2]

        return transition_probabilities

    # @classmethod
    # def from_matrix(cls, transition_probabilities, distributions, starts, ends=None,
    #                 state_names=None, name=None, verbose=False, merge='All' ):
    #     """Create a model from a more standard matrix format.
    #
    #     Take in a 2D matrix of floats of size n by n, which are the transition
    #     probabilities to go from any state to any other state. May also take in
    #     a list of length n representing the names of these nodes, and a model
    #     name. Must provide the matrix, and a list of size n representing the
    #     distribution you wish to use for that state, a list of size n indicating
    #     the probability of starting in a state, and a list of size n indicating
    #     the probability of ending in a state.
    #
    #     Parameters
    #     ----------
    #     transition_probabilities : array-like, shape (n_normal_states, n_normal_states)
    #         The probabilities of each state transitioning to each other state.
    #
    #     distributions : array-like, shape (n_normal_states)
    #         The distributions for each state. Silent states are indicated by
    #         using None instead of a distribution object.
    #
    #     starts : array-like, shape (n_normal_states)
    #         The probabilities of starting in each of the states.
    #
    #     ends : array-like, shape (n_normal_states), optional
    #         If passed in, the probabilities of ending in each of the states.
    #         If ends is None, then assumes the model has no explicit end
    #         state. Default is None.
    #
    #     state_names : array-like, shape (n_normal_states), optional
    #         The name of the states. If None is passed in, default names are
    #         generated. Default is None
    #
    #     name : str, optional
    #         The name of the model. Default is None
    #
    #     verbose : bool, optional
    #         The verbose parameter for the underlying bake method. Default is False.
    #
    #     merge : 'None', 'Partial', 'All', optional
    #         The merge parameter for the underlying bake method. Default is All
    #
    #     Returns
    #     -------
    #     model : Model
    #         The baked model ready to go.
    #
    #     Examples
    #     --------
    #     matrix = [ [ 0.4, 0.5 ], [ 0.4, 0.5 ] ]
    #     distributions = [NormalDistribution(1, .5), NormalDistribution(5, 2)]
    #     starts = [ 1., 0. ]
    #     ends = [ .1., .1 ]
    #     state_names= [ "A", "B" ]
    #
    #     model = Model.from_matrix( matrix, distributions, starts, ends,
    #         state_names, name="test_model" )
    #     """
    #
    #     # Build the initial model
    #     model = Model( name=name )
    #     state_names = state_names or ["s{}".format(i) for i in range(len(distributions))]
    #
    #     # Build state objects for every state with the appropriate distribution
    #     states = [ State( distribution, name=name ) for name, distribution in
    #                zip( state_names, distributions) ]
    #
    #     n = len( states )
    #
    #     # Add all the states to the model
    #     for state in states:
    #         model.add_state( state )
    #
    #     # Connect the start of the model to the appropriate state
    #     for i, prob in enumerate( starts ):
    #         if prob != 0:
    #             model.add_transition( model.start, states[i], prob )
    #
    #     # Connect all states to each other if they have a non-zero probability
    #     for i in range( n ):
    #         for j, prob in enumerate( transition_probabilities[i] ):
    #             if prob != 0.:
    #                 model.add_transition( states[i], states[j], prob )
    #
    #     if ends is not None:
    #         # Connect states to the end of the model if a non-zero probability
    #         for i, prob in enumerate( ends ):
    #             if prob != 0:
    #                 model.add_transition( states[j], model.end, prob )
    #
    #     model.bake()
    #     return model

    def concatenate(self, other, suffix='', prefix='', transition_probability=1.0):
        """Concatenate this model to another model.

        Concatenate this model to another model in such a way that a single
        probability 1 edge is added between self.end and other.start. Rename
        all other states appropriately by adding a suffix or prefix if needed.

        Parameters
        ----------
        other : HiddenMarkovModel
            The other model to concatenate

        suffix : str, optional
            Add the suffix to the end of all state names in the other model.
            Default is ''.

        prefix : str, optional
            Add the prefix to the beginning of all state names in the other
            model. Default is ''.

        transition_probability :
            The transition probability from the last model and the other model

        Returns
        -------
        None
        """
        # other.name = "{}{}{}".format(prefix, other.name, suffix)
        # for state in other.states:
        #     state.name = "{}{}{}".format(prefix, state.name, suffix)

        # self.subModels[self.n_subModels] = other
        self.append_subModel(other)
        # self.subModels.append(other)
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
                # Don't believe partially mapped read (the first and last)
                # if col == 0 and row == 0:
                #     neighbor_states = "all_match_states"
                #     for neighbor_state in neighbor_states:
                #         neighbor_state_index = self.state_to_index[neighbor_state] - repeat_start_index
                #         if neighbor_state_index > repeat_end_index - repeat_start_index:
                #             continue
                #         prob = dynamic_table[row][col] + log(self.transition_map[state][neighbor_state])
                #
                #         if prob - dynamic_table[neighbor_state_index][col] > 1e-10:
                #             dynamic_table[neighbor_state_index][col] = prob
                #             vpath_table_row[neighbor_state_index][col] = row
                #             vpath_table_col[neighbor_state_index][col] = col

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
                    #neighbor_state_index = self.state_to_index[neighbor_state] - repeat_start_index
                    neighbor_state_index = self.state_to_index[neighbor_state]
                    #if neighbor_state_index > repeat_end_index - repeat_start_index:
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
                #neighbor_state_index = self.state_to_index[neighbor_state] - repeat_start_index
                neighbor_state_index = self.state_to_index[neighbor_state]
                #if neighbor_state_index > repeat_end_index - repeat_start_index:
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
                #neighbor_state_index = self.state_to_index[neighbor_state] - repeat_start_index
                neighbor_state_index = self.state_to_index[neighbor_state]
                #if neighbor_state_index > repeat_end_index - repeat_start_index:
                if neighbor_state_index > repeat_end_index:
                    continue
                log_prob = dynamic_table[row][col] + self.transition_matrix[row_index][neighbor_state_index] + state.distribution[ch]
                zero_based_neighbor_index = neighbor_state_index - repeat_start_index

                if log_prob - dynamic_table[zero_based_neighbor_index][col + 1] > 1e-10:
                    dynamic_table[zero_based_neighbor_index][col + 1] = log_prob
                    vpath_table_row[zero_based_neighbor_index][col + 1] = row
                    vpath_table_col[zero_based_neighbor_index][col + 1] = col

    cpdef get_encoded_sequence(self, sequence):
        key_map = {'A':0, 'C':1, 'G':2, 'T':3}
        encoded_seq = [key_map[ch] for ch in sequence]
        return np.array(encoded_seq, dtype=np.intc)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cpdef tuple viterbi(self, sequence):
        """
        :param sequence: a sequence
        :return: log probability and viterbi path
        """
        if not self.is_baked:
            raise ValueError("ERROR: To call viterbi, the model must have been baked")

        # Find start and end index of repeats matcher
        cdef Model repeat_matcher_model = self.subModels[1]
        cdef int repeat_start_index = self.state_to_index[repeat_matcher_model.start]
        cdef int repeat_end_index = self.state_to_index[repeat_matcher_model.end]

        # Initialize dynamic programming table
        # Rows represent states and Columns represent sequence
        cdef int sequence_length = len(sequence)
        cdef int[::1] encoded_sequence = self.get_encoded_sequence(sequence)

        # self.dynamic_table = np.ones((self.n_states, sequence_length + 1), dtype=np.double) * (-np.inf)
        # self.dynamic_table[self.state_to_index[self.start]][0] = np.log(1)

        cdef double[::1,:] dynamic_table = np.full((self.n_states, sequence_length + 1), -np.inf, dtype=np.double, order='F')
        dynamic_table[self.state_to_index[self.start]][0] = log(1)

        # Storing previous states row and column separately (Naive version)
        cdef int[::1,:] vpath_table_row = np.zeros((self.n_states, sequence_length + 1), dtype=np.intc, order='F')
        cdef int[::1,:] vpath_table_col = np.zeros((self.n_states, sequence_length + 1), dtype=np.intc, order='F')

        cdef int row, col
        cdef int ch

        # Hoist the flat tables into locals so the DP can run with the GIL released:
        # attribute access on self would need it back.
        cdef int[::1] indptr = self.nbr_indptr
        cdef int[::1] indices = self.nbr_indices
        cdef unsigned char[::1] silent = self.silent
        cdef double[:, ::1] emissions = self.emissions
        cdef double[::1] weights = self.nbr_logp
        cdef double threshold = self.dp_score_threshold
        cdef int start_index = self.state_to_index[self.start]

        cdef int fill_status = 0
        with nogil:
            fill_status = _viterbi_fill(encoded_sequence, dynamic_table, vpath_table_row,
                          vpath_table_col, indptr, indices, silent, emissions,
                          weights, threshold, sequence_length,
                          start_index)
        if fill_status != 0:
            raise MemoryError('Viterbi work queue could not be grown')


        # For the last update
        col = sequence_length
        state = self.states[self.n_states-2]
        row = self.state_to_index[state]

        cdef int neighbor_state_index = 0
        cdef double log_prob = 0
        neighbor_indices = self.neighbors[state]
        for neighbor_state_index in neighbor_indices:
            log_prob = dynamic_table[row][col] + self.transition_matrix[row][neighbor_state_index]

            if log_prob - dynamic_table[neighbor_state_index][col] > 1e-10:
                dynamic_table[neighbor_state_index][col] = log_prob
                vpath_table_row[neighbor_state_index][col] = row
                vpath_table_col[neighbor_state_index][col] = col

        # Back tracking viterbi path from the Prefix Matcher End
        cdef list vpath = []
        cdef int end_index = self.state_to_index[self.subModels[self.n_subModels-1].end]

        vpath.insert(0, (end_index, self.subModels[self.n_subModels-1].end))
        row, col = vpath_table_row[end_index][sequence_length], vpath_table_col[end_index][sequence_length]

        cdef double logp = dynamic_table[self.state_to_index[self.subModels[self.n_subModels-1].end]][sequence_length]
        if logp == -np.inf:  # no path with satisfying the threshold
            return logp, vpath

        while row != 0 or col != 0:
            vpath.insert(0, (self.state_to_index[self.states[row]], self.states[row]))
            row, col = vpath_table_row[row][col], vpath_table_col[row][col]
        vpath.insert(0, (self.state_to_index[self.states[row]], self.states[row]))
        return logp, vpath

    @cython.boundscheck(False)
    @cython.wraparound(False)


    @cython.boundscheck(False)
    @cython.wraparound(False)

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
