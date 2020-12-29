#cython: boundscheck=False
#cython: cdivision=True

from collections import defaultdict
from operator import attrgetter

from .base cimport DiscreteDistribution
from .base cimport State

import numpy as np
cimport numpy as np
from libc.math cimport log

cimport cython



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
    # 2D matrix (After conforming the topology, create one matrix for visualization)
    cdef object transition_matrix
    cdef dict state_to_index

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
        # 2D matrix (After conforming the topology, create one matrix for visualization)
        self.transition_matrix = None
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

    def bake(self, read_length=None, merge=None, sort_by_name=False):
        """
        In a model, start state comes the first and end state comes the last.
        Other states are in the middle, and they are sorted by their name.
        e.g.)
        start - I0 - D1 - M1, I1 - D2 - M2 - I2, ... D10 - M10 - I10 - end

        setting start_index and end_index

        setting connections between subModels
        """
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

        self.is_baked = True

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

        # print("repeat unit number taret {}".format(repeat_unit_number))
        # for i in range(repeat_start_index, repeat_end_index+1):
        #     print(self.states[i].name)
        # print("all repeat unit states")

        # Initialize dynamic programming table
        # Rows represent states and Columns represent sequence
        cdef int sequence_length = len(sequence)
        cdef int state_count = repeat_end_index - repeat_start_index + 1

        cdef double[:,:] dynamic_table = np.ones((state_count, sequence_length + 1), dtype=np.double) * (-np.inf)
        dynamic_table[0][0] = log(1)

        # Storing previous states row and column separately (Naive version)
        cdef int[:,:] vpath_table_row = np.zeros((state_count, sequence_length + 1), dtype=np.intc)
        cdef int[:,:] vpath_table_col = np.zeros((state_count, sequence_length + 1), dtype=np.intc)

        cdef int row, col
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
                if col != 0 and dynamic_table[row][col] == -np.inf:
                    continue
                state = self.states[row_index]
                self._update_tables_for_subseq(row, col, repeat_start_index, repeat_end_index, sequence, state, vpath_table_row, vpath_table_col, dynamic_table)

        # # For the last update
        col = sequence_length
        for row in range(state_count-1):
            row_index = repeat_start_index + row
            if col != 0 and dynamic_table[row][col] == -np.inf:
                continue
            state = self.states[row_index]
            neighbor_states = self.transition_map[state]
            neighbor_state_index = 0
            prob = 0

            if state.is_silent():  # Silent state: Stay in the same column
                for neighbor_state in neighbor_states:
                    neighbor_state_index = self.state_to_index[neighbor_state] - repeat_start_index
                    if neighbor_state_index > repeat_end_index - repeat_start_index:
                        continue
                    prob = dynamic_table[row][col] + log(self.transition_map[state][neighbor_state])

                    if prob - dynamic_table[neighbor_state_index][col] > 1e-10:
                        dynamic_table[neighbor_state_index][col] = prob
                        vpath_table_row[neighbor_state_index][col] = row
                        vpath_table_col[neighbor_state_index][col] = col

        # Back tracking viterbi path from the Prefix Matcher End
        cdef list vpath = []
        vpath.insert(0, (state_count - 1,  self.states[repeat_end_index]))
        row, col = vpath_table_row[state_count-1][sequence_length], vpath_table_col[state_count-1][sequence_length]
        row_index = row + repeat_start_index

        # print(row, col)
        while row != 0 or col != 0:
            # print(row, col)
            # print(self.states[row_index].name)
            vpath.insert(0, (self.state_to_index[self.states[row_index]], self.states[row_index]))
            row, col = vpath_table_row[row][col], vpath_table_col[row][col]
            row_index = row + repeat_start_index

        vpath.insert(0, (self.state_to_index[self.states[row_index]], self.states[row_index]))
        # cdef double logp = dynamic_table[self.state_to_index[self.subModels[self.n_subModels-1].end]][sequence_length]

        return 0, vpath

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _update_tables_for_subseq(self,
                               int row,
                               int col,
                               int repeat_start_index,
                               int repeat_end_index,
                               char* sequence,
                               State state,
                               int[:,:] vpath_table_row,
                               int[:,:] vpath_table_col,
                               double[:,:] dynamic_table):

        neighbor_states = self.transition_map[state]
        cdef int neighbor_state_index = 0
        cdef double prob = 0

        if state.is_silent():  # Silent state: Stay in the same column
            for neighbor_state in neighbor_states:
                neighbor_state_index = self.state_to_index[neighbor_state] - repeat_start_index
                if neighbor_state_index > repeat_end_index - repeat_start_index:
                    continue
                prob = dynamic_table[row][col] + log(self.transition_map[state][neighbor_state])

                if prob - dynamic_table[neighbor_state_index][col] > 1e-10:
                    dynamic_table[neighbor_state_index][col] = prob
                    vpath_table_row[neighbor_state_index][col] = row
                    vpath_table_col[neighbor_state_index][col] = col
        else:  # Not a silent state: Emit a character and move to the next column
            for neighbor_state in neighbor_states:
                neighbor_state_index = self.state_to_index[neighbor_state] - repeat_start_index
                if neighbor_state_index > repeat_end_index - repeat_start_index:
                    continue
                prob = dynamic_table[row][col] + log(self.transition_map[state][neighbor_state]) + \
                       log(state.distribution[sequence[col]])

                if prob - dynamic_table[neighbor_state_index][col + 1] > 1e-10:
                    dynamic_table[neighbor_state_index][col + 1] = prob
                    vpath_table_row[neighbor_state_index][col + 1] = row
                    vpath_table_col[neighbor_state_index][col + 1] = col


    @cython.boundscheck(False)
    cpdef tuple viterbi(self, sequence):
        """
        :param sequence: a sequence
        :return: log probability and viterbi path
        """
        if not self.is_baked:
            print("ERROR: To call viterbi, the model must have been baked")
            raise ValueError

        # Find start and end index of repeats matcher
        cdef Model repeat_matcher_model = self.subModels[1]
        cdef int repeat_start_index = self.state_to_index[repeat_matcher_model.start]
        cdef int repeat_end_index = self.state_to_index[repeat_matcher_model.end]

        # Initialize dynamic programming table
        # Rows represent states and Columns represent sequence
        cdef int sequence_length = len(sequence)
        # self.dynamic_table = np.ones((self.n_states, sequence_length + 1), dtype=np.double) * (-np.inf)
        # self.dynamic_table[self.state_to_index[self.start]][0] = np.log(1)

        cdef double[:,:] dynamic_table = np.ones((self.n_states, sequence_length + 1), dtype=np.double) * (-np.inf)
        dynamic_table[self.state_to_index[self.start]][0] = log(1)

        # Storing previous states row and column separately (Naive version)
        cdef int[:,:] vpath_table_row = np.zeros((self.n_states, sequence_length + 1), dtype=np.intc)
        cdef int[:,:] vpath_table_col = np.zeros((self.n_states, sequence_length + 1), dtype=np.intc)

        cdef int row, col
        for col in range(sequence_length):
            # Filling out suffix matcher table once
            for row in range(0, repeat_start_index):
                if col != 0 and dynamic_table[row][col] == -np.inf:
                    continue
                state = self.states[row]
                self._update_dynamic_table(row, col, sequence, state, vpath_table_row, vpath_table_col, dynamic_table)

            # Filling out repeat matcher table twice
            for iteration in range(2):
                for row in range(repeat_start_index, repeat_end_index+1):
                    state = self.states[row]
                    self._update_dynamic_table(row, col, sequence, state, vpath_table_row, vpath_table_col, dynamic_table)

            # Filling out prefix matcher table once
            for row in range(repeat_end_index+1, len(self.states)):
                state = self.states[row]
                self._update_dynamic_table(row, col, sequence, state, vpath_table_row, vpath_table_col, dynamic_table)

        # For the last update
        col = sequence_length
        state = self.states[self.n_states-2]
        row = self.state_to_index[state]

        cdef int neighbor_state_index = 0
        cdef double prob = 0
        for neighbor_state in self.transition_map[state]:
            neighbor_state_index = self.state_to_index[neighbor_state]
            prob = dynamic_table[row][col] + log(self.transition_map[state][neighbor_state])

            if prob - dynamic_table[neighbor_state_index][col] > 1e-10:
                dynamic_table[neighbor_state_index][col] = prob
                # update path table
                vpath_table_row[neighbor_state_index][col] = row
                vpath_table_col[neighbor_state_index][col] = col

        # Back tracking viterbi path from the Prefix Matcher End
        cdef list vpath = []
        cdef int end_index = self.state_to_index[self.subModels[self.n_subModels-1].end]

        vpath.insert(0, (end_index, self.subModels[self.n_subModels-1].end))
        row, col = vpath_table_row[end_index][sequence_length], vpath_table_col[end_index][sequence_length]

        while row != 0 or col != 0:
            vpath.insert(0, (self.state_to_index[self.states[row]], self.states[row]))
            row, col = vpath_table_row[row][col], vpath_table_col[row][col]
        vpath.insert(0, (self.state_to_index[self.states[row]], self.states[row]))
        cdef double logp = dynamic_table[self.state_to_index[self.subModels[self.n_subModels-1].end]][sequence_length]

        return logp, vpath

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _update_dynamic_table(self,
                               int row,
                               int col,
                               char* sequence,
                               State state,
                               int[:,:] vpath_table_row,
                               int[:,:] vpath_table_col,
                               double[:,:] dynamic_table):

        neighbor_states = self.transition_map[state]
        cdef int neighbor_state_index = 0
        cdef double prob = 0

        if state.is_silent():  # Silent state: Stay in the same column
            for neighbor_state in neighbor_states:
                neighbor_state_index = self.state_to_index[neighbor_state]
                prob = dynamic_table[row][col] + log(self.transition_map[state][neighbor_state])

                if prob - dynamic_table[neighbor_state_index][col] > 1e-10:
                    dynamic_table[neighbor_state_index][col] = prob
                    vpath_table_row[neighbor_state_index][col] = row
                    vpath_table_col[neighbor_state_index][col] = col
        else:  # Not a silent state: Emit a character and move to the next column
            for neighbor_state in neighbor_states:
                neighbor_state_index = self.state_to_index[neighbor_state]
                prob = dynamic_table[row][col] + log(self.transition_map[state][neighbor_state]) + \
                       log(state.distribution[sequence[col]])

                if prob - dynamic_table[neighbor_state_index][col + 1] > 1e-10:
                    dynamic_table[neighbor_state_index][col + 1] = prob
                    vpath_table_row[neighbor_state_index][col + 1] = row
                    vpath_table_col[neighbor_state_index][col + 1] = col

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