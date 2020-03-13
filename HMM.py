import random
import numpy as np
np.random.seed(420)
from tqdm import tqdm
from HMM_helper import obs_map_reverser


class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0. 
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.
            
            D:          Number of observations.
            
            A:          The transition matrix.
            
            O:          The observation matrix.
            
            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is the invariant distribution
                        for transition matrix A (so at any point, the
                        marginal probability of being in state i is
                        given by A_start[i])
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = np.array(A)
        self.O = np.array(O)
        self.A_start = self.invariant()


    def invariant(self):
        ''' Update the invariant distribution based on self.A '''
        eig_vals, vecs = np.linalg.eig(self.A.T)
        eig_vec = np.real(vecs[:, np.abs(eig_vals - 1.0).argmin()])
        return eig_vec / np.sum(eig_vec)


    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state 
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        '''

        x = np.array(x); pi = self.A_start
        t1, t2 = HMM_table_filler(x, self.A, self.O, func=np.max, 
                                  pi=pi, funcB=np.argmax)
        t2 = t2.astype(int)

        states = [np.argmax(t1[-1, :])]
        for stateInd in range(len(x), 1, -1):
            states.append(t2[stateInd, states[-1]])
        states.reverse()

        return ''.join([str(c) for c in states])


    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.

                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''

        x = np.array(x); pi = self.A_start
        return HMM_table_filler(x, self.A, self.O, func=np.sum, \
                                pi=pi, norm=normalize)



    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.

                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        x = np.array(x); pi = self.A_start
        return HMM_table_filler(x, self.A, self.O, func=np.sum, \
                                pi=pi, norm=normalize, go_backward=True)



    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to D - 1. In other words, a list of
                        lists.

            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to L - 1. In other words, a list of
                        lists.

                        Note that the elements in X line up with those in Y.
        '''

        # Calculate each element of A using the M-step formulas.

        margs = [np.arange(self.L) == np.array(y_seq)[:, None].astype(np.float32) 
                 for y_seq in Y]
        self.O_update(X, margs)

        double_margs = []
        for y_seq in Y:
            d_probs = np.zeros((self.L, self.L))
            for j, k in zip(y_seq[:-1], y_seq[1:]):
                d_probs[j, k] += 1
            double_margs.append(d_probs)
        self.A_update(double_margs)



    def unsupervised_learning(self, X, N_iters, verbose=True):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.

            N_iters:    The number of iterations to train on.
        '''

        range_obj = tqdm(range(N_iters)) if verbose else range(N_iters)
        for _ in range_obj:

            # Expectation
            marg_probs = []
            double_margs = [] # P(y^i, y^{i + 1} | x)
            for x_seq in X:
                alphas = self.forward(x_seq, normalize=True)[1:] # Drop first row
                betas = self.backward(x_seq, normalize=True)[1:] # Drop first row

                # Compute marginals
                probs = alphas * betas
                probs /= np.sum(probs, axis=1)[:, None] # Normalize
                marg_probs.append(probs)

                # Compute double marginals, list of (L, L) matrices where 
                # each matrix is for one sequence, and M_ij is 
                # total transition prob from state i to j summed across the
                # elements of the sequence
                d_probs = np.empty((len(x_seq) - 1, self.L, self.L))
                for i in range(len(x_seq) - 1):
                    mat = np.outer(alphas[i], betas[i + 1]) * \
                                  self.A * self.O[:, x_seq[i+1]].reshape(1, -1)
                    mat /= np.sum(mat)
                    d_probs[i] = mat
                double_margs.append(np.sum(d_probs, axis=0))

            # Maximization
            self.O_update(X, marg_probs)
            self.A_update(double_margs)

    def A_update(self, double_margs):
        double_margs = np.array(double_margs)
        A_counts = np.sum(double_margs, axis=tuple(range(double_margs.ndim - 2)))
        self.A = A_counts / np.sum(A_counts, axis=1)[:, None]
        self.A_start = self.invariant()


    def O_update(self, X, marg_probs):
        O_counts = np.zeros((self.L, self.D))
        for x_seq, probs in zip(X, marg_probs):
            for x, dtbn in zip(x_seq, probs):
                O_counts[:, x] += dtbn
        self.O = O_counts / np.sum(O_counts, axis=1)[:, None]

    def generate_emission(self, M):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random. 

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''

        emission = []
        states = []

        states.append(np.random.choice(self.L, p = self.A_start))
        for i in range(M):
            curr_state = states[-1]
            emission.append(np.random.choice(self.D, p = self.O[curr_state]))
            states.append(np.random.choice(self.L, p = self.A[curr_state]))

        return emission, states[:-1]


    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = sum(alphas[-1])
        return prob


    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum([betas[1][j] * self.A_start[j] * self.O[j][x[0]] \
                    for j in range(self.L)])

        return prob





def from_hmm(hmm, obs_map, syll_map=None, rhyme_dict=None):
    '''Make a shakespeare hmm from an hmm '''
    shake_hmm = ShakespeareHMM(hmm.A, hmm.O, obs_map, syll_map, rhyme_dict)        
    return shake_hmm


class ShakespeareHMM(HiddenMarkovModel):
    '''
    Class implementation of Shakespeare Hidden Markov Models.
    '''
    def __init__(self, A, O, obs_map, syll_map=None, rhyme_dict=None):
        '''
        Initializes a Shakespeare HMM, inherits from HMMs,
        but also store a syll_map and a rhyming pairs dictionary,
        override the typical generate emission function with
        one that tries to enforce iambic pentameter
        '''
        super().__init__(A, O)
        self.syll = syll_map
        self.rhyme = rhyme_dict
        self.ind_to_word = obs_map_reverser(obs_map)

    def to_word(self, emission):
        sentence = ' '.join([self.ind_to_word[i] for i in emission]).capitalize()
        return sentence + ','

    # p(a | b) propto p(b | a) * p(a), p(a) comes from invariant, p(b | a)
    # is just the A matrix at ab.
    def gen_line(self, backward=False, seed_state=None,
                 syll_count=10, get_states=False, as_word=True):
        ''' Get a single line of syll_count syllables '''

        A = self.A
        if backward:
            A = (self.A.T * self.A_start[None, :]); A /= A.sum(axis=1)[:, None]
        pi = A[seed_state] if seed_state else self.A_start 
        emission = []
        states = []
 
        states.append(np.random.choice(self.L, p = pi))
        remain = syll_count
        while remain != 0:
            curr_state = states[-1]
            pred = lambda l: min(l) <= remain or any(np.mod(l, 10) == remain)
            inds = np.array([i for i in range(self.D) if pred(self.syll[i])])
            p = self.O[curr_state, inds]; p /= np.sum(p)
            ind = np.random.choice(inds, p = p)
            emission.append(ind)
            l = self.syll[ind]
            count_decrement = l.min() if l.min() <= remain else remain
            remain -= count_decrement
            states.append(np.random.choice(self.L, p = A[curr_state]))
        states.pop()
        if backward:
            emission = emission[::-1]; states = states[::-1]
 
        emission = emission if not as_word else self.to_word(emission)
        return (emission, states) if get_states else emission
    

    def gen_pair(self):                                                       
        '''                                                                    
        Generate a pair of iambic pentameter lines that rhyme                  
        '''                                                                    
        # Get random line ending with word that rhymes with stuff
        seed1 = np.random.choice(list(self.rhyme.keys()))
        syll1 = self.syll[seed1].min()
        seed_state1 = np.argmax(self.O[:, seed1] * self.A_start)
        line1 = self.gen_line(syll_count=10-syll1, backward=True, 
                              seed_state=seed_state1, as_word=False)
        line1.append(seed1)

        # Get second line using word rhyming with 1st line
        seed2 = np.random.choice(self.rhyme[seed1])
        syll2 = self.syll[seed2].min()
        seed_state2 = np.argmax(self.O[:, seed2] * self.A_start)
        line2 = self.gen_line(syll_count=10-syll2, backward=True, 
                              seed_state=seed_state2, as_word=False)
        line2.append(seed2)
        return (self.to_word(line1), self.to_word(line2))
                                                                               
    def generate_sonnet(self, do_syll=True, do_rhyme=True,\
                        n_lines=14, syll_count=10, quats=3, coups=1):
        ''' Get a sonnet with n_lines lines '''
        if do_syll and do_rhyme:
            assert(self.rhyme is not None and self.syll is not None)
            sonnet1 = np.ravel([self.gen_pair() for _ in range(2*quats + coups)])
            sonnet = [sonnet1[4*i + off] for i in range(quats) \
                                         for off in [0, 2, 1, 3]]
            sonnet += list(sonnet1[4*quats:])
        elif do_syll:
            assert(self.syll is not None)
            sonnet = [self.gen_line(syll_count) for i in range(n_lines)]
        else:
            emissions = [self.generate_emission(syll_count)[0] for i in range(n_lines)]
            sonnet = [self.to_word(e) for e in emissions]
        return '\n'.join(sonnet)








def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learning.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to L - 1. In other words, a list of lists.
                    Note that the elements in X line up with those in Y.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)
    
    # Compute L and D.
    L = len(states)
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM

def unsupervised_HMM(n_states, X, N_iters, verbose=True):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.
        
        N_iters:    The number of iterations to train on.
    '''

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)
    
    # Compute L and D.
    L = n_states
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    random.seed(2020)   
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    random.seed(155)   
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, N_iters, verbose=verbose)

    return HMM



def HMM_table_filler(X, A, O, func, pi=None, funcB=None, norm=False,
                     go_backward = False):
    '''
    Generic algorithm to compute a table of values
    for a hidden markov model.

    Args:
        X : emission sequence
        A : Transition probs matrix
        O : Emission probs matrix
        func : combining function rule to use (max for viterbi, sum for forward)
        pi : Initial state probabilities
        funcB : function to generate lookup table (as in viterbi, when we
                care about sequences as well as probs)
        norm : To normalize after computing each row in t1
        go_backward: Jee what do you think (but also you guys wrote this
                    without numpy so idk maybe I should clarify smh)

    Returns:
        t1 : A table of values where t1[i, j] is the value / probability
            for sequence up to i ending in state j 
        t2 : A lookup table for backtracking the sequence of states 
            (for viterbi)
    '''
    # get some paramaters and check basic things
    k, l = O.shape
    n = X.shape[0]
    assert (A.shape == (k, k))

    if pi is None:
        # Assume equal probability
        pi = np.ones((k)) * 1 / k
    assert (pi.shape == (k,))

    t = np.zeros((n+1, k)) # make a table to fill in values

    # Second table for algorithm's like viterbi
    # where arguments / indices must be saved
    if funcB is not None:
        t2 = np.zeros((n+1, k))


    if go_backward:
        t[n, :] = 1
        for i in range(n)[::-1]:
            for j in range(k):
                newVals = t[i + 1, :] * A[j, :] * O[:, X[i]]
                t[i, j] = func(newVals)
            if norm:
                t[i, :] /= np.sum(t[i, :])
        return t


    t[0, :] = pi
    t[1, :] = pi * O[:, X[0]] # initialize table

    for i in range(1, n):
        for j in range(k):
            newVals = t[i, :] * A[:, j] * O[j, X[i]]
            t[i+1, j] = func(newVals)
            if funcB is not None:
                t2[i+1, j] = funcB(newVals)
        if norm:
            t[i+1, :] /= np.sum(t[i+1, :])

    return (t, t2) if funcB is not None else t

