import tweaks as _
from numpy import *
import numpy as np
    
# ------------------------------------------------------- State distributions --

def ergodic_distribution(transition_matrix):
    initial_pos = [1] + [0] * (_.N_STATES - 1)
    return dot(initial_pos, linalg.matrix_power(transition_matrix, 100))
    
def state_fractions(states):
    return array([sum(array(states) == s) * 1.0 / _.N for s in range(_.N_STATES)])
    
def value_fractions(values):
    return array([sum(array(values) == v) * 1.0 / _.N for v in range(_.N_VALUES)])
    
# -------------------------------------------------------- Viterbi (not cool) --
    
def states_viterbi(values, transition_matrix, emission_matrix):
    best_next = zeros([_.N, _.N_STATES], dtype=int)
    weights = zeros([_.N, _.N_STATES])
    path = zeros(_.N, dtype=int)

    # Compute best weight for each state s in each position i: -log(P), where
    # P is the best probability, with which we can emit values[i:] if at 
    # position i we're in state s
    weights[_.N-1] = log(emission_matrix[:, values[_.N-1]])
    for i in reversed(range(_.N-1)):
        for from_ in range(_.N_STATES):
        
            # Initial weight
            weights[i][from_] = -_.INF
            
            # Try all transitions
            for to in range(_.N_STATES):
                new_weight = weights[i+1][to] + log(transition_matrix[from_][to])
                
                if new_weight > weights[i][from_]:
                    weights[i][from_] = new_weight
                    best_next[i][from_] = to
                
        # Add emission cost
        weights[i] += log(emission_matrix[:, values[i]])
        
    # Select best first state
    first_state = argmax(weights[0:1])
    if _.USE_APRIORI:
        first_state = 0
        
    # Restore path
    state = first_state
    for i in range(_.N):
        path[i] = state
        state = best_next[i][state]
        
    return path
       
# ----------------------------------------------------- Forward-Backward (ok) --
    
def w_sum(w, axis=0):
    # Infer shape for maximum weight, so it could correctly broadcast
    shape = list(w.shape)
    shape[axis] = 1
    
    # Use maximum weight as common multiplier
    w_max = np.max(w, axis=axis)
    return w_max + log(sum(exp(w - w_max.reshape(shape)), axis=axis))
    
def w_sum2(w1, w2):
    return w_sum(concatenate((w1[newaxis,...], w2[newaxis,...])))
    
def w_norm(w, axis=0):
    shape = list(w.shape)
    shape[axis] = 1
    return w - w_sum(w, axis).reshape(shape)
    
def p_norm(p, axis=0):
    shape = list(p.shape)
    shape[axis] = 1
    return p / sum(p, axis).reshape(shape)
        
def tail_weights(values, T, E, last_column):       
    W = zeros([_.N, _.N_STATES]) # [W]eights
    
    W[_.N-1] = last_column + E[:, values[_.N - 1]]
    for i in reversed(range(_.N - 1)):
        emission_cost = tile(E[:, values[i]], [_.N_STATES, 1]).transpose()
        transition_cost = T
        target_cost = tile(W[i + 1], [_.N_STATES, 1])
                
        cost = emission_cost + transition_cost + target_cost
        W[i] = w_sum(cost, axis=1)
        
    return W
        
def head_tail_weights(values, T, E):
    # Fill state weights at first moment of time (first_column)
    # and at last moment (last_column)
    last_column = zeros(_.N_STATES)
    first_column = -_.INF * ones(_.N_STATES)
    first_column[0] = 0

    # Do dynamic programming to get absolute state weights
    tail_W = tail_weights(values, T, E, last_column)
    head_W = tail_weights(values[::-1], transpose(T), E, first_column)

    return head_W[::-1], tail_W
        
def likelihood(head_W):
    return w_sum(head_W[-1])
        
def posteriors(values, E, head_W, tail_W):    
    # Get absolute state weights
    W = head_W + tail_W - E[:, values].transpose()
    
    # Normalize to get posterior weights
    return W - likelihood(head_W)
    
def states_forward_backward(values, transition_matrix, emission_matrix):
    E = log(emission_matrix)   # [E]mission matrix
    T = log(transition_matrix) # [T]ransition matrix
    
    # Get absolute weights of generating head or tail of the sequence
    head_W, tail_W = head_tail_weights(values, T, E)    
    
    # Restore states -- get argmax(posteriors)
    Wp = posteriors(values, E, head_W, tail_W)
    return argmax(Wp, 1), likelihood(head_W)
    
# ---------------------------------------------------------------- Baum-Welch --

def hmm(values, transition_matrix_init=None, emission_matrix_init=None):
    # Generate initial values
    if transition_matrix_init is None:
        transition_matrix_init = np.random.random([_.N_STATES, _.N_STATES])
        transition_matrix_init = p_norm(transition_matrix_init, 1)
    if emission_matrix_init is None:
        emission_matrix_init = np.random.random([_.N_STATES, _.N_VALUES])
        emission_matrix_init = p_norm(emission_matrix_init, 1)
        
    # Initialize emission and transition matrices
    E = log(emission_matrix_init)   # [E]mission matrix
    T = log(transition_matrix_init) # [T]ransition matrix
    
    last_likelihood = None
    while True:
        # Use apriori information
        if _.USE_APRIORI:
            E[0] = log(ones(_.N_VALUES) / _.N_VALUES)
        
        # Compute absolute weights
        head_W, tail_W = head_tail_weights(values, T, E)
        print 'Likelihood =', likelihood(head_W)
        
        # Gather transition matrix
        partial_T = -_.INF * ones(T.shape)
        for i in range(_.N-1):
            partial_Ti = head_W[i][..., newaxis] + tail_W[i + 1][newaxis, ...] + T
            partial_T = w_sum2(partial_T, partial_Ti)
        
        # Gather emission matrix
        partial_E = -_.INF * ones(E.shape)
        Wp = posteriors(values, E, head_W, tail_W)
        for i in range(_.N):
            partial_E[:, values[i]] = w_sum2(partial_E[:, values[i]], Wp[i])
                    
        # Normalize partial counts with posterior probability
        new_likelihood = likelihood(head_W) 
        partial_T -= new_likelihood
        partial_E -= new_likelihood
        
        if last_likelihood is not None and abs(new_likelihood - last_likelihood) < 1e-3:
            break
        last_likelihood = new_likelihood
        
        # Normalize partial counts to get probabilities
        T = w_norm(partial_T, axis=1)
        E = w_norm(partial_E, axis=1)
        
    return exp(T), exp(E)
