from tweaks import *

# ------------------------------------------------------ State distributions --

def ergodic_distribution(transition_matrix):
    initial_pos = [1] + [0] * (N_STATES - 1)
    return dot(initial_pos, linalg.matrix_power(transition_matrix, 100))
    
def state_fractions(states):
    N = len(states)
    return array([sum(array(states) == s) * 1.0 / N for s in range(N_STATES)])
    
# ----------------------------------------------------------- HMM parameters --
    
def hmm(values):
    pass
    
def states_viterbi(values, transition_matrix, emission_matrix):
    pass

def states_forward_backward(values, transition_matrix, emission_matrix):
    pass

   