from tweaks import *

def ergodic_distribution(transition_matrix):
    initial_pos = [1] + [0] * (N_STATES - 1)
    return dot(initial_pos, linalg.matrix_power(transition_matrix, 100))
    
def state_fractions(states):
    N = len(states)
    return [sum(array(states) == state) * 1.0 / N for state in range(N_STATES)]
    
def hmm(values):
    pass
    
def states_viterbi(values, transition_matrix, emission_matrix):
    pass

def states_forward_backward(values, transition_matrix, emission_matrix):
    pass

   