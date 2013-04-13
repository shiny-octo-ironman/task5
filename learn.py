from tweaks import *

# ------------------------------------------------------ State distributions --

def ergodic_distribution(transition_matrix):
    initial_pos = [1] + [0] * (N_STATES - 1)
    return dot(initial_pos, linalg.matrix_power(transition_matrix, 100))
    
def state_fractions(states):
    N = len(states)
    return array([sum(array(states) == s) * 1.0 / N for s in range(N_STATES)])
    
# ----------------------------------------------------------- HMM parameters --
    
def states_viterbi(values, transition_matrix, emission_matrix):
    N = len(values)
    best_next = zeros([N, N_STATES], dtype=int)
    weights = zeros([N, N_STATES])
    path = zeros(N, dtype=int)

    # Compute best weight for each state s in each position i: -log(P), where
    # P is the best probability, with which we can emit values[i:] if at 
    # position i we're in state s
    weights[N-1] = log(emission_matrix[:, values[N-1]])
    for i in reversed(range(N-1)):
        for from_ in range(N_STATES):
        
            # Initial weight
            weights[i][from_] = -INF
            
            # Try all transitions
            for to in range(N_STATES):
                new_weight = weights[i+1][to] + log(transition_matrix[from_][to])
                
                if new_weight > weights[i][from_]:
                    weights[i][from_] = new_weight
                    best_next[i][from_] = to
                
        # Add emission cost
        weights[i] += log(emission_matrix[:, values[i]])
        
    # Select best first state
    first_state = argmax(weights[0])
    if USE_APRIORI:
        first_state = 0
        
    # Restore path
    state = first_state
    for i in range(N):
        path[i] = state
        state = best_next[i][state]
        
    return path
    
def posteriors(values, transition_matrix, emission_matrix):
    pass

def states_forward_backward(values, transition_matrix, emission_matrix):
    p = posteriors(values, transition_matrix, emission_matrix)
    return argmax(p, 0)

def hmm(values):
    pass
