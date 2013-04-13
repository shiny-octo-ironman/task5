from tweaks import *
    
# ------------------------------------------------------ State distributions --

def ergodic_distribution(transition_matrix):
    initial_pos = [1] + [0] * (N_STATES - 1)
    return dot(initial_pos, linalg.matrix_power(transition_matrix, 100))
    
def state_fractions(states):
    N = len(states)
    return array([sum(array(states) == s) * 1.0 / N for s in range(N_STATES)])
    
# ------------------------------------------------------- Viterbi (not cool) --
    
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
    first_state = argmax(weights[0:1])
    if USE_APRIORI:
        first_state = 0
        
    # Restore path
    state = first_state
    for i in range(N):
        path[i] = state
        state = best_next[i][state]
        
    return path
       
# ---------------------------------------------------- Forward-Backward (ok) --
    
def tail_weights(values, T, E, last_column):
    N = len(values)             
    W = zeros([N, N_STATES]) # [W]eights
    
    W[N-1] = last_column + E[:, values[N - 1]]
    for i in reversed(range(N-1)):
        emission_cost = tile(E[:, values[i]], [N_STATES, 1]).transpose()
        transition_cost = T
        target_cost = tile(W[i + 1], [N_STATES, 1])        
        cost = emission_cost + transition_cost + target_cost
        W[i] = log(sum(exp(cost), axis=1))
        
    return W
        
def posteriors(values, transition_matrix, emission_matrix):    
    E = log(emission_matrix)   # [E]mission matrix
    T = log(transition_matrix) # [T]ransition matrix
    
    # Fill state weights at first moment of time (first_column)
    # and at last moment (last_column)
    last_column = zeros(N_STATES)
    first_column = zeros(N_STATES)
    if USE_APRIORI:
        first_column = -INF * ones(N_STATES)
        first_column[0] = 0

    # Do dynamic programming to get absolute state weights
    tail_W = tail_weights(values, T, E, last_column)
    head_W = tail_weights(values[::-1], transpose(T), E, first_column)
    W = head_W[::-1] + tail_W - E[:, values].transpose()
   
    # Normalize to get posterior weights
    W -= log(sum(exp(head_W[0])))
    return W
    
def states_forward_backward(values, transition_matrix, emission_matrix):
    P = posteriors(values, transition_matrix, emission_matrix)
    return argmax(P, 1)
    
# --------------------------------------------------------------- Baum-Welch --

def hmm(values):
    pass
