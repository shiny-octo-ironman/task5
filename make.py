import tweaks as _
import numpy as np
from numpy import *

# ------------------------------------------------------ Random distributions --

def p_logp(p):
    if p < 1e-8: 
        return 0
    return p * log(p)

def entropy(probabilities):
    return sum(-p_logp(p) for p in probabilities)
    
def entropy_uniform(N):
    p = 1.0 / N
    return -p_logp(p) * N

def random_distribution(N, min_entropy):
    if N == 1:
        return array([1.0])

    distribution = np.random.random_sample(N)
    distribution /= sum(distribution)
    
    while entropy_uniform(N) / entropy(distribution) < min_entropy:
        distribution[np.random.randint(N)] *= 1.1
        distribution /= sum(distribution)
        
    return distribution

def random_distribution_nonzeros(N, N_nonzeros, min_entropy):
    result = zeros(N)
    non_zeros = np.random.choice(N, min(N, N_nonzeros), replace=False)
    result[non_zeros] = random_distribution(N, min_entropy)
    return result
    
def random(probabilities):
    N = len(probabilities)
    bins = add.accumulate(probabilities)
    values = range(N)
    return values[digitize([np.random.rand()], bins)]    

# ----------------------------------------------------- Transition & emission --

def transition():
    result = zeros([_.N_STATES, _.N_STATES])
    
    # Honest state
    result[0, 0] = _.TRANSITION_HONEST_INERTIA
    result[0, 1:] = \
        ones(_.N_STATES - 1) * (1 - _.TRANSITION_HONEST_INERTIA) / (_.N_STATES - 1)
    
    # Dishonest states, return to honest
    result[1:, 0] = 1 - _.TRANSITION_DISHONEST_INERTIA
    
    # Dishonest states, rest
    for s in range(1, _.N_STATES):
        # Dishonest self-inertia
        result[s, s] = _.N_STATES > 2 and _.TRANSITION_DISHONEST_SELF_INERTIA or 1.0
        
        # Transition to other dishonest states
        if _.N_STATES > 2:
            indices = range(1, s) + range(s + 1, _.N_STATES)
            
            distribution = random_distribution_nonzeros(
                _.N_STATES - 2, 
                _.TRANSITION_DISHONEST_MAX, 
                _.TRANSITION_MIN_ENTROPY
            )
            
            result[s, indices] = \
                distribution * (1 - _.TRANSITION_DISHONEST_SELF_INERTIA)
        
        # Multiply by dishonest inertia (rest is transition to honest state)
        result[s, 1:] *= _.TRANSITION_DISHONEST_INERTIA
        
    return result

def emission():
    result = zeros([_.N_STATES, _.N_VALUES])
    
    # Honest emission
    result[0] = ones(_.N_VALUES) / _.N_VALUES
    
    # Dishonest emissions
    for state in range(1, _.N_STATES):
        result[state] = random_distribution_nonzeros(
            _.N_VALUES, 
            _.EMISSION_DISHONEST_MAX, 
            _.EMISSION_MIN_ENTROPY
        )
        
    return result
    
# ------------------------------------------------------------------ Modeling --
    
def states(transition_matrix, sample_length):
    state = 0
    result = [0] * sample_length
    
    for i in range(sample_length):
        result[i] = state
        state = random(transition_matrix[state])
        
    return array(result, dtype=int)
    
def values(states, emission_matrix):
    return array([random(emission_matrix[state]) for state in states], dtype=int)
    
# -------------------------------------------------- Reproducible experiments --

def experiment_seeds():
    return np.random.randint(0, 1000, _.K)
