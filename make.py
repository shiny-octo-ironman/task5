from tweaks import *

# ----------------------------------------------------- Random distributions --

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

# ---------------------------------------------------- Transition & emission --

def transition():
    result = zeros([N_STATES, N_STATES])
    
    # Honest state
    result[0, 0] = TRANSITION_HONEST_INERTIA
    result[0, 1:] = \
        ones(N_STATES - 1) * (1 - TRANSITION_HONEST_INERTIA) / (N_STATES - 1)
    
    # Dishonest states, return to honest
    result[1:, 0] = 1 - TRANSITION_DISHONEST_INERTIA
    
    # Dishonest states, rest
    for s in range(1, N_STATES):
        # Dishonest self-inertia
        result[s, s] = N_STATES > 2 and TRANSITION_DISHONEST_SELF_INERTIA or 1.0
        
        # Transition to other dishonest states
        if N_STATES > 2:
            indices = range(1, s) + range(s + 1, N_STATES)
            
            distribution = random_distribution_nonzeros(
                N_STATES - 2, 
                TRANSITION_DISHONEST_MAX, 
                TRANSITION_MIN_ENTROPY
            )
            
            result[s, indices] = \
                distribution * (1 - TRANSITION_DISHONEST_SELF_INERTIA)
        
        # Multiply by dishonest inertia (rest is transition to honest state)
        result[s, 1:] *= TRANSITION_DISHONEST_INERTIA
        
    return result

def emission():
    result = zeros([N_STATES, N_VALUES])
    
    # Honest emission
    result[0] = ones(N_VALUES) / N_VALUES
    
    # Dishonest emissions
    for state in range(1, N_STATES):
        result[state] = random_distribution_nonzeros(
            N_VALUES, 
            EMISSION_DISHONEST_MAX, 
            EMISSION_MIN_ENTROPY
        )
        
    return result
    
# ----------------------------------------------------------------- Modeling --
    
def states(transition_matrix, sample_length):
    state = 0
    result = [0] * sample_length
    
    for i in range(sample_length):
        result[i] = state
        state = random(transition_matrix[state])
        
    return result
    
def values(states, emission_matrix):
    return [random(emission_matrix[state]) for state in states]