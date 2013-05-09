from numpy import * 
import numpy as np
import tweaks as _
import make
import learn

def states_accuracy(correct_states, states):
    return array([mean(correct_states == states)])

def segments_precision(correct_states, states):
    TP = mean(logical_and(states > 0, correct_states > 0))
    FP = mean(logical_and(states > 0, correct_states == 0))
    if FP + TP > 0:
        return array([TP / (FP+TP)])
    return 0
    
def segments_recall(correct_states, states):
    TP = mean(logical_and(states > 0, correct_states > 0))
    FN = mean(logical_and(states == 0, correct_states > 0))
    if FN + TP > 0:
        return array([TP / (FN+TP)])
    return 0
    
def everything(experiment_seed):
    # Generate seeds
    np.random.seed(experiment_seed)
    et_seed, state_seed, value_seed, bw_seed = np.random.randint(0, 1000, 4)

    # Generate transition and emission matrices
    np.random.seed(et_seed)
    T = make.transition()
    E = make.emission()
    
    # Generate states
    np.random.seed(state_seed)
    states = make.states(T, _.N)
    
    # Generate values
    np.random.seed(value_seed)
    values = make.values(states, E)

    # Run Viterbi and Forward-Backward
    states_v  = learn.states_viterbi(values, T, E)
    states_fb, likelihood = learn.states_forward_backward(values, T, E)   
    
    # Gather data information
    T_ent = sum(make.entropy(T[state]) for state in range(_.N_STATES))
    E_ent = sum(make.entropy(E[state]) for state in range(_.N_STATES))
    st_expected = learn.ergodic_distribution(T)
    st_real = learn.state_fractions(states)
    val_real = learn.value_fractions(values)

    return ({
        'd_summary': {
            'T_ent':       array([T_ent]),
            'E_ent':       array([E_ent]),
            'likelihood':  array([likelihood]),
            'st_expected': st_expected,
            'st_real':     st_real,
            'val_real':    val_real
        },
    
        'v': {
            'acc':  states_accuracy(states, states_v),
            'prec': segments_precision(states, states_v),
            'rec':  segments_recall(states, states_v)
        },
        
        'fb': {        
            'acc':  states_accuracy(states, states_fb),
            'prec': segments_precision(states, states_fb),
            'rec':  segments_recall(states, states_fb)
        },
        
        #'bw': {
        #    'T_kl':       None,
        #    'E_kl':       None,
        #    'likelihood': None,
        #    'iterations': None,
        #            
        #    'fb': {
        #        'acc':  None,
        #        'prec': None,
        #        'rec':  None
        #    }
        #} 
    },
    {
        'T': T,
        'E': E,
        'states': states,
        'values': values
    })
