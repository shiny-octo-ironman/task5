from tweaks import *
import make
import learn

np.random.seed(42)
N = 2000

# ----------------------------------------------------------- Data generation --

def main():
    transition_matrix = make.transition()
    emission_matrix = make.emission()
    states = make.states(transition_matrix, N)
    values = make.values(states, emission_matrix)

    print "Transition matrix: \n", transition_matrix
    print 
    print "Emission matrix: \n", log(emission_matrix)
    print
    print "States: ", states
    print
    print "Values: ", values + 1
    print
    print "States expected fraction: ", learn.ergodic_distribution(transition_matrix)
    print "States real fraction: ", learn.state_fractions(states)
    print

    # -------------------------------------------------------- Viterbi and F.-B. --

    states_vi = learn.states_viterbi(values, transition_matrix, emission_matrix)
    states_fb = learn.states_forward_backward(values, transition_matrix, emission_matrix)

    print "-----------------------------------------------------------------------"
    print "Viterbi states: ", states_vi
    print
    print "Viterbi accuracy: ", mean(states_vi == states)
    print
    print "Fwd-Back states: ", states_fb
    print
    print "Fwd-Back accuracy: ", mean(states_fb == states)
    
main()
