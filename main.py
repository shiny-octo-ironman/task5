from tweaks import *
import make
import learn

transition_matrix = make.transition()
emission_matrix = make.emission()
states = make.states(transition_matrix, 100)
values = make.values(states, emission_matrix)

print "Transition matrix: \n", transition_matrix
print 
print "Emission matrix: \n", emission_matrix
print
print "States: ", states
print
print "Values: ", values
print
print "States exp. fraction: ", learn.ergodic_distribution(transition_matrix)
print "States real fraction: ", learn.state_fractions(states)