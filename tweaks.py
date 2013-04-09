from scipy.stats import *
import numpy as np
from numpy import *

# There is 1 "honest" state (index 0) and a few "dishonest" state (indices 1, 
# 2, ...). "Honest" state emits all values with equal probabilities, and is the
# initial state.

N_STATES = 1 + 2 # one "honest" state plus a few "dishonest" states
N_VALUES = 6     # six outcomes of a dice roll

# When generating transition or emission probabilities, generate values, not
# defined by "inertia" tweaks below in such a manner, that entropy of resulting 
# probability distribution will be at least X times less, than entropy of a 
# uniform distribution
TRANSITION_MIN_ENTROPY = 1.2
EMISSION_MIN_ENTROPY   = 1.2

# In dishonest states, only K emission weights will be non-zero, and only M
# transition weights will be non-zero (not counting transition to honest state).
# In a sense, it's a "sparsity" tweak.
TRANSITION_DISHONEST_MAX = N_STATES - 1
EMISSION_DISHONEST_MAX   = N_VALUES     # smaller value will make dishonesty 
                                        # easily detectable by segmentation 
                                        # algorithm :)

# Probability of transition from honest state to itself.
TRANSITION_HONEST_INERTIA = 0.9

# Probability of transition from dishonest state to some other dishonest state 
# (including itself). Not quite the same thing as TRANSITION_HONEST_INERTIA:
# 1 - TRANSITION_DISHONEST_INERTIA is the probability of a return to "honest" 
# state.
TRANSITION_DISHONEST_INERTIA = 0.9

# TRANSITION_DISHONEST_INERTIA * TRANSITION_DISHONEST_SELF_INERTIA will be
# the probability of transition from dishonest state to itself
TRANSITION_DISHONEST_SELF_INERTIA = 0.8