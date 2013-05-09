import tweaks as _
import numpy as np
from numpy import *
import make
import check
from pprint import pprint

# -------------------------------------------------------------------- Set up --

np.random.seed(_.MASTER_SEED)
set_printoptions(precision=4, suppress=True)

# ----------------------------------------------------------- Run experiments --

seeds = make.experiment_seeds()
for seed in seeds:
    perf, data = check.everything(seed)
    pprint(perf)
