# Have this file available so that I can debug with PyCharm
# Carbon copy of equivalent cell in Replay.ipynb

# Imports

import numpy as np

from geodesic_agent import GeodesicAgent
from graph import CommunityGraph

# Reproducibility
np.random.seed(865612)

## Community graph, a la Anna Schapiro
# Store?
save = False

# Physics
num_nbrhds = 3
num_nbrs = 5
num_states = num_nbrhds * num_nbrs

# Build object
init_state_dist = np.ones(num_states) / num_states
cg = CommunityGraph(num_nbrhds, num_nbrs, init_state_dist)
all_experiences = cg.get_all_transitions()
T = cg.transitions

## Agent parameters
goal_states = np.arange(num_states)
alpha = 1.0
gamma = 0.95
num_replay_steps = 500

# Set up agent
ga = GeodesicAgent(cg.num_states, cg.num_actions, goal_states, T, alpha=alpha, gamma=gamma,
                   s0_dist=init_state_dist)
ga.curr_state = 0
ga.remember(all_experiences)  # Preload our agent with all possible memories

## Run replay
check_convergence = False
conv_thresh = 1e-8
replayed_exps, stats_for_nerds, backups = ga.replay(num_steps=num_replay_steps, verbose=True, prospective=True,
                                                    check_convergence=check_convergence, convergence_thresh=conv_thresh)
needs, gains, all_MEVBs = stats_for_nerds

# Save
if save:
    np.savez('Data/cg_3hd_5rs.npz', replay_seqs=replayed_exps, needs=needs, gains=gains, all_MEVBs=all_MEVBs,
             backups=backups, num_nbrhds=num_nbrhds, num_nbrs=num_nbrs,
             num_states=num_states, alpha=alpha, gamma=gamma)
