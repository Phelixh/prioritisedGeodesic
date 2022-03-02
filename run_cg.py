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
save = True

# Physics
num_nbrhds = 3
num_nbrs = 5
num_states = num_nbrhds * num_nbrs

# Different initial state distributions
one_source = np.zeros(num_states)
one_source[1] = 1
some_sources = np.zeros(num_states)
some_sources[1] = 1/3
some_sources[6] = 1/3
some_sources[11] = 1/3
all_sources = np.ones(num_states) / num_states

init_state_dists = [one_source, some_sources, all_sources]

# Different goal state distributions
all_goals = np.arange(num_states)  # All states are goals
one_goal = np.array([7])           # One state is a goal
some_goals = np.array([2, 7, 12])  # One state per neighbourhood is a goal

goal_states = [one_goal, some_goals, all_goals]

## Agent parameters
alpha = 1.0   # Learning rate
gamma = 0.95  # Temporal discounting
num_replay_steps = 500

# Run simulations
for i in range(len(init_state_dists)):
    for j in range(len(goal_states)):
        # Logging
        print(i, j)

        # Build object
        cg = CommunityGraph(num_nbrhds, num_nbrs, init_state_dists[i])
        all_experiences = cg.get_all_transitions()
        T = cg.transitions

        # Set up agent
        ga = GeodesicAgent(cg.num_states, cg.num_actions, goal_states[j], T, alpha=alpha, gamma=gamma,
                           s0_dist=init_state_dists[i])
        ga.curr_state = 0
        ga.remember(all_experiences)  # Preload our agent with all possible memories

        ## Run replay
        check_convergence = False
        conv_thresh = 1e-8
        replayed_exps, stats_for_nerds, backups = ga.replay(num_steps=num_replay_steps, verbose=True, prospective=True,
                                                            check_convergence=check_convergence, convergence_thresh=conv_thresh)
        needs, trans_needs, gains, all_MEVBs = stats_for_nerds

        # Save
        if save:
            np.savez('Data/cg_%dc_%dn_%d_%d.npz' % (num_nbrs, num_nbrhds, i, j), replay_seqs=replayed_exps, needs=needs,
                     gains=gains, all_MEVBs=all_MEVBs, trans_needs=trans_needs, backups=backups, num_nbrhds=num_nbrhds,
                     num_nbrs=num_nbrs, num_states=num_states, alpha=alpha, gamma=gamma, memories=all_experiences)
