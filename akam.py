import numpy as np
import matplotlib.pyplot as plt

from gridworld import GridWorld
from geodesic_agent import GeodesicAgent

# Reproducibility
np.random.seed(865612)

# Some flags
draw_arena = False
save = True

## Setup
# Basic parameters
width = 10
height = 10
num_states = width * height

# Create walls
p_wall = 0.5
walls = []
for i in range(num_states):
    if np.random.uniform() < p_wall:
        action = np.random.choice(4)

        # Implement direct wall
        walls.append((i, action))

        # Implement reverse wall
        if action == 0 and not i % width == 0:  # blocking right
            walls.append((i - 1, 2))
        elif action == 1 and i - width >= 0:  # blocking down
            walls.append((i - width, 3))
        elif action == 2 and not i % width == width - 1:  # blocking left
            walls.append((i + 1, 0))
        elif action == 3 and i + width < num_states:  # blocking up
            walls.append((i + width, 1))

# Add extra walls to probe bottlenecking
for i in range(height):
    if i == 5:
        continue

    walls.append((3 + (i * width), 2))
    walls.append((4 + (i * width), 0))

# Manually remove some walls
walls.remove((92, 2))
walls.remove((93, 0))

# Build Arena
arena = GridWorld(width, height, walls=walls)
if draw_arena:
    ax = arena.draw()
    plt.show()

# Simulation parameters
init_state_dist = np.zeros(num_states)
init_state_dist[0] = 1  # Top-left
# init_state_dist = np.ones(num_states) / num_states

# Goals
num_goals = 7
possible_goals = np.random.choice(num_states, size=7, replace=False)
possible_goals = np.append(possible_goals, width - 1)
init_goal_dist = np.zeros(num_states)
init_goal_dist[possible_goals[0]] = 1
all_goals = np.arange(num_states)

p_goal_stay = 0.90
goal_transition_mat = np.eye(num_states)
for i in range(num_goals):
    goal = possible_goals[i]
    goal_transition_mat[goal, possible_goals] = (1 - p_goal_stay) / (num_goals - 1)
    goal_transition_mat[goal, goal] = p_goal_stay

# Agent parameters
alpha = 0.30  # Learning rate
gamma = 0.95  # Temporal discounting
num_replay_steps = 50

## Run replay
# Build agent
all_experiences = arena.get_all_transitions()
T = arena.transitions

ga = GeodesicAgent(num_states, arena.num_actions, all_goals, T, alpha=alpha, gamma=gamma, s0_dist=init_state_dist)
ga.curr_state = 0
ga.remember(all_experiences)  # Preload our agent with all possible memories

# Replay
check_convergence = False
conv_thresh = 1e-8
replayed_exps, stats_for_nerds, backups = ga.dynamic_replay(num_replay_steps, goal_transition_mat,
                                                            init_goal_dist, prospective=True, verbose=True,
                                                            check_convergence=check_convergence, otol=conv_thresh)
state_needs, transition_needs, gains, all_MEVBs, all_DEVBs = stats_for_nerds

if save:
    np.savez('Data/akam_onestart_0.30.npz',
             replay_seqs=replayed_exps, state_needs=state_needs, gains=gains, all_MEVBs=all_MEVBs,
             trans_needs=transition_needs, all_DEVBs=all_DEVBs,
             backups=backups, num_states=num_states, alpha=alpha, gamma=gamma, memories=all_experiences,
             walls=walls, width=width, height=height, arena=arena, possible_goals=possible_goals)

