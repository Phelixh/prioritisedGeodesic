import numpy as np
import matplotlib.pyplot as plt

from graph import DiGraph
from geodesic_agent import GeodesicAgent, SftmxGeodesicAgent
from RL_utils import softmax

np.random.seed(865612)

### Convenience functions


def action_seq(goal, arm_length, start_to_choice):
    """
    Given maze parameters, given the full sequence of actions that leads from the start state to the goal state.

    Args:
        goal ():
        arm_length ():
        start_to_choice ():

    Returns:

    """
    actions = []

    # Get to choice point
    for _ in range(start_to_choice - 1):
        actions.append(1)

    # Get to goal arm
    actions.append(goal + 2)

    # Continue to the end
    for _ in range(arm_length - 1):
        actions.append(1)

    return actions

### Main script


## Set parameters
# Geometry
num_arms = 2
arm_length = 2
start_to_choice = 2

## Build underlying MDP
# Basics
num_states = num_arms * arm_length + start_to_choice
num_actions = 2 + num_arms

# Fill out edges
edges = np.zeros((num_states, num_actions))
for state in range(num_states):  # Convenient way of filling in null actions
    edges[state, :] = state

# Overwrite null actions with actual transitions
for state in range(start_to_choice):
    if not state == 0:
        edges[state, 0] = state - 1  # Go left

    if not state == start_to_choice - 1:
        edges[state, 1] = state + 1  # Go right

# Add choice transitions
for action in range(2, num_actions):
    edges[start_to_choice - 1, action] = start_to_choice + (action - 2) * arm_length

# Add within-arm transitions
for arm in range(num_arms):
    for rel_state in range(arm_length):
        state = start_to_choice + arm * arm_length + rel_state

        if not rel_state == 0:
            edges[state, 0] = state - 1  # Backwards through arm

        if not rel_state == arm_length - 1:
            edges[state, 1] = state + 1  # Forwards through arm


# Task
num_sessions = 100
num_trials = 20

# Build graph object
s0_dist = np.zeros(num_states)
s0_dist[0] = 1
carey_maze = DiGraph(num_states, edges, init_state_dist=s0_dist)
T = carey_maze.transitions
all_experiences = carey_maze.get_all_transitions()

# Define MDP-related parameters
goal_states = start_to_choice + np.arange(num_arms) * arm_length + (arm_length - 1)

## Build agent
behav_lr = 1
behav_temp = 0.4

num_replay_steps = 3
replay_lr = 1 / num_trials
replay_temp = 0.8

decay_rate = 0.75
noise = 0.00  # Update noise

alpha = 0.9
policy_temperature = 0.3
use_softmax = True
replay_mode = 'full'

## Run task

# Build storage variables
choices = np.zeros((num_sessions, num_trials))
rewards = np.zeros((num_sessions, num_trials))
sess_seq = np.zeros((num_sessions, num_trials))

postoutcome_replays = np.zeros((num_sessions, num_trials, num_replay_steps, 3)) - 1   # so obvious if some row is unfilled
states_visited = np.zeros((num_sessions, num_trials, start_to_choice + arm_length))
posto_Gs = np.zeros((num_sessions, num_trials, num_states, num_actions, num_states))

# Agent parameters
behav_goal_vals = np.zeros(num_arms)
replay_goal_vals = np.zeros(num_arms)

# Instantiate agent
if not use_softmax:
    ga = GeodesicAgent(num_states, num_actions, goal_states, alpha=alpha, goal_dist=None, s0_dist=s0_dist, T=T)
else:
    ga = SftmxGeodesicAgent(num_states, num_actions, goal_states, alpha=alpha, goal_dist=None, s0_dist=s0_dist, T=T,
                            policy_temperature=policy_temperature)

ga.remember(all_experiences)

# Simulate
sess_type = 0
for session in range(num_sessions):
    # Set basic task variables
    rvec = np.zeros(num_states)
    if sess_type == 0:
        rvec[goal_states[0]] = 1.5
        rvec[goal_states[1]] = 1
    else:
        rvec[goal_states[1]] = 1.5
        rvec[goal_states[0]] = 1

    for trial in range(num_trials):
        if trial % 50 == 0:
            print('session %d, trial %d' % (session, trial))

        ga.curr_state = 0
        sess_seq[session, trial] = sess_type

        # Choose an arm
        chosen_arm = np.random.choice(num_arms, p=softmax(behav_goal_vals, behav_temp))
        choices[session, trial] = chosen_arm

        # Go to it
        act_seq = action_seq(chosen_arm, arm_length, start_to_choice)
        for adx, action in enumerate(act_seq):
            next_state, reward = carey_maze.step(ga.curr_state, action=action, reward_vector=rvec)
            ga.basic_learn((ga.curr_state, action, next_state), decay_rate=decay_rate, noise=noise)  # Update GR

            ga.curr_state = next_state
            states_visited[session, trial, adx + 1] = ga.curr_state

            # Check for rewards
            if adx == len(act_seq) - 1:
                behav_goal_vals[chosen_arm] += behav_lr * (reward - behav_goal_vals[chosen_arm])
                replay_goal_vals[chosen_arm] += replay_lr * (reward - replay_goal_vals[chosen_arm])
                rewards[session, trial] = reward

        # Post-outcome replay
        posto_Gs[session, trial, :] = ga.G
        replays, _, _ = ga.replay(num_replay_steps, goal_dist=softmax(replay_goal_vals, replay_temp), verbose=True,
                                  check_convergence=False, prospective=True, EVB_mode=replay_mode)
        postoutcome_replays[session, trial, :, :] = replays

# Save everything
np.savez('./Data/carey_session_data.npz', posto=postoutcome_replays, state_trajs=states_visited,
         choices=choices, rewards=rewards, sess_seq=sess_seq, posto_Gs=posto_Gs, allow_pickle=True)

# Visualize
vis = False
if vis:
    plt.figure()
    for i in range(num_trials):
        if rewards[i] == 1:
            plt.scatter(i, choices[i], marker='x', color='r')
        else:
            plt.scatter(i, choices[i], marker='.', color='b')

print('success')
