import numpy as np
import matplotlib.pyplot as plt

from gridworld import Arena
from geodesic_agent import GeodesicAgent

np.random.seed(865612)

# Physics
width = 10
height = 7
num_states = width * height

num_sims = 100
replay_steps = [5, 10, 20, 40, 60, 80]

uni_times = np.zeros((num_sims, len(replay_steps)))
pri_times = np.zeros((num_sims, len(replay_steps)))
for rdx, num_replay_steps in enumerate(replay_steps):
    for sim in range(num_sims):
        print(num_replay_steps, sim)

        # Build object
        num_start_states = 5
        several_start_states = np.zeros(num_states)
        several_start_states[np.random.choice(num_states, size=num_start_states, replace=False)] = 1 / num_start_states
        init_state_dist = several_start_states

        arena = Arena(width, height, init_state_distribution=init_state_dist)
        all_experiences = arena.get_all_transitions()
        T = arena.transitions

        ## Agent parameters
        corner_goals = np.array([width - 1, (height - 1) * width, height * width - 1]) # Non-start corners
        all_goals = np.arange(0, width * height)
        goals = corner_goals

        alpha = 1.0
        gamma = 0.95

        # Set up prioritized agent
        ga_pri = GeodesicAgent(arena.num_states, arena.num_actions, goals, T, alpha=alpha, gamma=gamma,
                           s0_dist=init_state_dist)
        ga_pri.curr_state = 0
        ga_pri.remember(all_experiences)  # Preload our agent with all possible memories

        # Set up uniform agent
        ga_uni = GeodesicAgent(arena.num_states, arena.num_actions, goals, T, alpha=alpha, gamma=gamma,
                           s0_dist=init_state_dist)
        ga_uni.curr_state = 0
        ga_uni.remember(all_experiences)  # Preload our agent with all possible memories

        ## Run replay
        # ga_pri.replay(num_steps=num_replay_steps, verbose=False, prospective=True)
        ga_uni.uniform_replay(num_replay_steps)

        ## Run behaviour
        # Pick a goal at random
        active_goal = np.random.choice(corner_goals)

        # Simulate the agents on the MDP and see if they get there
        trajectory_length = 1000
        state_seqs_pri, _, _ = arena.sample_trajectory(trajectory_length, policy=ga_pri.policies[active_goal])
        state_seqs_uni, _, _ = arena.sample_trajectory(trajectory_length, policy=ga_uni.policies[active_goal])

        try:
            pri_time = np.argwhere(state_seqs_pri == active_goal)[0][0]
            uni_time = np.argwhere(state_seqs_uni == active_goal)[0][0]

            pri_times[sim, rdx] = pri_time
            uni_times[sim, rdx] = uni_time
        except:
            pri_times[sim, rdx] = np.nan
            uni_times[sim, rdx] = np.nan

# np.savez('learning_curve_2.npz', pri_times=pri_times, uni_times=uni_times)

# plt.figure()
# plt.plot(replay_steps, np.nanmean(pri_times, axis=0), label='prioritized')
# plt.plot(replay_steps, np.nanmean(uni_times, axis=0), label='uniform')
# plt.show()

# Extra learning for uniform
ext_replay_steps = [100, 200, 400, 800]
add_uni_times = np.zeros((num_sims, len(ext_replay_steps)))
for rdx, num_replay_steps in enumerate(ext_replay_steps):
    for sim in range(num_sims):
        print(num_replay_steps, sim)

        # Build object
        num_start_states = 5
        several_start_states = np.zeros(num_states)
        several_start_states[np.random.choice(num_states, size=num_start_states, replace=False)] = 1 / num_start_states
        init_state_dist = several_start_states

        arena = Arena(width, height, init_state_distribution=init_state_dist)
        all_experiences = arena.get_all_transitions()
        T = arena.transitions

        ## Agent parameters
        corner_goals = np.array([width - 1, (height - 1) * width, height * width - 1]) # Non-start corners
        all_goals = np.arange(0, width * height)
        goals = corner_goals

        alpha = 1.0
        gamma = 0.95

        # Set up uniform agent
        ga_uni = GeodesicAgent(arena.num_states, arena.num_actions, goals, T, alpha=alpha, gamma=gamma,
                           s0_dist=init_state_dist)
        ga_uni.curr_state = 0
        ga_uni.remember(all_experiences)  # Preload our agent with all possible memories

        ## Run replay
        ga_uni.uniform_replay(num_replay_steps)

        ## Run behaviour
        # Pick a goal at random
        active_goal = np.random.choice(corner_goals)

        # Simulate the agents on the MDP and see if they get there
        trajectory_length = 1000
        state_seqs_uni, _, _ = arena.sample_trajectory(trajectory_length, policy=ga_uni.policies[active_goal])

        try:
            uni_time = np.argwhere(state_seqs_uni == active_goal)[0][0]
            add_uni_times[sim, rdx] = uni_time
        except:
            add_uni_times[sim, rdx] = np.nan

uni_times = np.hstack((uni_times, add_uni_times))
plt.figure()
plt.plot(np.mean(uni_times, axis=0))
plt.show()