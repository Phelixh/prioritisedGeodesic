from geodesic_agent import GeodesicAgent
from gridworld import GridWorld
import time

import numpy as np
import matplotlib.pyplot as plt

def add_global_epsilon_noise(policy, epsilon):
	'''
		Semi-dumb function that takes any deterministic policy and returns 
		a modification of that policy with epsilon exploratory noise.
	'''
	eps_policy = np.zeros_like(policy)
	for state in range(eps_policy.shape[0]):
		best_actions = np.flatnonzero(eps_policy[state, :] == np.max(eps_policy[state, :]))
		num_best_actions = len(best_actions) # Split 1 - epsilon ties equally

		eps_policy[state, :] = epsilon / eps_policy.shape[1]
		eps_policy[state, best_actions] += (1 - epsilon) / num_best_actions # Deterministic for epsilon = 0
		
	return eps_policy

def add_local_epsilon_noise(policy, epsilon):
	'''
		Semi-dumb function that takes any deterministic policy and returns 
		a modification of that policy with epsilon exploratory noise.
	'''
	eps_policy = np.zeros_like(policy)
	best_actions = np.flatnonzero(eps_policy == np.max(eps_policy))
	num_best_actions = len(best_actions) # Split 1 - epsilon ties equally

	eps_policy += epsilon / len(eps_policy)
	eps_policy[best_actions] += (1 - epsilon) / num_best_actions # Deterministic for epsilon = 0
		
	return eps_policy


np.random.seed(5) # Reproducibility, because I'm a good scientist, I guess

# Grid parameters
width = 9
height = 6
num_states = width * height
num_actions = 4

stoch = 0.0 # Grid stochasticity

init_state_distribution = np.zeros(num_states)
init_state_distribution[0] = 1

reward_vector = np.zeros(num_states)
reward_vector[num_states - 1] = 1

term_states = [] # States that start terminal
additional_walls = [] # User-defined barriers

# Agent parameters
alpha = 0.3
gamma = 0.95
epsilon = 0.05
goal_states = [width - 1, (height - 1) * width, num_states - 1] # Top right, bottom left, bottom right
replay_steps = [0, 5, 10, 15, 20]

# Simulation parameters
num_expts = 30
num_burnin_steps = 2000
num_traj = 30
num_steps = 75
uniform_policy = np.ones((num_states, num_actions)) / num_actions
replay_modulus = 100

# Storage
agents = np.empty((num_expts, len(replay_steps)), dtype=object)
goal_seqs = np.zeros((num_expts, len(replay_steps))) - 1
state_seqs = np.zeros((num_traj, num_steps + 1, num_expts, len(replay_steps))) - 1
burnin_state_seqs = np.zeros((num_burnin_steps + 1, num_expts, len(replay_steps))) - 1
action_seqs = np.zeros((num_traj, num_steps, num_expts, len(replay_steps)))
reward_seqs = np.zeros((num_traj, num_steps, num_expts, len(replay_steps)))
replay_seqs = [np.empty((num_traj, num_expts, replay_step), dtype=object) for replay_step in replay_steps[1:]]

# Simulate
for rdx, replay_step in enumerate(replay_steps):
	for expt in range(num_expts):
		# Get GridWorld and associated objects
		gw = GridWorld(width, height, stoch, additional_walls, term_states, init_state_distribution)
		all_experiences = gw.get_all_transitions()
		T = gw.transitions

		# Set up agent
		geo_agent = GeodesicAgent(num_states, num_actions, goal_states, T, alpha=alpha, gamma=gamma)
		geo_agent.remember(all_experiences) # Pre-load our agent with all possible memories
		goal = np.random.choice(goal_states)
		goal_seqs[expt, rdx] = goal

		# Begin with online learning step before goal is activated
		# This is a single trajectory that lasts many steps and doesn't terminate
		# Agent behaves as if possessing no brain and just diffuses uniformly
		print('starting burn in')
		curr_state = gw.sample_initial_state()
		burnin_state_seqs[0, expt, rdx] = curr_state
		for t in range(num_burnin_steps):
			# Behave
			geo_agent.curr_state = curr_state
			action, next_state, reward = gw.step(curr_state, policy=uniform_policy[curr_state, :], 
													 reward_vector=reward_vector)

			# Learn
			experience = (curr_state, action, next_state)
			geo_agent.learn([experience])
			geo_agent.remember([experience])
			for goal_state in goal_states:
				geo_agent.update_state_policy(curr_state, goal_state, epsilon=0, set_policy=True)

			# Update
			curr_state = next_state
			burnin_state_seqs[t + 1, expt, rdx] = next_state

			if t % replay_modulus == 0 and replay_step > 0: # Only replay every replay_modulus steps
				geo_agent.replay(replay_step)

		print('burnin burned in')
		# Start performance test with many short episodes
		# Agent behaves as if possessing brain, navigates directly to goal if possible
		gw.set_terminal([goal])
		for traj in range(num_traj):
			print(rdx, expt, traj)

			# Get initial state
			curr_state = gw.sample_initial_state()
			state_seqs[traj, 0, expt, rdx] = curr_state

			# Start trajectory
			for t in range(0, num_steps):
				# Behave
				geo_agent.curr_state = curr_state

				# Goal is active, pursue as if possessing brain
				eps_policy = add_local_epsilon_noise(geo_agent.policies[goal][curr_state, :], epsilon)
				action, next_state, reward = gw.step(curr_state, policy=eps_policy, 
													 reward_vector=reward_vector)

				# Store
				state_seqs[traj, t + 1, expt, rdx] = next_state
				action_seqs[traj, t, expt, rdx] = action
				reward_seqs[traj, t, expt, rdx] = reward

				# Learn
				experience = (curr_state, action, next_state)
				geo_agent.learn([experience])
				geo_agent.remember([experience])
				for goal_state in goal_states:
					geo_agent.update_state_policy(curr_state, goal_state, epsilon=0, set_policy=True)

				# Update
				curr_state = next_state
			
			# Replay
			if replay_step != 0:
				replay_seqs[rdx - 1][traj, expt, :] = geo_agent.replay(replay_step, return_seq=True)

		agents[expt, rdx] = geo_agent

# Save all data
savez_dict = {'agents' : agents,
			  'state_seqs' : state_seqs,
			  'action_seqs' : action_seqs,
			  'reward_seqs' : reward_seqs,
			  'goal_seqs' : goal_seqs,
			  'replay_step_nos' : replay_steps}

for rdx, replay_seq in enumerate(replay_seqs):
	savez_dict['replay_seq_%d' % rdx] = replay_seq

np.savez('geodesic_grid_data.npz', **savez_dict)
