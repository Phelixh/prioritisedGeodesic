import numpy as np


def dynamics_policy_onestep(policy, T):
	"""
	Given a dynamics matrix T[i, a, j] = P(state j | state i, action a) and
	a policy policy[i, a] = P(action a | state i), return the one-step transition
	matrix one_step_T[i, j] = P(state j | state i).
	"""
	num_states = policy.shape[0]
	num_actions = T.shape[1]
	one_step_T = np.zeros((num_states, num_states))

	for i in range(num_states):
		for j in range(num_states):
			for a in range(num_actions):
				one_step_T[i, j] += policy[i, a] * T[i, a, j]

	return one_step_T


def compute_occupancy(policy, T, gamma=0.95):
	"""
		Compute future state occupancy matrix given a policy `policy`
		and transition dynamics matrix `T`
	"""
	# Convert dynamics + policy to one-step transition matrix
	one_step_T = dynamics_policy_onestep(policy, T)

	# Compute resultant future occupancy matrix and evaluate
	num_states = policy.shape[0]
	M = np.linalg.inv(np.eye(num_states) - gamma * one_step_T)

	return M


def oned_twod(state, width, height):
	"""
		Given a state with 1d coordinate `state`, transform it into a
		2d coordinate in a box with width `width` and height `height`
	"""

	row, col = np.unravel_index(state, (height, width))
	return row, col


def is_malformed_policy(policy):
	"""
		Check if any rows don't add up to 1.
	"""

	for row in range(policy.shape[0]):
		if abs(np.sum(policy[row, :]) - 1) > 1e-4:
			return True

	return False


def softmax(vals, temperature):
	"""
	Compute a probability distribution over actions using the softmax rule.

	Args:
		vals (array-like): Array of Q-values.
		temperature (float): Softmax temperature.

	Returns:
		Action probabilities.
	"""

	nums = np.exp(vals / temperature)
	return nums / np.sum(nums)
