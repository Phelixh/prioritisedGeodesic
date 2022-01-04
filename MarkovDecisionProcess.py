import numpy as np
np.set_printoptions(precision=3, suppress=True)

class MarkovDecisionProcess(object):
	'''
		The MarkovDecisionProcess class allows agents to interact with an environment.
		Users should provide the environment properties:
			* State-action transition matrix `transitions`
			* Action set [0, 1, ..., num_actions - 1]
			* Initial state distribution `init_state_distribution

		A reward vector is not necessary, but may be provided at each of the interaction 
		functions in order to add a reward component. Rewards will be assigned on the
		basis of arrival to a state.
	'''

	def __init__(self, transitions, num_actions, init_state_distribution=None):
		'''
			Initialize MarkovDecisionProcess object with environment parameters.
		'''
		self.num_states = transitions.shape[0]
		self.num_actions = num_actions
		self.transitions = transitions

		if init_state_distribution is None:
			self.s0_dist = np.ones(self.num_states) / self.num_states
		else:
			self.s0_dist = init_state_distribution

	def sample_trajectories(self, num_trajectories, length, policy, reward_vector=None):
		'''
			Sample `num_trajectories` trajectories of length `length`, using policy `policy`.
		'''
		state_seqs = np.zeros((num_trajectories, length), dtype=int)
		action_seqs = np.zeros((num_trajectories, length), dtype=int) # last action is meaningless
		reward_seqs = np.zeros((num_trajectories, length))

		for i in range(num_trajectories):
			state_seq, action_seq, reward_seq = self.sample_trajectory(length, policy, reward_vector)
			state_seqs[i, :] = state_seq
			action_seqs[i, :] = action_seq
			reward_seqs[i, :] = reward_seq

		return state_seqs, action_seqs, reward_seqs

	def sample_trajectory(self, length, policy, reward_vector=None):
		'''
			Sample 1 trajectory of length `length`, using policy `policy`.
		'''
		state_seq = np.zeros(length, dtype=int)
		action_seq = np.zeros(length, dtype=int)
		reward_seq = np.zeros(length)

		# Sample initial state
		state_seq[0] = self.sample_initial_state()

		# Sample subsequent states
		for t in range(1, length + 1):
			action, next_state, reward = self.execute_policy(state_seq[t - 1], policy, reward_vector)

			if t < length:
				state_seq[t] = next_state

			action_seq[t - 1] = action
			reward_seq[t - 1] = reward

		return state_seq, action_seq, reward_seq

	def sample_initial_state(self):
		'''
			Sample the initial state distribution to pick an initial state.
		'''
		return np.random.choice(self.num_states, p=self.s0_dist)

	def step(self, state, policy=None, action=None, reward_vector=None):
		assert(policy is not None or action is not None), 'Buddy, you gotta give me at least an action or a policy.'

		if reward_vector is None:
			reward_vector = np.zeros(self.num_states)

		if policy is not None:
			return self.execute_policy(state, policy, reward_vector)

		if action is not None:
			return self.perform_action(state, action, reward_vector)

	def perform_action(self, state, action, reward_vector=None):
		'''
			Perform action `action` in state `state`, and observe the resultant state and reward.
		'''
		next_state = np.random.choice(self.num_states, p=self.transitions[state, action, :])
		if reward_vector is not None:
			reward = reward_vector[next_state]
		else:
			reward = 0

		return next_state, reward

	def execute_policy(self, state, policy, reward_vector=None):
		'''
			Execute policy `policy` in state `state` and return the resultant action, successor state
			and reward.
		'''
		action = np.random.choice(self.num_actions, p=policy)
		next_state, reward = self.perform_action(state, action, reward_vector)
		return action, next_state, reward

####### Testing script
if __name__ == '__main__':
	num_states = 4  # 0: top left, 1 : top right, 2: bottom left, 3: bottom right
	num_actions = 4 # 0: left, 1: up, 2: right, 3: down
	
	# Set up transition matrix as deterministic
	T = np.zeros((num_states, num_actions, num_states))
	T[0, 0, :] = [1, 0, 0, 0]
	T[0, 1, :] = T[0, 0, :]
	T[0, 2, :] = [0, 1, 0, 0]
	T[0, 3, :] = [0, 0, 1, 0]

	T[1, 0, :] = [1, 0, 0, 0]
	T[1, 1, :] = [0, 1, 0, 0]
	T[1, 2, :] = T[1, 1, :]
	T[1, 3, :] = [0, 0, 0, 1]

	T[2, 0, :] = [0, 0, 1, 0]
	T[2, 1, :] = [1, 0, 0, 0]
	T[2, 2, :] = [0, 0, 0, 1]
	T[2, 3, :] = T[2, 0, :]

	T[3, 0, :] = [0, 0, 1, 0]
	T[3, 1, :] = [0, 1, 0, 0]
	T[3, 2, :] = [0, 0, 0, 1]
	T[3, 3, :] = [0, 0, 0, 1]

	# Does the init work?
	mdp = MarkovDecisionProcess(T, num_actions)

	# Can we sample trajectories?
	policy = (1 / num_actions) * np.ones((num_states, num_actions))
	state_seqs, action_seqs, reward_seqs = mdp.sample_trajectories(10, 15, policy, reward_vector = np.array([1, 0, 1, 1]))

	print(state_seqs)
	print(action_seqs)
	print(reward_seqs)

