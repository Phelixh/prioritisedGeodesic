import numpy as np
from RL_utils import dynamics_policy_onestep
import time
import math

# TODO: think about subtlety of setting EVB to 0 when s_kp == goal for n-step backups

class GeodesicAgent(object):
	'''
		The GeodesicAgent class solves MDPs using the Geodesic Representation (GR),
		which is a control version of the Successor Representation (SR). Unlike the SR,
		which develops a matrix of future state occupancy under a given policy, the GR
		develops a matrix G[i, a, j] = gamma^d(i, a, j), where gamma is a discounting
		factor and d(i, a, j) is the length of the shortest path from i to j after taking 
		action a.

		Unlike the SR, the GR can be used to solve navigation problems in the face of 
		dynamic or uncertain environments.
	'''

	def __init__(self, num_states, num_actions, goal_states, T, goal_dist=None, 
				 s0_dist=None, alpha=0.3, gamma=0.95, min_gain=0):
		# MDP properties
		self.num_states = num_states
		self.num_actions = num_actions
		self.goal_states = goal_states
		self.curr_state = -1 # Pre-initial state indicating action has not yet started
		if goal_dist is None:
			goal_dist = np.ones(len(goal_states)) / len(goal_states)
		self.goal_dist = goal_dist
		self.T = T
		self.s0_dist = s0_dist

		# Agent properties
		self.alpha = alpha # Learning rate
		self.gamma = gamma
		self.min_gain = min_gain
		self.G = np.zeros((num_states, num_actions, num_states)) # Geodesic representation matrix is gamma^shortest_path
		self.memory = [] # Memory bank for later replay

		uniform_policy = np.ones((num_states, num_actions)) / num_actions
		self.policies = { goal_states[i] : uniform_policy.copy() for i in range(len(goal_states)) } # Separate policies for each goal state
																				   					# Initialised each as uniform

	def derive_policy(self, goal_state, G=None, set_policy=False, epsilon=0):
		''' 
			Derive the policy for reaching a given goal state. Since
			the GR represents the shortest (expected) paths, we can 
			simply take the max at every state.

			Allow an epsilon-greedy addition to facilitate exploration.
		'''

		# Allow arbitrary G to be fed in, default to current G
		if G is None:
			G = self.G

		policy = np.zeros((self.num_states, self.num_actions))

		# Compute policy
		for state in range(self.num_states):
			best_actions = np.flatnonzero(G[state, :, goal_state] == np.max(G[state, :, goal_state]))
			num_best_actions = len(best_actions) # Split 1 - epsilon ties equally

			policy[state, :] = epsilon / self.num_actions
			policy[state, best_actions] += (1 - epsilon) / num_best_actions # Deterministic for epsilon = 0
			
		# Cache if wanted
		if set_policy:
			self.policies[goal_state] = policy

		return policy

	def update_state_policy(self, state, goal_state, G=None, set_policy=False, epsilon=0):
		''' 
			Update the internal policy only for state `state`. 

			Allow an epsilon-greedy addition to facilitate exploration.
		'''

		# Allow arbitrary G to be fed in, default to current G
		if G is None:
			G = self.G

		# Compute policy
		best_actions = np.flatnonzero(G[state, :, goal_state] == np.max(G[state, :, goal_state]))
		num_best_actions = len(best_actions) # Split 1 - epsilon ties equally

		if not set_policy: # Only re-copy the whole thing if we're not planning on saving it.
			policy = self.policies[goal_state].copy()
		else:
			policy = self.policies[goal_state]

		policy[state, :] = epsilon / self.num_actions # Deterministic for epsilon = 0
		policy[state, best_actions] += (1 - epsilon) / num_best_actions

		# Cache if wanted
		if set_policy:
			self.policies[goal_state] = policy

		return policy

	def learn(self, transitions, update_policies=True):
		'''
			Batch update internal G matrix with set of transitions.
		'''
		updates = self.compute_Geodesic_update(transitions)
		for tdx, transition in enumerate(transitions):
			s, a, _ = transition
			self.G[s, a, :] += updates[tdx]

			if update_policies:
				for goal_state in self.goal_states:
					self.update_state_policy(s, goal_state, set_policy=True)

	def multi_learn(self, replay_seq, update_policies=True):
		'''
			Update internal G matrix for a sequential set of transitions that are assumed
			to be optimal. The last entry in `curr_replay_seq` is taken to be the most
			recent (and reference) transition from which the GR error is computed. Using
			eligibility traces, this error is then applied in reverse to every previous 
			transition in the sequence.
		'''
		dG = self.compute_trace_Geodesic_update(replay_seq[-1], replay_seq[:-1])
		self.G += dG

		if update_policies:
			for memory in replay_seq:
				s_k, a_k, s_kp = memory
				for goal_state in self.goal_states:
					self.update_state_policy(s_k, goal_state, set_policy=True)

	def remember(self, transitions):
		'''
			Add a set of transitions to the memory bank.
		'''
		for transition in transitions:
			if transition not in self.memory:
				self.memory.extend([transition])

	def compute_Geodesic_update(self, transitions):
		'''
			Given a set of transitions, compute the Bellman update to the GR
			for each of them. This is useful for cases in which we might want
			to develop a list of what the updates *would* be, but not commit
			to actually updating G. 
		'''
		updates = []
		for transition in transitions:
			s, a, s_p = transition

			## Bellman update for Geodesic representation
			# Convenience variables for various terms
			one_hot = np.zeros(self.num_states)
			one_hot[s_p] = 1

			zero_hot = np.ones(self.num_states)
			zero_hot[s_p] = 0

			max_G = np.max(self.G[s_p, :, :], axis=0)

			# Put them all together again
			updates.append(self.alpha * (one_hot + np.multiply(zero_hot, self.gamma * max_G) - self.G[s, a, :]))

		return updates

	def compute_trace_Geodesic_update(self, transition, replay_seq):
		'''
			Compute update to G tensor for a sequential set of transitions assumed to
			be on-policy with respect to G*. The last entry in replay_seq is taken to be
			the reference transition from which the GR error is computed.

			A subtlety with this type of update is that one should not update states already
			visited on the path, due to the absorbing assumption in the GR. Thus we zero-out
			entries in delta corresponding to states already in replay_seq. 
		'''
		# Compute delta elements
		s_k, a_k, s_kp = transition
		one_hot = np.zeros(self.num_states)
		one_hot[s_kp] = 1

		zero_hot = np.ones(self.num_states)
		zero_hot[s_kp] = 0 # Will add other zero_hot elements through the sequence of memories

		G_max = np.max(self.G[s_kp, :, :], axis=0)

		# Dummy copy to make changes to
		dG = np.zeros_like(self.G)

		# Do update for original transition
		delta = one_hot + np.multiply(zero_hot, self.gamma * G_max) - self.G[s_k, a_k, :]
		dG[s_k, a_k, :] += self.alpha * delta

		# Walk through memories in replay list and make updates
		# Doing this in reverse order makes the bookkeeping easier, since you just add 0s
		# to zero_hot
		for mdx, memory in reversed(list(enumerate(replay_seq))):
			s_k, a_k, s_kp = memory

			# Zero out contributions to my current successor state (since it has already been taken into account)
			zero_hot[s_kp] = 0

			# Collect and update
			delta = one_hot + np.multiply(zero_hot, self.gamma * G_max) - self.G[s_k, a_k, :]
			dG[s_k, a_k, :] += self.alpha * math.pow(self.gamma, mdx + 1) * delta

		return dG

	def __creates_loop(self, transition, replay_seq):
		'''
			Does adding transition to replay_seq create a loop?
		'''
		s_k, _, s_kp = transition
		loop = False
		for memory in replay_seq:
			s_m, _, s_mp = memory

			if s_kp == s_m:
				loop = True
				break

		return loop

	def replay(self, num_steps, goal_states=None, goal_dist=None, prospective=False, verbose=False):
		'''
			Perform replay, prioritised under a (meta-) expected value of backup rule.
			Do this by iterating over all available transitions in memory, and averaging
			the EVBs over the list of potential future goal states.
		'''
		# Input validation, blah blah
		if goal_states is None:
			goal_states = self.goal_states
		if goal_dist is None:
			goal_dist = self.goal_dist

		# If verbose usage, build storage structures
		if verbose:
			needs = np.zeros((num_steps, len(goal_states), self.num_states, self.num_states))
			gains = np.zeros((num_steps, len(goal_states), len(self.memory)))
			all_MEVBs = np.zeros((num_steps, len(goal_states), len(self.memory)))

		# Start replaying
		replay_seq = [] # Maintain a list of replayed memories for use in multi-step backups
		for step in range(num_steps):
			MEVBs = np.zeros(len(self.memory)) # Best transition is picked greedily at each step

			# Compute EVB for all transitions across all goal states
			for gdx, goal in enumerate(goal_states):

				# If we have a policy cached, grab it. Otherwise, recompute fully.
				if goal in self.goal_states:
					policy = self.policies[goal]
				else:
					policy = self.derive_policy(goal)

				# Compute SR induced by this policy and the task dynamics
				M_pi = self.compute_occupancy(policy, self.T)

				# Log, if wanted
				if verbose:
					needs[step, gdx, :, :] = M_pi

				# Compute EVB for each transition
				for tdx, transition in enumerate(self.memory):
					need, gain, evb = self.compute_multistep_EVB(transition, goal, policy,
																 replay_seq, 
																 state=self.curr_state,
																 M=M_pi,
																 prospective=prospective)

					MEVBs[tdx] += goal_dist[gdx] * mevb

					# Log quantities, if desired
					if verbose:
						all_MEVBs[step, gdx, tdx] = evb
						gains[step, gdx, tdx] = gain

			# Pick the best one
			best_memory = self.memory[np.argmax(MEVBs)]
			replay_seq.append(best_memory)

			# Learn!
			self.nstep_learn(replay_seq)

		if verbose:
			return np.array(replayed_experiences), (needs, gains, all_MEVBs)

	def nstep_learn(transition_seq, update_policies=True):
		'''
			Update GR according to transition sequence. Treat last transition in sequence
			as primary transition.
		'''
		dG = self.compute_nstep_update(transition_seq[-1], transition_seq[:-1])
		self.G += self.alpha * dG

		if update_policies:
			for goal in self.goal_states:
				self.policies[goal] = self.derive_policy(goal)

	def compute_multistep_EVB(self, transition, goal, policy, replay_seq, curr_state, M, prospective=False):
		'''
			Compute the expected value of GR backup for a particular sequence of transitions
			with respect to a particular goal state. Derivation for the factorization
			EVB = need * gain follows from Mattar & Daw (2018), defining GR analogues
			of Q and V functions.
		'''
		# Collect variables
		s_k, a_k, s_kp = transition

		# Compute the need of this transition wrt this goal
		need = self.compute_need(curr_state, s_k, M, prospective)

		# Derivation of this update prioritisation shows that EVB = 0 for the special case where
		# s_k == goal
		if s_k == goal:
			return need, 0, 0

		# Get longest subsequence from the end of replay_seq that is an optimal path towards
		# goal. This whole subsequence qualifies for a multi-step backup, and so we compute
		# the gain over it, rather than simply the state at the end.
		s_m, a_m, s_mp = replay_seq[-1]
		if s_mp == s_k: # This path must, of course, end in the transition under consideration
			optimal_subseq = self.get_optimal_subseq(replay_seq, goal)
		else:
			optimal_subseq = []

		# Compute gain for this subsequence as well as the transition under consideration
		gain = self.compute_nstep_gain(transition, optimal_subseq, goal, policy)

	def get_optimal_subseq(self, replay_seq, goal, tol=1e-6):
		'''
			Compute the longest subsequence (starting from the end) in replay_seq that constitutes
			an optimal path towards goal under the given policy.
		'''
		optimal_subseq = []
		for tdx, transition in enumerate(reversed(replay_seq)):
			s_k, a_k, s_kp = transition

			# If a_k is optimal in s_k...
			if abs(self.G[s_k, a_k, goal] - np.max(G[s_k, :, goal])) <= tol: 

				# ... and also, it leads to the first member of the optimal subsequence...
				if s_kp == optimal_subseq[0][0]:

					# then add it to the optimal subsequence.
					optimal_subseq.insert(0, transition)

			# Otherwise, quit
			break

		return optimal_subseq

	def compute_nstep_update(self, transition, replay_seq, goal_states=None):
		'''
			Given a primary transition and a potentially-empty subsequence of transitions leading to it,
			compute what the net update to the GR is.
		'''

		# Collect variables
		s_k, a_k, s_kp = transition
		dG = np.zeros_like(self.G)

		if goal_states is None:
			goal_states = self.goal_states

		# For each goal...
		for goal in goal_states:

			# Compute GR delta wrt this goal
			if s_kp == goal:
				GR_delta = 1 - self.G[s_k, a_k, goal]
			else:
				GR_delta = self.gamma * np.max(self.G[s_kp, :, goal]) - self.G[s_k, a_k, goal]

			# Implement delta due to primary transition
			dG[s_k, a_k, goal] += GR_delta

			# Find optimal subsequence wrt this goal
			optimal_subseq = self.get_optimal_subseq(replay_seq, goal)

			# Backpropagate delta throughout this subsequence as relevant
			for mdx, memory in enumerate(optimal_subseq):
				s_m, a_m, s_mp = memory
				dG[s_m, a_m, goal] += (self.gamma ** (mdx + 1)) * GR_delta

		return dG

	def compute_nstep_gain(self, transition, optimal_subseq, goal, policy, G_p=None):
		'''
			Compute gain blah
		'''

		# Collect variables
		s_k, a_k, s_kp = transition

		# Compute new GR given this primary transition + optimal subsequence
		if G_p is None:
			G_p = self.G.copy() + self.alpha * self.compute_nstep_update(transition, 
																		 optimal_subseq, 
																		 goal_states=[goal])

		## Rederive policy wrt this new GR
		pi_p = self.derive_policy(goal, G=G_p)

		## Compute gain
		gain = 0

		# Get gain due to primary transition
		for action in range(self.num_actions):
			gain += (pi_p[s_k, action] - policy[s_k, action]) * G_p[s_k, action, goal]

		# Get gain due to states visited during n-step backup
		for mdx, memory in enumerate(optimal_subseq):
			s_m, a_m, s_mp = memory
			for action in range(self.num_actions):
				gain += (pi_p[s_m, action] - policy[s_m, action]) * G_p[s_m, action, goal]

		return gain

	def dumb_replay(self, num_steps, return_seq=True):
		'''
			Dumb replay is dumb. Uniformly sample at random from memory and replay what you get 
		'''
		replayed_experiences = []
		for i in range(num_steps):
			memory = np.random.choice(self.memory)
			self.learn([memory])
			replayed_experiences.append(memory)

		if return_seq:
			return replayed_experiences

	def compute_EVB(self, transition, goal_state, policy, state=None, M=None, prospective=False):
		'''
			Compute the expected value of GR backup for a particular transition
			with respect to a particular goal state. Derivation for the factorization
			EVB = need * gain follows from Mattar & Daw (2018), defining GR analogues
			of Q and V functions.
		'''
		# Collect variables
		s_k, a_k, s_kp = transition

		if s_k == goal_state: # Can be shown analytically that this case nets 0 gain
			return self.min_gain, self.min_gain, self.min_gain
		
		if state is None:
			state = self.curr_state
		if M is None:
			M = self.compute_occupancy(policy, self.T)

		# Compute EVB factors and evaluate
		need = self.compute_need(state, s_k, M, prospective)
		gain = self.compute_gain(transition, goal_state, policy)

		return need, gain, need * gain

	def compute_need(self, state, s_k, M, prospective=False):
		'''
			Compute the need term of the GR EVB equation.
		'''
		
		if prospective: # Average needs across all possible start states
			return np.average(M[:, s_k], weights=self.s0_dist)
		else:
			return M[state, s_k]

	def compute_occupancy(self, policy, T):
		'''
			Compute future state occupancy matrix given a policy `policy`
			and transition dynamics matrix `T`
		'''
		# Convert dynamics + policy to one-step transition matrix
		one_step_T = dynamics_policy_onestep(policy, T)

		# Compute resultant future occupancy matrix and evaluate
		M = np.linalg.inv(np.eye(self.num_states) - self.gamma * one_step_T)

		return M

	def compute_gain(self, transition, goal_state, policy):
		'''
			Compute the gain term of the GR EVB equation.
		'''
		s_k, a_k, s_kp = transition

		# Old policy
		pi = policy

		# G after learning
		G_p = self.G.copy()
		dG = self.compute_Geodesic_update([transition])[0].flatten()
		G_p[s_k, a_k, :] += dG

		# New policy
		pi_p = self.update_state_policy(s_k, goal_state, G=G_p) 

		gain = 0
		for action in range(self.num_actions):
			gain += (pi_p[s_k, action] - pi[s_k, action]) * G_p[s_k, action, goal_state]
		
		return gain

if __name__ == '__main__':
	side_length = 3
	num_states = side_length ** 2
	num_actions = 4
	uga = GeodesicAgent(num_states, num_actions, [0], gamma=0.9)

	# Build transition bank to learn G
	transitions = []
	for state in range(num_states):
		for action in range(num_actions):
			state_2d = np.unravel_index(state, (side_length, side_length))

			if action == 0: # Go left
				if state_2d[1] == 0: # Already at left border
					succ_state = state
				else:
					succ_state = state - 1

			if action == 1: # Go up
				if state_2d[0] == 0: # Already at top border
					succ_state = state
				else:
					succ_state = state - side_length

			if action == 2: # Go right
				if state_2d[1] == side_length - 1: # Already at right border
					succ_state = state
				else:
					succ_state = state + 1

			if action == 3: # Go down
				if state_2d[0] == side_length - 1: # Already at bottom border
					succ_state = state
				else:
					succ_state = state + side_length

			transitions.append((state, action, succ_state))

	num_learning_iterations = 1000
	for _ in range(num_learning_iterations):
		uga.learn(transitions)

	print(uga.G[0,2,2])
	print(uga.derive_policy(0))