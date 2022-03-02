import numpy as np
from RL_utils import dynamics_policy_onestep

# TODO: think about zeroing out self-distances
# TODO: need term should be occupancy in the modified self-looped MDP not the true MDP


class GeodesicAgent(object):
	"""
		The GeodesicAgent class solves MDPs using the Geodesic Representation (GR),
		which is a control version of the Successor Representation (SR). Unlike the SR,
		which develops a matrix of future state occupancy under a given policy, the GR
		develops a matrix G[i, a, j] = gamma^d(i, a, j), where gamma is a discounting
		factor and d(i, a, j) is the length of the shortest path from `i` to `j` after taking
		action `a`.

		Unlike the SR, the GR can be used to solve navigation problems in the face of 
		dynamic or uncertain environments.
	"""

	def __init__(self, num_states: int, num_actions: int, goal_states: np.ndarray,
				 T: np.ndarray,
				 goal_dist: np.ndarray = None,
				 s0_dist: np.ndarray = None,
				 alpha: float = 0.3,
				 gamma: float = 0.95,
				 min_gain: float = 0):
		"""
		Construct the GeodesicAgent object.

		Args:
			num_states (int): Number of states in the underlying MDP
			num_actions (int): Number of actions available at each state
			goal_states (np.ndarray): A subset of states that are marked as "goals"
			T (np.ndarray): The one-step transition matrix for the MDP.
				T[s, a, g] gives the probability of transitioning from state s to state g after
				taking action a.
			goal_dist (np.ndarray): The distribution over which goals are most likely to manifest
			s0_dist (np.ndarray): Initial state distribution
			alpha (float): Learning rate
			gamma (float): Temporal discount rate
			min_gain (float): Minimum value for gain computation
		"""

		# MDP properties
		self.num_states = num_states
		self.num_actions = num_actions
		self.goal_states = goal_states
		self.curr_state = -1  # Pre-initial state indicating action has not yet started
		if goal_dist is None:
			goal_dist = np.ones(len(goal_states)) / len(goal_states)
		self.goal_dist = goal_dist
		self.T = T
		self.s0_dist = s0_dist

		# Agent properties
		self.alpha = alpha  # Learning rate
		self.gamma = gamma
		self.min_gain = min_gain
		self.G = np.zeros(
			(num_states, num_actions, num_states))  # Geodesic representation matrix is gamma^shortest_path
		self.memory = []  # Memory bank for later replay

		# Separate policies for each goal state, each initialised as uniform
		uniform_policy = np.ones((num_states, num_actions)) / num_actions
		self.policies = {goal_states[i]: uniform_policy.copy() for i in range(len(goal_states))}

		# Separate transition structures for each of the modified MDPs, one per goal
		self.mod_Ts = {}
		for i in range(len(goal_states)):
			mod_T = self.T.copy()
			goal = goal_states[i]
			mod_T[goal, :, :] = 0
			mod_T[goal, :, goal] = 1

			self.mod_Ts[goal_states[i]] = mod_T

	def derive_policy(self, goal_state, G=None, set_policy=False, epsilon=0):
		"""
		Derive the policy for reaching a given goal state. Since
		the GR represents the shortest (expected) paths, we can
		simply take the max at every state.

		Allow an epsilon-greedy addition to facilitate exploration.

		Args:
			goal_state (int): The goal with respect to which the policy is derived.
			G (np.ndarray, optional): The Geodesic representation over which the policy is derived.
				If none is provided, the agent's current one will be used.
			set_policy (boolean): If True, the computed policy will update the agent's current policy
				 for the specified goal.
			epsilon (float): Epsilon parameter for epsilon-greedy action policy.

		Returns:
			policy (np.ndarray): The computed policy.
		"""

		# Allow arbitrary G to be fed in, default to current G
		if G is None:
			G = self.G

		policy = np.zeros((self.num_states, self.num_actions))

		# Compute policy
		for state in range(self.num_states):
			best_actions = np.flatnonzero(G[state, :, goal_state] == np.max(G[state, :, goal_state]))
			num_best_actions = len(best_actions)  # Split 1 - epsilon ties equally

			policy[state, :] = epsilon / self.num_actions
			policy[state, best_actions] += (1 - epsilon) / num_best_actions  # Deterministic for epsilon = 0

		# Cache if wanted
		if set_policy:
			self.policies[goal_state] = policy

		return policy

	def update_state_policy(self, state, goal_state, G=None, set_policy=False, epsilon=0):
		"""
		Update the internal policy only for a given state.

		Allow an epsilon-greedy addition to facilitate exploration.

		Args:
			state (int): The state receiving the policy update
			goal_state (int): The state with respect to which the policy is being updated
			G (np.ndarray, optional): The Geodesic representation over which the policy is derived.
				If none is provided, the agent's current one will be used.
			set_policy (boolean): If True, the computed policy will update the agent's current policy
				 for the specified goal.
			epsilon (float): Epsilon parameter for epsilon-greedy action policy.

		Returns:
			policy (np.ndarray): The computed policy.
		"""

		# Allow arbitrary G to be fed in, default to current G
		if G is None:
			G = self.G

		# Compute policy
		best_actions = np.flatnonzero(G[state, :, goal_state] == np.max(G[state, :, goal_state]))
		num_best_actions = len(best_actions)  # Split 1 - epsilon ties equally

		if not set_policy:  # Only re-copy the whole thing if we're not planning on saving it.
			policy = self.policies[goal_state].copy()
		else:
			policy = self.policies[goal_state]

		policy[state, :] = epsilon / self.num_actions  # Deterministic for epsilon = 0
		policy[state, best_actions] += (1 - epsilon) / num_best_actions

		# Cache if wanted
		if set_policy:
			self.policies[goal_state] = policy

		return policy

	def remember(self, transitions):
		"""
                Add a set of transitions to the memory bank.

                Args:
                    transitions (list): The list of memories to be added to the memory bank.
                        Each memory must be a tuple (s, a, g), indicating that action a was
                        taken in state s and reached state g.
                """
		for transition in transitions:
			if transition not in self.memory:
				self.memory.extend([transition])

	def replay(self, num_steps, goal_states=None, goal_dist=None, prospective=False, verbose=False,
			   check_convergence=True, convergence_thresh=0.0, otol=1e-6, learn_seq=None):
		"""
		Perform replay, prioritised under a (meta-) expected value of backup rule.
		Do this by iterating over all available transitions in memory, and averaging
		the EVBs over the list of potential future goal states.

		Args:
			num_steps (int): Maximum number of steps of replay to be performed.
			goal_states (np.ndarray): The set of particular goal states with respect to which replay should occur.
			goal_dist (np.ndarray): The distribution weighting those goals.
			prospective (boolean): Controls whether the agent plans prospectively or using their current state.
				If prospective=False, the need term of EVB is computed with respect to the agent's current state.
				If prospective=True, the need term of EVB is computed with respect to the agent's initial
				state distribution.
			verbose (boolean): Controls whether various intermediate variables are returned at the end of the process.
			check_convergence (boolean): Controls whether replay can end early if the Geodesic representation has
				converged.
			convergence_thresh (float): Tolerance on absolute mean change in the GR for convergence.
			otol (float): Ties in EVB are broken randomly. Otol defines the threshold for a tie.
			learn_seq (list): If provided, learn_seq stipulates the sequence of state to be replayed. All the EVB
				metrics are still computed for analysis purposes, but the outcome is ignored.

		Returns:
			replay_seq (np.ndarray):
			needs (np.ndarray):
			gains (np.ndarray):
			all_MEVBs (np.ndarray):
			backups (np.ndarray):
		"""
		# Input validation, blah blah
		if goal_states is None:
			goal_states = self.goal_states
		if goal_dist is None:
			goal_dist = self.goal_dist

		# If verbose usage, build storage structures
		state_needs = None
		transition_needs = None
		gains = None
		all_MEVBs = None
		if verbose:
			state_needs = np.zeros((num_steps, len(goal_states), self.num_states, self.num_states))
			transition_needs = np.zeros((num_steps, len(goal_states), len(self.memory)))
			gains = np.zeros((num_steps, len(goal_states), len(self.memory)))
			all_MEVBs = np.zeros((num_steps, len(goal_states), len(self.memory)))

		# Start replaying
		replay_seq = []  # Maintain a list of replayed memories for use in multistep backups
		backups = []  # Maintain a list of transitions replayed in each backup step
		for step in range(num_steps):
			MEVBs = np.zeros(len(self.memory))  # Best transition is picked greedily at each step
			G_ps = {}  # At each replay step, cache G primes since they are goal-invariant

			# Compute EVB for all transitions across all goal states
			for gdx, goal in enumerate(goal_states):
				# If we have a policy cached, grab it. Otherwise, recompute fully.
				if goal in self.goal_states:
					policy = self.policies[goal]
				else:
					policy = self.derive_policy(goal)

				# Compute SR induced by this policy and the task dynamics
				M_pi = self.compute_occupancy(policy, self.mod_Ts[goal])

				# Log, if wanted
				if verbose:
					state_needs[step, gdx, :, :] = M_pi

				# Compute EVB for each transition
				for tdx, transition in enumerate(self.memory):
					if tdx in G_ps.keys():
						G_p = G_ps[tdx]
					else:
						dG, _ = self.compute_nstep_update(transition, replay_seq=replay_seq,
														  goal_states=goal_states)
						G_p = self.G + self.alpha * dG
						G_ps[tdx] = G_p

					need, gain, evb = self.compute_multistep_EVB(transition, goal, policy,
																 replay_seq,
																 curr_state=self.curr_state,
																 M=M_pi,
																 G_p=G_p,
																 prospective=prospective)

					MEVBs[tdx] += goal_dist[gdx] * evb

					# Log quantities, if desired
					if verbose:
						all_MEVBs[step, gdx, tdx] = evb
						gains[step, gdx, tdx] = gain
						transition_needs[step, gdx, tdx] = need

			# Pick the best one
			if learn_seq:
				best_memory = learn_seq[step]
			else:
				best_memories = np.argwhere(np.abs(MEVBs - np.max(MEVBs)) <= otol).flatten()
				best_memory = self.memory[np.random.choice(best_memories)]

			replay_seq.append(best_memory)

			# Learn!
			if check_convergence:
				backup, mag_delta = self.nstep_learn(replay_seq, ret_update_mag=True)
				if mag_delta <= convergence_thresh:  # Reached convergence
					# Cap the storage data structures, if necessary
					if verbose:
						state_needs = state_needs[:step, :, :, :]
						gains = gains[:step, :, :]
						all_MEVBs = all_MEVBs[:step, :, :]
						transition_needs = transition_needs[:step, :, :]

					break
			else:
				backup = self.nstep_learn(replay_seq)

			backups.append(backup)

		if verbose:
			return np.array(replay_seq), (state_needs, transition_needs, gains, all_MEVBs), backups

	def nstep_learn(self, transition_seq, update_policies=True, ret_update_mag=False):
		"""
			Update GR according to transition sequence. Treat last transition in sequence
			as primary transition.
		"""
		dG, opt_subseqs = self.compute_nstep_update(transition_seq[-1], replay_seq=transition_seq[:-1])
		self.G += self.alpha * dG

		if update_policies:
			for goal in self.goal_states:
				self.policies[goal] = self.derive_policy(goal)

		if ret_update_mag:
			return opt_subseqs, np.sum(self.alpha * np.abs(dG))
		else:
			return opt_subseqs

	def compute_multistep_EVB(self, transition, goal, policy, replay_seq, curr_state, M, G_p=None, prospective=False):
		"""
			Compute the expected value of GR backup for a particular sequence of transitions
			with respect to a particular goal state. Derivation for the factorization
			EVB = need * gain follows from Mattar & Daw (2018), defining GR analogues
			of Q and V functions.
		"""
		# Collect variables
		s_k, a_k, s_kp = transition

		# Compute the need of this transition wrt this goal
		need = self.compute_need(curr_state, s_k, M, prospective)

		# Derivation of this update prioritisation shows that EVB = 0 for the special case where
		# s_k == goal
		if s_k == goal:
			return need, 0, 0

		# Compute gain for this transition (and induced n-step backup)
		gain = self.compute_nstep_gain(transition, replay_seq, goal, policy, G_p=G_p)

		# Compute and return EVB + factors
		return need, gain, need * gain

	def get_optimal_subseq(self, replay_seq, goal, tol=1e-6, end=None):
		"""
			Compute the longest subsequence (starting from the end) in replay_seq that constitutes
			an optimal path towards goal under the given policy.
		"""
		if end == goal:  # Special case
			return []

		optimal_subseq = []
		for tdx, transition in enumerate(reversed(replay_seq)):
			s_k, a_k, s_kp = transition

			if tdx == 0 and s_kp != end:  # We require that the sequence conclude at state end
				break

			if s_k == goal:  # Self-trajectories from the goal, to the goal, are somewhat ill-defined
				break

			# If a_k is optimal in s_k...
			if self.check_optimal(s_k, a_k, goal, tol=tol):
				# ... and also, it leads to the first member of the optimal subsequence...
				if not optimal_subseq or s_kp == optimal_subseq[0][0]:
					# then add it to the optimal subsequence.
					optimal_subseq.insert(0, transition)
				else:
					break

			# Otherwise, quit
			else:
				break

		return optimal_subseq

	def compute_nstep_update(self, transition, replay_seq=None, optimal_subseqs=None, goal_states=None):
		"""
			Given a primary transition and a potentially-empty subsequence of transitions leading to it,
			compute what the net update to the GR is.

			Either one of replay_seq or optimal_subseq must be provided.
		"""
		# Collect variables
		s_k, a_k, s_kp = transition
		dG = np.zeros_like(self.G)

		if goal_states is None:
			goal_states = self.goal_states

		# For each goal...
		computed_subseqs = {}
		for gdx, goal in enumerate(goal_states):
			# Compute GR delta wrt this goal
			if s_kp == goal:
				GR_delta = 1 - self.G[s_k, a_k, goal]
			else:
				GR_delta = self.gamma * np.max(self.G[s_kp, :, goal]) - self.G[s_k, a_k, goal]

			# Implement delta due to primary transition
			dG[s_k, a_k, goal] += GR_delta

			# Find optimal subsequence wrt this goal
			if optimal_subseqs is not None:
				optimal_subseq = optimal_subseqs[gdx]
			elif not self.check_optimal(s_k, a_k, goal):  # Exploratory actions do not backpropagate
				optimal_subseq = []
			else:
				optimal_subseq = self.get_optimal_subseq(replay_seq, goal, end=s_k)

			computed_subseqs[goal] = optimal_subseq

			# Backpropagate delta throughout this subsequence as relevant
			for mdx, memory in enumerate(reversed(optimal_subseq)):
				s_m, a_m, s_mp = memory
				dG[s_m, a_m, goal] += (self.gamma ** (mdx + 1)) * GR_delta

		return dG, computed_subseqs

	def check_optimal(self, s_k, a_k, goal, tol=1e-6):
		return abs(self.G[s_k, a_k, goal] - np.max(self.G[s_k, :, goal])) <= tol

	def compute_nstep_gain(self, transition, replay_seq, goal, policy, G_p=None, optimal_subseq=None):
		"""
			Compute gain blah
		"""

		# Collect variables
		s_k, a_k, s_kp = transition

		# Get optimal subsequence of replay_seq with respect to goal
		if optimal_subseq is None:
			optimal_subseq = self.get_optimal_subseq(replay_seq, goal, end=s_k)

		# Compute new GR given this primary transition + optimal subsequence
		if G_p is None:
			dG, _ = self.compute_nstep_update(transition, optimal_subseqs=[optimal_subseq], goal_states=[goal])
			G_p = self.G.copy() + self.alpha * dG

		## Compute gain
		gain = 0

		# Get gain due to primary transition
		pi_p = self.update_state_policy(s_k, goal, G=G_p)
		for action in range(self.num_actions):
			gain += (pi_p[s_k, action] - policy[s_k, action]) * G_p[s_k, action, goal]

		# Get gain due to states visited during n-step backup
		for mdx, memory in enumerate(optimal_subseq):
			s_m, a_m, s_mp = memory
			pi_p = self.update_state_policy(s_m, goal, G=G_p)
			for action in range(self.num_actions):
				gain += (pi_p[s_m, action] - policy[s_m, action]) * G_p[s_m, action, goal]

		return gain

	def compute_need(self, state, s_k, M, prospective=False):
		"""
			Compute the need term of the GR EVB equation.
		"""

		if prospective:  # Average needs across all possible start states
			return np.average(M[:, s_k], weights=self.s0_dist)
		else:
			return M[state, s_k]

	def compute_occupancy(self, policy, T):
		"""
			Compute future state occupancy matrix given a policy `policy`
			and transition dynamics matrix `T`
		"""
		# Convert dynamics + policy to one-step transition matrix
		one_step_T = dynamics_policy_onestep(policy, T)

		# Compute resultant future occupancy matrix and evaluate
		M = np.linalg.inv(np.eye(self.num_states) - self.gamma * one_step_T)

		return M


if __name__ == '__main__':
	pass
