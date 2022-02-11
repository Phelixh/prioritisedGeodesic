import numpy as np

from MarkovDecisionProcess import MarkovDecisionProcess
from RL_utils import oned_twod, compute_occupancy

class DiGraph(MarkovDecisionProcess):
	'''
		The DiGraph class is a particular instance of the MarkovDecisionProblem
		class. In particular, it is a directed graph in which environmental
		dynamics are deterministic. 
	'''

	def __init__(self, num_vertices, edges, init_state_dist=None):
		# Convert graph specification to MDP
		num_states = num_vertices
		num_actions = edges.shape[1]
		self.transitions = np.zeros((num_states, num_actions, num_states))

		# Build MDP transition matrix from graph adjacency matrix
		for source in range(num_states):
			for action in range(num_actions):
				target = int(edges[source, action])
				self.transitions[source, action, target] = 1

		# Build MDP using super-class's constructor
		super().__init__(self.transitions, num_actions, init_state_dist)

class CommunityGraph(DiGraph):
	'''
		The CommunityGraph class is a particular graph structure governed by small,
		fully-connected neighbourhoods with sparse transitions between them.

		In particular, for each neighbourhood, neighbour ID 0 is the input node, which
		receives transitions from the output nodes of all other neighbourhoods. 
		Output nodes are given neighbour ID neighbourhood_size - 1.
	'''

	def __init__(self, num_neighbourhoods, neighbourhood_size, init_state_dist=None):
		num_vertices = num_neighbourhoods * neighbourhood_size

		# Transitions to all the neighbours + attempt to hop to other neighbourhood
		num_actions = neighbourhood_size + num_neighbourhoods 

		self.edges = np.zeros((num_vertices, num_actions), dtype=int)
		for nbrhd in range(num_neighbourhoods):
			for nbr in range(neighbourhood_size):
				nbr_vid = self.__class__.nbr_to_vtx(nbr, nbrhd, neighbourhood_size)

				for action in range(num_actions):
					if action <= neighbourhood_size - 1: # First few actions transition within the neighbourhood
						target_nbr = (nbr + action + 1) % neighbourhood_size
						target_vid = self.__class__.nbr_to_vtx(target_nbr, nbrhd, neighbourhood_size)
						self.edges[nbr_vid, action] = target_vid

					else: # Remaining actions attempt to transition to another neighbourhood
						if nbr == neighbourhood_size - 1: # Output node
							target_nbrhd = action - neighbourhood_size
							target_nbr = 0
							target_vid = self.__class__.nbr_to_vtx(target_nbr, target_nbrhd, neighbourhood_size)
							self.edges[nbr_vid, action] = target_vid

						else: # Non-output node
							self.edges[nbr_vid, action] = nbr_vid

		# Build MDP using super-class's constructor
		super().__init__(num_vertices, self.edges, init_state_dist)

	def nbr_to_vtx(nbr, nbrhd, neighbourhood_size):
		return neighbourhood_size * nbrhd + nbr 

	def get_all_transitions(self):
		Ts = []
		for src_id in range(self.num_states):
			for action in range(self.num_actions):
				transition = (src_id, action, self.edges[src_id, action])
				Ts.append(transition)

		return Ts


