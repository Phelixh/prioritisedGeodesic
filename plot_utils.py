from gridworld import GridWorld
from geodesic_agent import GeodesicAgent
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
from copy import copy

from RL_utils import oned_twod

def plot_replay(gridworld, replay_sequence, cmap=None, ax=None, figsize=(12,12)):
	'''
		Given a GridWorld object, and a list of replayed states in that GridWorld, plot the sequence in a nice
		and pretty way.

		Params:
			gridworld: GridWorld object
			replay_seequence: N x 3 array, where each row is (start state, action, successor state)
	'''

	# Paint the grid world to the figure
	ax = gridworld.draw(use_reachability=True, ax=ax, figsize=figsize)

	## Now add arrows for replayed states
	# Colours!
	arrow_colours = plt.cm.winter(np.linspace(0, 1, replay_sequence.shape[0]))
	CENTRE_OFFSET = 0.5 # oned_twod gives the coordinate of the top left corner of the state
	for i in range(replay_sequence.shape[0]):
		# Get plotting coordinates
		start, action, successor = replay_sequence[i, :]
		start_y, start_x = np.array(oned_twod(start, gridworld.width, gridworld.height)) + CENTRE_OFFSET
		succ_y, succ_x = np.array(oned_twod(successor, gridworld.width, gridworld.height)) + CENTRE_OFFSET

		# Plot
		ax.arrow(start_x, start_y, succ_x - start_x, succ_y - start_y, 
				  length_includes_head=True, head_width=0.25, color=arrow_colours[i])

	# Add colourbar
	divider = make_axes_locatable(ax)
	cax = divider.append_axes('right', size='5%', pad=0.1)
	cbar = mpl.colorbar.ColorbarBase(cax,cmap=plt.cm.winter,orientation='vertical', ticks=[0, 1])
	cbar.ax.set_yticklabels(['start', 'end'])

	return ax

def plot_traj(gridworld, state_sequence, cmap=None, ax=None, figsize=(12,12)):
	'''
		Given a GridWorld object, and a list of replayed states in that GridWorld, plot the sequence in a nice
		and pretty way.

		Params:
			gridworld: GridWorld object
			state_sequence: N x 1 array, consisting of visited state sequence
	'''
	# Paint the grid world to the figure
	ax = gridworld.draw(use_reachability=True, ax=ax, figsize=figsize)
	
	## Now add arrows for traversed states
	# Colours!
	arrow_colours = plt.cm.winter(np.linspace(0, 1, len(state_sequence)))
	CENTRE_OFFSET = 0.5 # oned_twod gives the coordinate of the top left corner of the state
	for i in range(len(state_sequence) - 1):
		# Get plotting coordinates
		start = state_sequence[i]
		successor = state_sequence[i + 1]
		start_y, start_x = np.array(oned_twod(start, gridworld.width, gridworld.height)) + CENTRE_OFFSET
		succ_y, succ_x = np.array(oned_twod(successor, gridworld.width, gridworld.height)) + CENTRE_OFFSET

		# Plot
		ax.arrow(start_x, start_y, succ_x - start_x, succ_y - start_y, 
				  length_includes_head=True, head_width=0.25, color=arrow_colours[i])

	# Add colourbar
	divider = make_axes_locatable(ax)
	cax = divider.append_axes('right', size='5%', pad=0.1)
	cbar = mpl.colorbar.ColorbarBase(cax,cmap=plt.cm.winter,orientation='vertical', ticks=[0, 1])
	cbar.ax.set_yticklabels(['start', 'end'])

	return ax

def plot_need_gain(gridworld, transitions, need, gain, MEVB, specials=None, params={}):
	fig, axes = plt.subplots(1, 3, figsize=(18, 6))

	## Need
	# Plot need by shading in states
	ax = axes[0]
	ax.set_title('Need')
	gridworld.draw(use_reachability=True, ax=ax)

	# Grab boundaries
	min_need = np.min(need[np.nonzero(need)]) # Dumb hack so that need = 0 states don't appear slightly red
	max_need = np.max(need)
	alpha_fac = 1

	if 'min_need' in params.keys():
		min_need = params['min_need']
	if 'max_need' in params.keys():
		max_need = params['max_need']
	if 'alpha_fac' in params.keys():
		alpha_fac = params['alpha_fac']

	norm_need = mpl.colors.Normalize(vmin=min_need, vmax=max_need)(need)

	# Build custom palette without dumb red bottom boundary
	palette = copy(plt.get_cmap('Reds'))
	palette.set_under('white', 1.0)

	# Get colours for each state
	state_colours = palette(norm_need).reshape(-1, 4)
	for state in range(gridworld.num_states):
		if hasattr(gridworld, 'banned_states') and state in gridworld.banned_states:
			continue
		row, col = oned_twod(state, gridworld.width, gridworld.height)
		rect = patches.Rectangle((col, row), 1, 1, facecolor=state_colours[state])
		ax.add_patch(rect)

	# Add colour bar
	divider = make_axes_locatable(ax)
	cax = divider.append_axes('right', size='5%', pad=0.05)
	cbar = mpl.colorbar.ColorbarBase(cax,cmap=palette,orientation='vertical', ticks=[0, 1])
	cbar.ax.set_yticklabels(['%.3f' % min_need, '%.3f' % max_need])

	## Gain
	# Plot gain by shading arrows
	ax = axes[1]
	ax.set_title('Gain')
	gridworld.draw(use_reachability=True, ax=ax)

	# Grab boundaries
	min_gain = np.min(gain) 
	max_gain = np.max(gain)

	if 'min_gain' in params.keys():
		min_gain = params['min_gain']
	if 'max_gain' in params.keys():
		max_gain = params['max_gain']

	norm_gain = mpl.colors.Normalize(vmin=min_gain, vmax=max_gain)(gain)
	gain_colours = plt.cm.winter(norm_gain).reshape(-1, 4)
	gain_colours[:, 3] = norm_gain / alpha_fac # Modulate alpha in accordance with gain as well
	CENTRE_OFFSET = 0.5 # oned_twod gives the coordinate of the top left corner of the state
	for tdx, transition in enumerate(transitions):
		s_k, a_k, s_kp = transition
		start_y, start_x = np.array(oned_twod(s_k, gridworld.width, gridworld.height)) + CENTRE_OFFSET
		succ_y, succ_x = np.array(oned_twod(s_kp, gridworld.width, gridworld.height)) + CENTRE_OFFSET

		# Plot
		if abs(np.max(norm_gain) - norm_gain[tdx]) < 1e-8: # Distinguish the maximal gain transitions
			ax.arrow(start_x, start_y, succ_x - start_x, succ_y - start_y, 
				  length_includes_head=True, head_width=0.25, color='r')
		else:
			ax.arrow(start_x, start_y, succ_x - start_x, succ_y - start_y, 
				  length_includes_head=True, head_width=0.25, color=gain_colours[tdx])

	## MEVB
	# Plot MEVB by shading arrows
	ax = axes[2]
	ax.set_title('EVB')
	gridworld.draw(use_reachability=True, ax=ax)

	# Grab boundaries
	min_MEVB = np.min(MEVB) 
	max_MEVB = np.max(MEVB)

	if 'min_MEVB' in params.keys():
		min_MEVB = params['min_MEVB']
	if 'max_MEVB' in params.keys():
		max_MEVB = params['max_MEVB']

	norm_MEVB = mpl.colors.Normalize(vmin=min_MEVB, vmax=max_MEVB)(MEVB)
	MEVB_colours = plt.cm.winter(norm_MEVB).reshape(-1, 4)
	MEVB_colours[:, 3] = norm_MEVB / alpha_fac
	CENTRE_OFFSET = 0.5 # oned_twod gives the coordinate of the top left corner of the state
	for tdx, transition in enumerate(transitions):
		s_k, a_k, s_kp = transition
		start_y, start_x = np.array(oned_twod(s_k, gridworld.width, gridworld.height)) + CENTRE_OFFSET
		succ_y, succ_x = np.array(oned_twod(s_kp, gridworld.width, gridworld.height)) + CENTRE_OFFSET

		# Plot
		if specials is not None and transition in specials: # Custom distinction for a set of transitions
			ax.arrow(start_x, start_y, succ_x - start_x, succ_y - start_y, 
				  length_includes_head=True, head_width=0.25, color='k')
		elif abs(np.max(norm_MEVB) - norm_MEVB[tdx]) < 1e-8: # Distinguish the eventually-chosen transition
			ax.arrow(start_x, start_y, succ_x - start_x, succ_y - start_y, 
					  length_includes_head=True, head_width=0.25, color='r')
		else:
			ax.arrow(start_x, start_y, succ_x - start_x, succ_y - start_y, 
					  length_includes_head=True, head_width=0.25, color=MEVB_colours[tdx])

if __name__ == '__main__':
	## Set up parameters
	# Arena parameters
	width = 10
	height = 7
	goal_states = np.array([width - 1, (height - 1) * width, height * width - 1]) # Non-start corners

	# GridWorld parameters
	stoch = 0 # Grid stochasticity
	num_states = width * height
	num_actions = 4

	init_state_distribution = np.zeros(num_states)
	init_state_distribution[0] = 1

	# Agent parameters
	alpha = 1.0
	gamma = 0.95
	epsilon = 0.05

	# Create GridWorld
	# Get GridWorld and associated objects
	gw = GridWorld(width, height, stoch, init_state_distribution=init_state_distribution)
	all_experiences = gw.get_all_transitions()
	T = gw.transitions

	# Set up agent
	ga = GeodesicAgent(num_states, num_actions, goal_states, T, alpha=alpha, gamma=gamma)
	ga.remember(all_experiences) # Pre-load our agent with all possible memories

	# Set up storage variables
	num_replay_steps = 10
	needs = np.zeros((num_replay_steps, num_states, num_states))
	all_MEVBs = np.zeros((num_replay_steps, len(ga.memory)))

	# Begin replay computation
	goal_dist = ga.goal_dist
	replayed_experiences = []
	curr_replay_seq = []
	for replay_step in range(num_replay_steps):
	    MEVBs = np.zeros(len(ga.memory)) # Each update is picked greedily
	    for gdx, goal_state in enumerate(goal_states):

	        # Get policy for this state
	        if goal_state in goal_states: # If we already have this on hand, use it
	            policy = ga.policies[goal_state]
	        else: # Otherwise, recompute entirely
	            policy = ga.derive_policy(goal_state) 

	        # Derive resultant state occupancy matrix
	        M_pi = ga.compute_occupancy(policy, T)
	        needs[replay_step, :, :] = M_pi # Save

	        # Compute EVB for each possible transition wrt this goal
	        for tdx, transition in enumerate(ga.memory):
	            if not curr_replay_seq or ga.creates_loop(transition, curr_replay_seq):
	                evb = ga.compute_EVB(transition, goal_state, policy, state=ga.curr_state, M=M_pi)
	            else:
	                evb = ga.compute_multistep_EVB(transition, goal_state, policy, 
	                                               curr_replay_seq, state=ga.curr_state, M=M_pi)

	            MEVBs[tdx] += goal_dist[gdx] * evb

	    # Save
	    all_MEVBs[replay_step, :] = MEVBs
	    
	    # Optimise
	    best_memory = ga.memory[np.argmax(MEVBs)]
	    replayed_experiences.append(best_memory)
	    s_k, a_k, s_kp = best_memory

	    # Evolve current replay list
	    # If most recent replay exists, and s_k continues it, and does not create a loop...
	    if curr_replay_seq and s_k == curr_replay_seq[-1][-1] \
	       and not ga.creates_loop(best_memory, curr_replay_seq): 

	        ak_best_anywhere = False 
	        for goal_state in goal_states:
	            if a_k == np.argmax(ga.G[s_k, :, goal_state]):
	                ak_best_anywhere = True
	                break

	        if ak_best_anywhere: # ... and a_k is the best action for any given policy in s_k, add to seq
	            curr_replay_seq.append(best_memory)
	        else:
	            curr_replay_seq = [best_memory]
	    else:
	        curr_replay_seq = [best_memory]

	    # Learn and update policy
	    if len(curr_replay_seq) == 1:
	        ga.learn(curr_replay_seq)
	    else:
	        ga.multi_learn(curr_replay_seq)

	gain = MEVB_to_gain(all_MEVBs[0, :], needs[0, :, :], ga.memory, 0)
	plot_need_gain(gw, ga.memory, needs[0, 0, :], gain, all_MEVBs[0, :])
	plt.show()



