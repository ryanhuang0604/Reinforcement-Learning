# Spring 2021, IOC 5269 Reinforcement Learning
# HW1-PartII: First-Visit Monte-Carlo and Temporal-difference policy evaluation

import gym
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict

env = gym.make("Blackjack-v0")

def mc_policy_evaluation(policy, env, num_episodes, gamma=1.0):
	"""
		Find the value function for a given policy using first-visit Monte-Carlo sampling
		
		Input Arguments
		----------
			policy: 
				a function that maps a state to action probabilities
			env:
				an OpenAI gym environment
			num_episodes: int
				the number of episodes to sample
			gamma: float
				the discount factor
		----------
		
		Output
		----------
			V: dict (that maps from state -> value)
		----------
	
		TODOs
		----------
			1. Initialize the value function
			2. Sample an episode and calculate sample returns
			3. Iterate and update the value function
		----------
		
	"""
	
	##### FINISH TODOS HERE #####
	
	# 1. Initialize the value function
	# value function
	V = defaultdict(float)
	N = defaultdict(int)
	
	for i in range(num_episodes) : 
		# initialize the list for storing states, actions, rewards
		states = []
		actions = []
		rewards = []
		
		# reset the environment
		observation = env.reset()
		
		while True: 
			# append the states to the states list
			states.append(observation)
			
			# select an action using apply_policy function
			action = policy(observation)
			actions.append(action)
			
			# perform the action in the environment
			observation, reward, done, info = env.step(action)
			rewards.append(reward)
			
			# if done, break
			if done:
				break
					
		# find the sum of rewards for each states and compute V
		G = 0
		returns = 0
		for t in range(len(states) - 1, -1, -1):
			st = states[t]
			rt = rewards[t]
			returns += rt
			if st not in states[:t]: 
				N[st] += 1
				V[st] += (returns - V[st]) / N[st]
				'''
				G_i = gamma * rt
				G = V[st] * N[st] + G_i
				V[st] = G / N[st]
				'''
			returns *= gamma

	# normalize V
	max_value = max([abs(V[value]) for value in V])
	for value in V:
		V[value] = V[value] / max_value
	
	#############################
	
	return V


def td0_policy_evaluation(policy, env, num_episodes, gamma=1.0):
	"""
		Find the value function for the given policy using TD(0)
	
		Input Arguments
		----------
			policy: 
				a function that maps a state to action probabilities
			env:
				an OpenAI gym environment
			num_episodes: int
				the number of episodes to sample
			gamma: float
				the discount factor
		----------
	
		Output
		----------
			V: dict (that maps from state -> value)
		----------
		
		TODOs
		----------
			1. Initialize the value function
			2. Sample an episode and calculate TD errors
			3. Iterate and update the value function
		----------
	"""

	##### FINISH TODOS HERE #####
	
	# 1. Initialize the value function
	# value function
	V = defaultdict(float)
	# for compute V
	alpha = 0.01
	
	for i in range(num_episodes) : 
		
		# reset the environment
		observation = env.reset()
	
		# compute V state by state
		while True: 
			action = policy(observation)
				
			observation_, reward, done, info = env.step(action)

			s = observation
			s_ = observation_
			V[s] = V[s] + alpha * (reward + gamma * V[s_] - V[s])
			observation = observation_

			# if done, break
			if done: 
				break

	# normalize V
	max_value = max([abs(V[value]) for value in V])
	for value in V:
		V[value] = V[value] / max_value
	
	#############################

	return V

	

def plot_value_function(V, title="Value Function"):
	"""
		Plots the value function as a surface plot.
		(Credit: Denny Britz)
	"""
	min_x = min(k[0] for k in V.keys())
	max_x = max(k[0] for k in V.keys())
	min_y = min(k[1] for k in V.keys())
	max_y = max(k[1] for k in V.keys())

	x_range = np.arange(min_x, max_x + 1)
	y_range = np.arange(min_y, max_y + 1)
	X, Y = np.meshgrid(x_range, y_range)

	# Find value for all (x, y) coordinates
	Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
	Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))

	def plot_surface(X, Y, Z, title):
		fig = plt.figure(figsize=(20, 10))
		ax = fig.add_subplot(111, projection='3d')
		surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
							   cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
		ax.set_xlabel('Player Sum')
		ax.set_ylabel('Dealer Showing')
		ax.set_zlabel('Value')
		ax.set_title(title)
		ax.view_init(ax.elev, -120)
		fig.colorbar(surf)
		plt.show()

	plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))
	plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))
	
	
def apply_policy(observation):
	"""
		A policy under which one will stick if the sum of cards is >= 20 and hit otherwise.
	"""
	score, dealer_score, usable_ace = observation
	return 0 if score >= 20 else 1


if __name__ == '__main__':
	V_mc_10k = mc_policy_evaluation(apply_policy, env, num_episodes=10000)
	plot_value_function(V_mc_10k, title="10,000 Steps")
	V_mc_500k = mc_policy_evaluation(apply_policy, env, num_episodes=500000)
	plot_value_function(V_mc_500k, title="500,000 Steps")


	V_td0_10k = td0_policy_evaluation(apply_policy, env, num_episodes=10000)
	plot_value_function(V_td0_10k, title="10,000 Steps")
	V_td0_500k = td0_policy_evaluation(apply_policy, env, num_episodes=500000)
	plot_value_function(V_td0_500k, title="500,000 Steps")