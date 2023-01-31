import numpy as np
import matplotlib.pyplot as plt
import random
import gym
import mdptoolbox.example
import mdp
import openai
import pandas as pd
from pprint import pprint

# setup the randoms tate
RANDOM_STATE = 545510477

def forest():
	'''
	https://pymdptoolbox.readthedocs.io/en/latest/api/example.html
	'''
	# 1500 states (big MDP)
	print("Forest - Value Iteration")
	#value_iter_forest()
	print("Forest - Policy Iteration")	
	#policy_iter_forest()
	print("Forest - Q Learning")	
	q_learning_forest()

def frozen_lake():
	'''
	https://gym.openai.com/envs/FrozenLake8x8-v0/
	'''
	# 64 states (small MDP)
	#env = gym.make("FrozenLake8x8-v0").env
	print("FrozenLake - Value Iteration")
	#value_iter_frozen_lake()
	print("FrozenLake - Policy Iteration")	
	#policy_iter_frozen_lake()
	print("FrozenLake - Q Learning")
	q_learning_frozen_lake()


def value_iter_frozen_lake():
	fl4 = openai.OpenAI_MDPToolbox("FrozenLake-v0", s=4)
	fl8 = openai.OpenAI_MDPToolbox("FrozenLake-v0", s=8)
	fl16 = openai.OpenAI_MDPToolbox("FrozenLake-v0", s=16)	

	state_space = [fl4, fl8, fl16]
	#discount = [0.15, 0.3, 0.45, 0.6, 0.75, 0.9]	
	discount = [0.9, 0.92, 0.94, 0.96, 0.98]	
	iters_all_discount = []
	time_all_discount = []
	maxutil_all_discount = []
	avgutil_all_discount = []
	error_all_discount = []
	iters_all_state = []	
	time_all_state = []	
	maxutil_all_state = []
	avgutil_all_state = []
	error_all_state = []

	#Tune by Discount Factor

	for d in discount:
		print("Discount Factor: " + str(d))
		vi = mdp.ValueIteration(fl4.P, fl4.R, d)
		vi.max_iter = 60
		#vi.setVerbose()
		vi.run()

		run_stats_dict = pd.DataFrame.from_dict(vi.run_stats)
		iters = run_stats_dict['Iteration']
		maxutil = run_stats_dict['Max V']
		avgutil = run_stats_dict['Mean V']
		error = run_stats_dict['Error']		
		time = run_stats_dict['Time']

		iters_all_discount.append(iters)
		maxutil_all_discount.append(maxutil)
		avgutil_all_discount.append(avgutil)
		error_all_discount.append(error)
		time_all_discount.append(time)	

	# for s in state_space:
	# 	print("State Space: " + str(s.states))
	# 	vi = mdp.ValueIteration(s.P, s.R, 0.98)
	# 	vi.max_iter = 1000
	# 	#vi.setVerbose()
	# 	vi.run()

	# 	run_stats_dict = pd.DataFrame.from_dict(vi.run_stats)
	# 	iters = run_stats_dict['Iteration']
	# 	maxutil = run_stats_dict['Max V']
	# 	avgutil = run_stats_dict['Mean V']
	# 	error = run_stats_dict['Error']		
	# 	time = run_stats_dict['Time']

	# 	iters_all_state.append(iters)
	# 	maxutil_all_state.append(maxutil)
	# 	avgutil_all_state.append(avgutil)
	# 	error_all_state.append(error)
	# 	time_all_state.append(time)	


	plt.clf()
	plt.plot(iters_all_discount[0], time_all_discount[0], label='DiscountFactor=0.90')
	plt.plot(iters_all_discount[1], time_all_discount[1], label='DiscountFactor=0.92')
	plt.plot(iters_all_discount[2], time_all_discount[2], label='DiscountFactor=0.94')
	plt.plot(iters_all_discount[3], time_all_discount[3], label='DiscountFactor=0.96')
	plt.plot(iters_all_discount[4], time_all_discount[4], label='DiscountFactor=0.98')	
	plt.xlabel('Iteration #')
	plt.ylabel('CPU Time (s)')
	plt.title('CPU Time for VI')	
	plt.legend()
	plt.savefig('FrozenLake_VI_CPUTime_DiscountFactor')

	plt.clf()
	plt.plot(iters_all_discount[0], maxutil_all_discount[0], label='DiscountFactor=0.90')
	plt.plot(iters_all_discount[1], maxutil_all_discount[1], label='DiscountFactor=0.92')
	plt.plot(iters_all_discount[2], maxutil_all_discount[2], label='DiscountFactor=0.94')
	plt.plot(iters_all_discount[3], maxutil_all_discount[3], label='DiscountFactor=0.96')
	plt.plot(iters_all_discount[4], maxutil_all_discount[4], label='DiscountFactor=0.98')	
	plt.xlabel('Iteration #')
	plt.ylabel('Max Utility')
	plt.title('MaxUtil/Iters for VI')	
	plt.legend()
	plt.savefig('FrozenLake_VI_MaxUtil_DiscountFactor')

	plt.clf()
	plt.plot(iters_all_discount[0], avgutil_all_discount[0], label='DiscountFactor=0.90')
	plt.plot(iters_all_discount[1], avgutil_all_discount[1], label='DiscountFactor=0.92')
	plt.plot(iters_all_discount[2], avgutil_all_discount[2], label='DiscountFactor=0.94')
	plt.plot(iters_all_discount[3], avgutil_all_discount[3], label='DiscountFactor=0.96')
	plt.plot(iters_all_discount[4], avgutil_all_discount[4], label='DiscountFactor=0.98')
	plt.xlabel('Iteration #')
	plt.ylabel('Average Utility')
	plt.title('AvgUtil/Iters for VI')	
	plt.legend()
	plt.savefig('FrozenLake_VI_AvgUtil_DiscountFactor')

	plt.clf()
	plt.plot(iters_all_discount[0], error_all_discount[0], label='DiscountFactor=0.90')
	plt.plot(iters_all_discount[1], error_all_discount[1], label='DiscountFactor=0.92')
	plt.plot(iters_all_discount[2], error_all_discount[2], label='DiscountFactor=0.94')
	plt.plot(iters_all_discount[3], error_all_discount[3], label='DiscountFactor=0.96')
	plt.plot(iters_all_discount[4], error_all_discount[4], label='DiscountFactor=0.98')
	plt.xlabel('Iteration #')
	plt.ylabel('Error')
	plt.title('Error/Iters for VI')	
	plt.legend()
	plt.savefig('FrozenLake_VI_Error_DiscountFactor')		

	# plt.clf()
	# plt.plot(iters_all_state[0], time_all_state[0], label='State Space=16')
	# plt.plot(iters_all_state[1], time_all_state[1], label='State Space=64')
	# plt.plot(iters_all_state[2], time_all_state[2], label='State Space=256')
	# plt.xlabel('Iteration #')
	# plt.ylabel('CPU Time (s)')
	# plt.title('CPU Time for VI')	
	# plt.legend()
	# plt.savefig('FrozenLake_VI_CPUTime_StateSpace')

	# plt.clf()
	# plt.plot(iters_all_state[0], maxutil_all_state[0], label='State Space=16')
	# plt.plot(iters_all_state[1], maxutil_all_state[1], label='State Space=64')
	# plt.plot(iters_all_state[2], maxutil_all_state[2], label='State Space=256')
	# plt.xlabel('Iteration #')
	# plt.ylabel('Max Utility')
	# plt.title('MaxUtil/Iters for VI')	
	# plt.legend()
	# plt.savefig('FrozenLake_VI_MaxUtil_StateSpace')

	# plt.clf()
	# plt.plot(iters_all_state[0], avgutil_all_state[0], label='State Space=16')
	# plt.plot(iters_all_state[1], avgutil_all_state[1], label='State Space=64')
	# plt.plot(iters_all_state[2], avgutil_all_state[2], label='State Space=256')
	# plt.xlabel('Iteration #')
	# plt.ylabel('Average Utility')
	# plt.title('AvgUtil/Iters for VI')	
	# plt.legend()
	# plt.savefig('FrozenLake_VI_AvgUtil_StateSpace')

	# plt.clf()
	# plt.plot(iters_all_state[0], error_all_state[0], label='State Space=16')
	# plt.plot(iters_all_state[1], error_all_state[1], label='State Space=64')
	# plt.plot(iters_all_state[2], error_all_state[2], label='State Space=256')
	# plt.xlabel('Iteration #')
	# plt.ylabel('Error')
	# plt.title('Error/Iters for VI')	
	# plt.legend()
	# plt.savefig('FrozenLake_VI_Error_StateSpace')		

def policy_iter_frozen_lake():
	fl4 = openai.OpenAI_MDPToolbox("FrozenLake-v0", s=4)
	fl8 = openai.OpenAI_MDPToolbox("FrozenLake-v0", s=8)
	fl16 = openai.OpenAI_MDPToolbox("FrozenLake-v0", s=16)	

	state_space = [fl4, fl8, fl16]
	#discount = [0.15, 0.3, 0.45, 0.6, 0.75, 0.9]	
	discount = [0.9, 0.92, 0.94, 0.96, 0.98]	
	iters_all_discount = []
	time_all_discount = []
	maxutil_all_discount = []
	avgutil_all_discount = []
	error_all_discount = []
	iters_all_state = []	
	time_all_state = []	
	maxutil_all_state = []
	avgutil_all_state = []
	error_all_state = []

	# Tune by Discount Factor

	for d in discount:
		print("Discount Factor: " + str(d))
		pi = mdp.PolicyIteration(fl8.P, fl8.R, d)
		pi.max_iter = 60
		#vi.setVerbose()
		pi.run()

		run_stats_dict = pd.DataFrame.from_dict(pi.run_stats)
		iters = run_stats_dict['Iteration']
		maxutil = run_stats_dict['Max V']
		avgutil = run_stats_dict['Mean V']
		error = run_stats_dict['Error']		
		time = run_stats_dict['Time']

		iters_all_discount.append(iters)
		maxutil_all_discount.append(maxutil)
		avgutil_all_discount.append(avgutil)
		error_all_discount.append(error)
		time_all_discount.append(time)	

	for s in state_space:
		print("State Space: " + str(s.states))
		pi = mdp.PolicyIteration(s.P, s.R, 0.98)
		pi.max_iter = 1000
		#vi.setVerbose()
		pi.run()

		run_stats_dict = pd.DataFrame.from_dict(pi.run_stats)
		iters = run_stats_dict['Iteration']
		maxutil = run_stats_dict['Max V']
		avgutil = run_stats_dict['Mean V']
		error = run_stats_dict['Error']		
		time = run_stats_dict['Time']

		iters_all_state.append(iters)
		maxutil_all_state.append(maxutil)
		avgutil_all_state.append(avgutil)
		error_all_state.append(error)
		time_all_state.append(time)	


	plt.clf()
	plt.plot(iters_all_discount[0], time_all_discount[0], label='DiscountFactor=0.90')
	plt.plot(iters_all_discount[1], time_all_discount[1], label='DiscountFactor=0.92')
	plt.plot(iters_all_discount[2], time_all_discount[2], label='DiscountFactor=0.94')
	plt.plot(iters_all_discount[3], time_all_discount[3], label='DiscountFactor=0.96')
	plt.plot(iters_all_discount[4], time_all_discount[4], label='DiscountFactor=0.98')	
	plt.xlabel('Iteration #')
	plt.ylabel('CPU Time (s)')
	plt.title('CPU Time for PI')	
	plt.legend()
	plt.savefig('FrozenLake_PI_CPUTime_DiscountFactor')

	plt.clf()
	plt.plot(iters_all_discount[0], maxutil_all_discount[0], label='DiscountFactor=0.90')
	plt.plot(iters_all_discount[1], maxutil_all_discount[1], label='DiscountFactor=0.92')
	plt.plot(iters_all_discount[2], maxutil_all_discount[2], label='DiscountFactor=0.94')
	plt.plot(iters_all_discount[3], maxutil_all_discount[3], label='DiscountFactor=0.96')
	plt.plot(iters_all_discount[4], maxutil_all_discount[4], label='DiscountFactor=0.98')	
	plt.xlabel('Iteration #')
	plt.ylabel('Max Utility')
	plt.title('MaxUtil/Iters for PI')	
	plt.legend()
	plt.savefig('FrozenLake_PI_MaxUtil_DiscountFactor')

	plt.clf()
	plt.plot(iters_all_discount[0], avgutil_all_discount[0], label='DiscountFactor=0.90')
	plt.plot(iters_all_discount[1], avgutil_all_discount[1], label='DiscountFactor=0.92')
	plt.plot(iters_all_discount[2], avgutil_all_discount[2], label='DiscountFactor=0.94')
	plt.plot(iters_all_discount[3], avgutil_all_discount[3], label='DiscountFactor=0.96')
	plt.plot(iters_all_discount[4], avgutil_all_discount[4], label='DiscountFactor=0.98')
	plt.xlabel('Iteration #')
	plt.ylabel('Average Utility')
	plt.title('AvgUtil/Iters for PI')	
	plt.legend()
	plt.savefig('FrozenLake_PI_AvgUtil_DiscountFactor')

	plt.clf()
	plt.plot(iters_all_discount[0], error_all_discount[0], label='DiscountFactor=0.90')
	plt.plot(iters_all_discount[1], error_all_discount[1], label='DiscountFactor=0.92')
	plt.plot(iters_all_discount[2], error_all_discount[2], label='DiscountFactor=0.94')
	plt.plot(iters_all_discount[3], error_all_discount[3], label='DiscountFactor=0.96')
	plt.plot(iters_all_discount[4], error_all_discount[4], label='DiscountFactor=0.98')
	plt.xlabel('Iteration #')
	plt.ylabel('Error')
	plt.title('Error/Iters for PI')	
	plt.legend()
	plt.savefig('FrozenLake_PI_Error_DiscountFactor')		

	plt.clf()
	plt.plot(iters_all_state[0], time_all_state[0], label='State Space=16')
	plt.plot(iters_all_state[1], time_all_state[1], label='State Space=64')
	plt.plot(iters_all_state[2], time_all_state[2], label='State Space=256')
	plt.xlabel('Iteration #')
	plt.ylabel('CPU Time (s)')
	plt.title('CPU Time for PI')	
	plt.legend()
	plt.savefig('FrozenLake_PI_CPUTime_StateSpace')

	plt.clf()
	plt.plot(iters_all_state[0], maxutil_all_state[0], label='State Space=16')
	plt.plot(iters_all_state[1], maxutil_all_state[1], label='State Space=64')
	plt.plot(iters_all_state[2], maxutil_all_state[2], label='State Space=256')
	plt.xlabel('Iteration #')
	plt.ylabel('Max Utility')
	plt.title('MaxUtil/Iters for PI')	
	plt.legend()
	plt.savefig('FrozenLake_PI_MaxUtil_StateSpace')

	plt.clf()
	plt.plot(iters_all_state[0], avgutil_all_state[0], label='State Space=16')
	plt.plot(iters_all_state[1], avgutil_all_state[1], label='State Space=64')
	plt.plot(iters_all_state[2], avgutil_all_state[2], label='State Space=256')
	plt.xlabel('Iteration #')
	plt.ylabel('Average Utility')
	plt.title('AvgUtil/Iters for PI')	
	plt.legend()
	plt.savefig('FrozenLake_PI_AvgUtil_StateSpace')

	plt.clf()
	plt.plot(iters_all_state[0], error_all_state[0], label='State Space=16')
	plt.plot(iters_all_state[1], error_all_state[1], label='State Space=64')
	plt.plot(iters_all_state[2], error_all_state[2], label='State Space=256')
	plt.xlabel('Iteration #')
	plt.ylabel('Error')
	plt.title('Error/Iters for PI')	
	plt.legend()
	plt.savefig('FrozenLake_PI_Error_StateSpace')	

def q_learning_frozen_lake():

	RANDOM_STATE = 123456

	fl4 = openai.OpenAI_MDPToolbox("FrozenLake-v0", s=4)

	time = []
	reward_discount = []
	reward_epsilon = []
	reward_alpha = []
	epsilon = [0.15, 0.3, 0.45, 0.6, 0.75, 0.9]
	epsilon_decay = [.9, 0.92, 0.94, 0.96, 0.99]
	discount = [0.9, 0.92, 0.94, 0.96, 0.98]
	alpha = [0.2, 0.4, 0.6, 0.8]
	mean_dis = []

	iters_all_discount = []
	time_all_discount = []
	maxutil_all_discount = []
	avgutil_all_discount = []
	error_all_discount = []
	iters_all_epsilon = []
	time_all_epsilon = []
	maxutil_all_epsilon = []
	avgutil_all_epsilon = []
	error_all_epsilon = []	
	iters_all_alpha = []
	time_all_alpha = []
	maxutil_all_alpha = []
	avgutil_all_alpha = []
	error_all_alpha = []	
	iters_all_ed = []
	time_all_ed = []
	maxutil_all_ed = []
	avgutil_all_ed  = []
	error_all_ed = []		

	for d in discount:
		print("Discount Factor: " + str(d))
		q = mdp.QLearning(fl4.P, fl4.R, gamma=d, epsilon=0.6, n_iter=500000, run_stat_frequency=1000)
		q.run()

		run_stats_dict = pd.DataFrame.from_dict(q.run_stats)
		print(run_stats_dict)
		iters = run_stats_dict['Iteration']
		maxutil = run_stats_dict['Max V']
		avgutil = run_stats_dict['Mean V']
		error = run_stats_dict['Error']		
		time = run_stats_dict['Time']	
		# time = q.time_mean
		# avgutil = q.v_mean
		# avgerror = q.error_mean
		# episodes = len(q.time_mean)

		# iters_all_discount.append(range(1,episodes+1))
		iters_all_discount.append(range(1,iters.size+1))
		maxutil_all_discount.append(maxutil)
		avgutil_all_discount.append(avgutil)
		error_all_discount.append(error)
		time_all_discount.append(time)		

	# for e in epsilon:
	# 	print("Epsilon: " + str(e))
	# 	q = mdp.QLearning(fl4.P, fl4.R, gamma=0.98, epsilon=e, n_iter=500000, run_stat_frequency=1000)
	# 	random.seed(RANDOM_STATE)
	# 	q.run()
		
	# 	run_stats_dict = pd.DataFrame.from_dict(q.run_stats)
	# 	print(run_stats_dict)
	# 	iters = run_stats_dict['Iteration']
	# 	maxutil = run_stats_dict['Max V']
	# 	avgutil = run_stats_dict['Mean V']
	# 	error = run_stats_dict['Error']
	# 	time = run_stats_dict['Time']

	# 	#iters_all_epsilon.append(iters)
	# 	iters_all_epsilon.append(range(1,iters.size+1))
	# 	maxutil_all_epsilon.append(maxutil)
	# 	avgutil_all_epsilon.append(avgutil)
	# 	error_all_epsilon.append(error)
	# 	time_all_epsilon.append(time)	

	# for a in alpha:
	# 	print("Alpha: " + str(a))
	# 	q = mdp.QLearning(fl4.P, fl4.R, gamma=0.92, epsilon=0.9, alpha=a, n_iter=500000, run_stat_frequency=1000)
	# 	q.run()
		
	# 	run_stats_dict = pd.DataFrame.from_dict(q.run_stats)
	# 	print(run_stats_dict)
	# 	iters = run_stats_dict['Iteration']
	# 	maxutil = run_stats_dict['Max V']
	# 	avgutil = run_stats_dict['Mean V']
	# 	error = run_stats_dict['Error']		
	# 	time = run_stats_dict['Time']

	# 	#iters_all_alpha.append(iters)
	# 	iters_all_alpha.append(range(1,iters.size+1))
	# 	maxutil_all_alpha.append(maxutil)
	# 	avgutil_all_alpha.append(avgutil)
	# 	error_all_alpha.append(error)
	# 	time_all_alpha.append(time)	

	# for ed in epsilon_decay:
	# 	print("Epsilon Decay: " + str(ed))
	# 	q = mdp.QLearning(P, R, gamma=0.98, epsilon=0.6, alpha=0.4, epsilon_decay=ed, n_iter=500000)
	# 	q.run()
		
	# 	run_stats_dict = pd.DataFrame.from_dict(q.run_stats)
	# 	iters = run_stats_dict['Iteration']
	# 	maxutil = run_stats_dict['Max V']
	# 	avgutil = run_stats_dict['Mean V']
	# 	error = run_stats_dict['Error']		
	# 	time = run_stats_dict['Time']

	# 	iters_all_ed.append(iters)
	# 	maxutil_all_ed.append(maxutil)
	# 	avgutil_all_ed.append(avgutil)
	# 	error_all_ed.append(error)
	# 	time_all_ed.append(time)			


	plt.clf()
	plt.plot(iters_all_discount[0], time_all_discount[0], label='DiscountFactor=0.90')
	plt.plot(iters_all_discount[1], time_all_discount[1], label='DiscountFactor=0.92')
	plt.plot(iters_all_discount[2], time_all_discount[2], label='DiscountFactor=0.94')
	plt.plot(iters_all_discount[3], time_all_discount[3], label='DiscountFactor=0.96')
	plt.plot(iters_all_discount[4], time_all_discount[4], label='DiscountFactor=0.98')
	plt.xlabel('Episode #')
	plt.ylabel('CPU Time (s)')
	plt.title('CPU Time/Episode for QLearning')	
	plt.legend()
	plt.savefig('FrozenLake_Q_AvgCPUTime_DiscountFactor')

	plt.clf()
	plt.plot(iters_all_discount[0], maxutil_all_discount[0], label='DiscountFactor=0.90')
	plt.plot(iters_all_discount[1], maxutil_all_discount[1], label='DiscountFactor=0.92')
	plt.plot(iters_all_discount[2], maxutil_all_discount[2], label='DiscountFactor=0.94')
	plt.plot(iters_all_discount[3], maxutil_all_discount[3], label='DiscountFactor=0.96')
	plt.plot(iters_all_discount[4], maxutil_all_discount[4], label='DiscountFactor=0.98')
	plt.xlabel('Episode #')
	plt.ylabel('Max Utility')
	plt.title('MaxUtil/Episode for QLearning')	
	plt.legend()
	plt.savefig('FrozenLake_Q_MaxUtil_DiscountFactor')


	plt.clf()
	plt.plot(iters_all_discount[0], error_all_discount[0], label='DiscountFactor=0.90')
	plt.plot(iters_all_discount[1], error_all_discount[1], label='DiscountFactor=0.92')
	plt.plot(iters_all_discount[2], error_all_discount[2], label='DiscountFactor=0.94')
	plt.plot(iters_all_discount[3], error_all_discount[3], label='DiscountFactor=0.96')
	plt.plot(iters_all_discount[4], error_all_discount[4], label='DiscountFactor=0.98')
	plt.xlabel('Episode #')
	plt.ylabel('Error')
	plt.title('Error/Episode for QLearning')	
	plt.legend()
	plt.savefig('FrozenLake_Q_AvgError_DiscountFactor')

	plt.clf()
	plt.plot(iters_all_discount[0], avgutil_all_discount[0], label='DiscountFactor=0.90')
	plt.plot(iters_all_discount[1], avgutil_all_discount[1], label='DiscountFactor=0.92')
	plt.plot(iters_all_discount[2], avgutil_all_discount[2], label='DiscountFactor=0.94')
	plt.plot(iters_all_discount[3], avgutil_all_discount[3], label='DiscountFactor=0.96')
	plt.plot(iters_all_discount[4], avgutil_all_discount[4], label='DiscountFactor=0.98')
	plt.xlabel('Episode #')
	plt.ylabel('Average Utility')
	plt.title('AvgUtil/Episode for QLearning')	
	plt.legend()
	plt.savefig('FrozenLake_Q_AvgUtil_DiscountFactor')


	# plt.clf()
	# plt.plot(iters_all_epsilon[0], time_all_epsilon[0], label='Epsilon=0.15')
	# plt.plot(iters_all_epsilon[1], time_all_epsilon[1], label='Epsilon=0.30')
	# plt.plot(iters_all_epsilon[2], time_all_epsilon[2], label='Epsilon=0.45')
	# plt.plot(iters_all_epsilon[3], time_all_epsilon[3], label='Epsilon=0.60')
	# plt.plot(iters_all_epsilon[4], time_all_epsilon[4], label='Epsilon=0.75')
	# plt.plot(iters_all_epsilon[5], time_all_epsilon[5], label='Epsilon=0.90')
	# plt.xlabel('Episode #')
	# plt.ylabel('CPU Time (s)')
	# plt.title('CPU Time for QLearning')	
	# plt.legend()
	# plt.savefig('FrozenLake_Q_CPUTime_Epsilon')

	# plt.clf()
	# plt.plot(iters_all_epsilon[0], maxutil_all_epsilon[0], label='Epsilon=0.15')
	# plt.plot(iters_all_epsilon[1], maxutil_all_epsilon[1], label='Epsilon=0.30')
	# plt.plot(iters_all_epsilon[2], maxutil_all_epsilon[2], label='Epsilon=0.45')
	# plt.plot(iters_all_epsilon[3], maxutil_all_epsilon[3], label='Epsilon=0.60')
	# plt.plot(iters_all_epsilon[4], maxutil_all_epsilon[4], label='Epsilon=0.75')
	# plt.plot(iters_all_epsilon[5], maxutil_all_epsilon[5], label='Epsilon=0.90')
	# plt.xlabel('Episode #')
	# plt.ylabel('Max Utility')
	# plt.title('MaxUtil/Iters for QLearning')	
	# plt.legend()
	# plt.savefig('FrozenLake_Q_MaxUtil_Epsilon')

	# plt.clf()
	# plt.plot(iters_all_epsilon[0], avgutil_all_epsilon[0], label='Epsilon=0.15')
	# plt.plot(iters_all_epsilon[1], avgutil_all_epsilon[1], label='Epsilon=0.30')
	# plt.plot(iters_all_epsilon[2], avgutil_all_epsilon[2], label='Epsilon=0.45')
	# plt.plot(iters_all_epsilon[3], avgutil_all_epsilon[3], label='Epsilon=0.60')
	# plt.plot(iters_all_epsilon[4], avgutil_all_epsilon[4], label='Epsilon=0.75')
	# plt.plot(iters_all_epsilon[5], avgutil_all_epsilon[5], label='Epsilon=0.90')
	# plt.xlabel('Episode #')
	# plt.ylabel('Average Utility')
	# plt.title('AvgUtil/Iters for QLearning')	
	# plt.legend()
	# plt.savefig('FrozenLake_Q_AvgUtil_Epsilon')

	# plt.clf()
	# plt.plot(iters_all_epsilon[0], error_all_epsilon[0], label='Epsilon=0.15')
	# plt.plot(iters_all_epsilon[1], error_all_epsilon[1], label='Epsilon=0.30')
	# plt.plot(iters_all_epsilon[2], error_all_epsilon[2], label='Epsilon=0.45')
	# plt.plot(iters_all_epsilon[3], error_all_epsilon[3], label='Epsilon=0.60')
	# plt.plot(iters_all_epsilon[4], error_all_epsilon[4], label='Epsilon=0.75')
	# plt.plot(iters_all_epsilon[5], error_all_epsilon[5], label='Epsilon=0.90')
	# plt.xlabel('Episode #')
	# plt.ylabel('Avg Error')
	# plt.title('Avg Error/Iters for QLearning')	
	# plt.legend()
	# plt.savefig('FrozenLake_Q_Error_Epsilon')

	# plt.clf()
	# plt.plot(iters_all_alpha[0], time_all_alpha[0], label='Alpha=0.20')
	# plt.plot(iters_all_alpha[1], time_all_alpha[1], label='Alpha=0.40')
	# plt.plot(iters_all_alpha[2], time_all_alpha[2], label='Alpha=0.60')
	# plt.plot(iters_all_alpha[3], time_all_alpha[3], label='Alpha=0.80')
	# plt.xlabel('Episode #')
	# plt.ylabel('CPU Time (s)')
	# plt.title('CPU Time for QLearning')	
	# plt.legend()
	# plt.savefig('Forest_Q_CPUTime_Alpha')

	# plt.clf()
	# plt.plot(iters_all_alpha[0], maxutil_all_alpha[0], label='Alpha=0.20')
	# plt.plot(iters_all_alpha[1], maxutil_all_alpha[1], label='Alpha=0.40')
	# plt.plot(iters_all_alpha[2], maxutil_all_alpha[2], label='Alpha=0.60')
	# plt.plot(iters_all_alpha[3], maxutil_all_alpha[3], label='Alpha=0.80')
	# plt.xlabel('Episode #')
	# plt.ylabel('Max Utility')
	# plt.title('MaxUtil/Iters for QLearning')	
	# plt.legend()
	# plt.savefig('Forest_Q_MaxUtil_Alpha')

	# plt.clf()
	# plt.plot(iters_all_alpha[0], avgutil_all_alpha[0], label='Alpha=0.20')
	# plt.plot(iters_all_alpha[1], avgutil_all_alpha[1], label='Alpha=0.40')
	# plt.plot(iters_all_alpha[2], avgutil_all_alpha[2], label='Alpha=0.60')
	# plt.plot(iters_all_alpha[3], avgutil_all_alpha[3], label='Alpha=0.80')
	# plt.xlabel('Episode #')
	# plt.ylabel('Average Utility')
	# plt.title('AvgUtil/Iters for QLearning')	
	# plt.legend()
	# plt.savefig('Forest_Q_AvgUtil_Alpha')


	# plt.clf()
	# plt.plot(iters_all_ed[0], time_all_ed[0], label='EpsilonDecay=0.90')
	# plt.plot(iters_all_ed[1], time_all_ed[1], label='EpsilonDecay=0.92')
	# plt.plot(iters_all_ed[2], time_all_ed[2], label='EpsilonDecay=0.94')
	# plt.plot(iters_all_ed[3], time_all_ed[3], label='EpsilonDecay=0.96')
	# plt.plot(iters_all_ed[4], time_all_ed[4], label='EpsilonDecay=0.98')
	# plt.xlabel('Episode #')
	# plt.ylabel('CPU Time (s)')
	# plt.title('CPU Time for QLearning')	
	# plt.legend()
	# plt.savefig('Forest_Q_CPUTime_EpsilonDecay')

	# plt.clf()
	# plt.plot(iters_all_ed[0], maxutil_all_ed[0], label='EpsilonDecay=0.90')
	# plt.plot(iters_all_ed[1], maxutil_all_ed[1], label='EpsilonDecay=0.92')
	# plt.plot(iters_all_ed[2], maxutil_all_ed[2], label='EpsilonDecay=0.94')
	# plt.plot(iters_all_ed[3], maxutil_all_ed[3], label='EpsilonDecay=0.96')
	# plt.plot(iters_all_ed[4], maxutil_all_ed[4], label='EpsilonDecay=0.98')
	# plt.xlabel('Iteration #')
	# plt.ylabel('Max Utility')
	# plt.title('MaxUtil/Iters for QLearning')	
	# plt.legend()
	# plt.savefig('Forest_Q_MaxUtil_EpsilonDecay')

	# plt.clf()
	# plt.plot(iters_all_ed[0], avgutil_all_ed[0], label='EpsilonDecay=0.90')
	# plt.plot(iters_all_ed[1], avgutil_all_ed[1], label='EpsilonDecay=0.92')
	# plt.plot(iters_all_ed[2], avgutil_all_ed[2], label='EpsilonDecay=0.94')
	# plt.plot(iters_all_ed[3], avgutil_all_ed[3], label='EpsilonDecay=0.96')
	# plt.plot(iters_all_ed[4], avgutil_all_ed[4], label='EpsilonDecay=0.98')
	# plt.xlabel('Episode #')
	# plt.ylabel('Average Utility')
	# plt.title('AvgUtil/Iters for QLearning')	
	# plt.legend()
	# plt.savefig('Forest_Q_AvgUtil_EpsilonDecay')	




	# iters10000 = range(0,10000)
	# iters1000000 = range(0,1000000)
	# time = []
	# reward_discount = []
	# reward_epsilon = []
	# reward_alpha = []
	# epsilon = [0.15, 0.3, 0.45, 0.6, 0.75, 0.9]
	# discount = [0.15, 0.3, 0.45, 0.6, 0.75, 0.9]
	# alpha = [0.2, 0.4, 0.6, 0.8]
	# mean_dis = []

	# for d in discount:
	# 	q = mdp.QLearning(fl8.P, fl8.R, gamma=d)
	# 	q.run()
	# 	#mean_dis.append(q.mean_discrepancy)
	# 	#print(q.max_iter)
	# 	#print(len(q.V))
	# 	reward_discount.append(q.rew)

	# for e in epsilon:
	# 	q = mdp.QLearning(fl8.P, fl8.R, gamma=0.3, epsilon=e)
	# 	q.run()
	# 	#mean_dis.append(q.mean_discrepancy)
	# 	#print(q.max_iter)
	# 	#print(len(q.V))
	# 	reward_epsilon.append(q.rew)

	# for a in alpha:
	# 	q = mdp.QLearning(fl8.P, fl8.R, gamma = 0.3, alpha=a)
	# 	q.run()
	# 	#mean_dis.append(q.mean_discrepancy)
	# 	#print(q.max_iter)
	# 	#print(len(q.V))
	# 	reward_alpha.append(q.rew)		


	# plt.clf()
	# plt.plot(iters10000, reward_epsilon[0], label='Epsilon=0.15')
	# plt.plot(iters10000, reward_epsilon[1], label='Epsilon=0.30')
	# plt.plot(iters10000, reward_epsilon[2], label='Epsilon=0.45')
	# plt.plot(iters10000, reward_epsilon[3], label='Epsilon=0.60')
	# plt.plot(iters10000, reward_epsilon[4], label='Epsilon=0.75')
	# plt.plot(iters10000, reward_epsilon[5], label='Epsilon=0.90')	
	# plt.xlabel('Iteration #')
	# plt.ylabel('Average Reward')
	# plt.title('Avg Reward/Iters by Epsilon for QLearning')	
	# plt.legend()
	# plt.savefig('FrozenLake_Q_RewardByEpsilon')

	# plt.clf()
	# plt.plot(iters10000, reward_discount[0], label='DiscountFactor=0.15')
	# plt.plot(iters10000, reward_discount[1], label='DiscountFactor=0.30')
	# plt.plot(iters10000, reward_discount[2], label='DiscountFactor=0.45')
	# plt.plot(iters10000, reward_discount[3], label='DiscountFactor=0.60')
	# plt.plot(iters10000, reward_discount[4], label='DiscountFactor=0.75')
	# plt.plot(iters10000, reward_discount[4], label='DiscountFactor=0.90')
	# plt.xlabel('Iteration #')
	# plt.ylabel('Average Reward')
	# plt.title('Avg Reward/Iters by Discount Factor for QLearning')	
	# plt.legend()
	# plt.savefig('FrozenLake_Q_RewardByDiscount')

	# plt.clf()
	# plt.plot(iters10000, reward_alpha[0], label='LearningRate=0.2')
	# plt.plot(iters10000, reward_alpha[1], label='LearningRate=0.4')
	# plt.plot(iters10000, reward_alpha[2], label='LearningRate=0.6')
	# plt.plot(iters10000, reward_alpha[3], label='LearningRate=0.8')
	# plt.xlabel('Iteration #')
	# plt.ylabel('Average Reward')
	# plt.title('Avg Reward/Iters by Learning Rate for QLearning')	
	# plt.legend()
	# plt.savefig('FrozenLake_Q_RewardByLearningRate')

def value_iter_forest():
	state_space = [1000,2000,3000,4000,5000]
	discount = [0.9, 0.92, 0.94, 0.96, 0.98]
	iters_all_discount = []
	time_all_discount = []
	maxutil_all_discount = []
	avgutil_all_discount = []
	error_all_discount = []
	iters_all_state = []	
	time_all_state = []	
	maxutil_all_state = []
	avgutil_all_state = []
	error_all_state = []

	# Tune by Discount Factor
	P, R = mdptoolbox.example.forest(1000)

	for d in discount:
		print("Discount Factor: " + str(d))
		vi = mdp.ValueIteration(P, R, d)
		vi.max_iter = 1000
		#vi.setVerbose()
		vi.run()

		run_stats_dict = pd.DataFrame.from_dict(vi.run_stats)
		iters = run_stats_dict['Iteration']
		maxutil = run_stats_dict['Max V']
		avgutil = run_stats_dict['Mean V']
		error = run_stats_dict['Error']		
		time = run_stats_dict['Time']

		iters_all_discount.append(iters)
		maxutil_all_discount.append(maxutil)
		avgutil_all_discount.append(avgutil)
		error_all_discount.append(error)
		time_all_discount.append(time)


	#Tune by State Space
	for s in state_space:
		P, R = mdptoolbox.example.forest(s)
		print("State Space: " + str(s))
	# 	t = []
	# 	r = []
	# 	i = []

		vi = mdp.ValueIteration(P, R, 0.98)
		vi.max_iter = 1000
		#vi.setVerbose()
		vi.run()

		run_stats_dict = pd.DataFrame.from_dict(vi.run_stats)
		iters = run_stats_dict['Iteration']
		maxutil = run_stats_dict['Max V']
		avgutil = run_stats_dict['Mean V']
		error = run_stats_dict['Error']		
		time = run_stats_dict['Time']

		iters_all_state.append(iters)
		maxutil_all_state.append(maxutil)
		avgutil_all_state.append(avgutil)
		error_all_state.append(error)
		time_all_state.append(time)		

	plt.clf()
	plt.plot(iters_all_discount[0], time_all_discount[0], label='DiscountFactor=0.90')
	plt.plot(iters_all_discount[1], time_all_discount[1], label='DiscountFactor=0.92')
	plt.plot(iters_all_discount[2], time_all_discount[2], label='DiscountFactor=0.94')
	plt.plot(iters_all_discount[3], time_all_discount[3], label='DiscountFactor=0.96')
	plt.plot(iters_all_discount[4], time_all_discount[4], label='DiscountFactor=0.98')
	plt.xlabel('Iteration #')
	plt.ylabel('CPU Time (s)')
	plt.title('CPU Time for VI')	
	plt.legend()
	plt.savefig('Forest_VI_CPUTime_DiscountFactor')

	plt.clf()
	plt.plot(iters_all_discount[0], maxutil_all_discount[0], label='DiscountFactor=0.90')
	plt.plot(iters_all_discount[1], maxutil_all_discount[1], label='DiscountFactor=0.92')
	plt.plot(iters_all_discount[2], maxutil_all_discount[2], label='DiscountFactor=0.94')
	plt.plot(iters_all_discount[3], maxutil_all_discount[3], label='DiscountFactor=0.96')
	plt.plot(iters_all_discount[4], maxutil_all_discount[4], label='DiscountFactor=0.98')
	plt.xlabel('Iteration #')
	plt.ylabel('Max Utility')
	plt.title('MaxUtil/Iters for VI')	
	plt.legend()
	plt.savefig('Forest_VI_MaxUtil_DiscountFactor')

	plt.clf()
	plt.plot(iters_all_discount[0], avgutil_all_discount[0], label='DiscountFactor=0.90')
	plt.plot(iters_all_discount[1], avgutil_all_discount[1], label='DiscountFactor=0.92')
	plt.plot(iters_all_discount[2], avgutil_all_discount[2], label='DiscountFactor=0.94')
	plt.plot(iters_all_discount[3], avgutil_all_discount[3], label='DiscountFactor=0.96')
	plt.plot(iters_all_discount[4], avgutil_all_discount[4], label='DiscountFactor=0.98')
	plt.xlabel('Iteration #')
	plt.ylabel('Average Utility')
	plt.title('AvgUtil/Iters for VI')	
	plt.legend()
	plt.savefig('Forest_VI_AvgUtil_DiscountFactor')

	plt.clf()
	plt.plot(iters_all_discount[0], error_all_discount[0], label='DiscountFactor=0.90')
	plt.plot(iters_all_discount[1], error_all_discount[1], label='DiscountFactor=0.92')
	plt.plot(iters_all_discount[2], error_all_discount[2], label='DiscountFactor=0.94')
	plt.plot(iters_all_discount[3], error_all_discount[3], label='DiscountFactor=0.96')
	plt.plot(iters_all_discount[4], error_all_discount[4], label='DiscountFactor=0.98')
	plt.xlabel('Iteration #')
	plt.ylabel('Error')
	plt.title('Error/Iters for VI')	
	plt.legend()
	plt.savefig('Forest_VI_Error_DiscountFactor')		

	plt.clf()
	plt.plot(iters_all_state[0], time_all_state[0], label='State Space=1000')
	plt.plot(iters_all_state[1], time_all_state[1], label='State Space=2000')
	plt.plot(iters_all_state[2], time_all_state[2], label='State Space=3000')
	plt.plot(iters_all_state[3], time_all_state[3], label='State Space=4000')
	plt.plot(iters_all_state[4], time_all_state[4], label='State Space=5000')
	plt.xlabel('Iteration #')
	plt.ylabel('CPU Time (s)')
	plt.title('CPU Time for VI')	
	plt.legend()
	plt.savefig('Forest_VI_CPUTime_StateSpace')

	plt.clf()
	plt.plot(iters_all_state[0], maxutil_all_state[0], label='State Space=1000')
	plt.plot(iters_all_state[1], maxutil_all_state[1], label='State Space=2000')
	plt.plot(iters_all_state[2], maxutil_all_state[2], label='State Space=3000')
	plt.plot(iters_all_state[3], maxutil_all_state[3], label='State Space=4000')
	plt.plot(iters_all_state[4], maxutil_all_state[4], label='State Space=5000')
	plt.xlabel('Iteration #')
	plt.ylabel('Max Utility')
	plt.title('MaxUtil/Iters for VI')	
	plt.legend()
	plt.savefig('Forest_VI_MaxUtil_StateSpace')

	plt.clf()
	plt.plot(iters_all_state[0], avgutil_all_state[0], label='State Space=1000')
	plt.plot(iters_all_state[1], avgutil_all_state[1], label='State Space=2000')
	plt.plot(iters_all_state[2], avgutil_all_state[2], label='State Space=3000')
	plt.plot(iters_all_state[3], avgutil_all_state[3], label='State Space=4000')
	plt.plot(iters_all_state[4], avgutil_all_state[4], label='State Space=5000')
	plt.xlabel('Iteration #')
	plt.ylabel('Average Utility')
	plt.title('AvgUtil/Iters for VI')	
	plt.legend()
	plt.savefig('Forest_VI_AvgUtil_StateSpace')

	plt.clf()
	plt.plot(iters_all_state[0], error_all_state[0], label='State Space=1000')
	plt.plot(iters_all_state[1], error_all_state[1], label='State Space=2000')
	plt.plot(iters_all_state[2], error_all_state[2], label='State Space=3000')
	plt.plot(iters_all_state[3], error_all_state[3], label='State Space=4000')
	plt.plot(iters_all_state[4], error_all_state[4], label='State Space=5000')
	plt.xlabel('Iteration #')
	plt.ylabel('Error')
	plt.title('Error/Iters for VI')	
	plt.legend()
	plt.savefig('Forest_VI_Error_StateSpace')		
				

def policy_iter_forest():
	state_space = [1000,2000,3000,4000,5000]
	discount = [0.9, 0.92, 0.94, 0.96, 0.98]
	iters_all_discount = []
	time_all_discount = []
	maxutil_all_discount = []
	avgutil_all_discount = []
	error_all_discount = []
	iters_all_state = []	
	time_all_state = []	
	maxutil_all_state = []
	avgutil_all_state = []
	error_all_state = []	

	# Tune by Discount Factor
	P, R = mdptoolbox.example.forest(1000)

	for d in discount:
		print("Discount Factor: " + str(d))
		pi = mdp.PolicyIteration(P, R, d)
		pi.max_iter = 1000
		#pi.setVerbose()
		pi.run()

		run_stats_dict = pd.DataFrame.from_dict(pi.run_stats)
		iters = run_stats_dict['Iteration']
		maxutil = run_stats_dict['Max V']
		avgutil = run_stats_dict['Mean V']
		error = run_stats_dict['Error']		
		time = run_stats_dict['Time']

		iters_all_discount.append(iters)
		maxutil_all_discount.append(maxutil)
		avgutil_all_discount.append(avgutil)
		error_all_discount.append(error)
		time_all_discount.append(time)


	#Tune by State Space
	for s in state_space:
		P, R = mdptoolbox.example.forest(s)
		print("State Space: " + str(s))
	# 	t = []
	# 	r = []
	# 	i = []

		pi = mdp.PolicyIteration(P, R, 0.98)
		pi.max_iter = 1000
		#vi.setVerbose()
		pi.run()

		run_stats_dict = pd.DataFrame.from_dict(pi.run_stats)
		iters = run_stats_dict['Iteration']
		maxutil = run_stats_dict['Max V']
		avgutil = run_stats_dict['Mean V']
		error = run_stats_dict['Error']		
		time = run_stats_dict['Time']

		iters_all_state.append(iters)
		maxutil_all_state.append(maxutil)
		avgutil_all_state.append(avgutil)
		error_all_state.append(error)
		time_all_state.append(time)		

	plt.clf()
	plt.plot(iters_all_discount[0], time_all_discount[0], label='DiscountFactor=0.90')
	plt.plot(iters_all_discount[1], time_all_discount[1], label='DiscountFactor=0.92')
	plt.plot(iters_all_discount[2], time_all_discount[2], label='DiscountFactor=0.94')
	plt.plot(iters_all_discount[3], time_all_discount[3], label='DiscountFactor=0.96')
	plt.plot(iters_all_discount[4], time_all_discount[4], label='DiscountFactor=0.98')
	plt.xlabel('Iteration #')
	plt.ylabel('CPU Time (s)')
	plt.title('CPU Time for PI')	
	plt.legend()
	plt.savefig('Forest_PI_CPUTime_DiscountFactor')

	plt.clf()
	plt.plot(iters_all_discount[0], maxutil_all_discount[0], label='DiscountFactor=0.90')
	plt.plot(iters_all_discount[1], maxutil_all_discount[1], label='DiscountFactor=0.92')
	plt.plot(iters_all_discount[2], maxutil_all_discount[2], label='DiscountFactor=0.94')
	plt.plot(iters_all_discount[3], maxutil_all_discount[3], label='DiscountFactor=0.96')
	plt.plot(iters_all_discount[4], maxutil_all_discount[4], label='DiscountFactor=0.98')
	plt.xlabel('Iteration #')
	plt.ylabel('Max Utility')
	plt.title('MaxUtil/Iters for PI')	
	plt.legend()
	plt.savefig('Forest_PI_MaxUtil_DiscountFactor')

	plt.clf()
	plt.plot(iters_all_discount[0], avgutil_all_discount[0], label='DiscountFactor=0.90')
	plt.plot(iters_all_discount[1], avgutil_all_discount[1], label='DiscountFactor=0.92')
	plt.plot(iters_all_discount[2], avgutil_all_discount[2], label='DiscountFactor=0.94')
	plt.plot(iters_all_discount[3], avgutil_all_discount[3], label='DiscountFactor=0.96')
	plt.plot(iters_all_discount[4], avgutil_all_discount[4], label='DiscountFactor=0.98')
	plt.xlabel('Iteration #')
	plt.ylabel('Average Utility')
	plt.title('AvgUtil/Iters for PI')	
	plt.legend()
	plt.savefig('Forest_PI_AvgUtil_DiscountFactor')

	plt.clf()
	plt.plot(iters_all_discount[0], error_all_discount[0], label='DiscountFactor=0.90')
	plt.plot(iters_all_discount[1], error_all_discount[1], label='DiscountFactor=0.92')
	plt.plot(iters_all_discount[2], error_all_discount[2], label='DiscountFactor=0.94')
	plt.plot(iters_all_discount[3], error_all_discount[3], label='DiscountFactor=0.96')
	plt.plot(iters_all_discount[4], error_all_discount[4], label='DiscountFactor=0.98')
	plt.xlabel('Iteration #')
	plt.ylabel('Error')
	plt.title('Error/Iters for PI')	
	plt.legend()
	plt.savefig('Forest_PI_Error_DiscountFactor')		

	plt.clf()
	plt.plot(iters_all_state[0], time_all_state[0], label='State Space=1000')
	plt.plot(iters_all_state[1], time_all_state[1], label='State Space=2000')
	plt.plot(iters_all_state[2], time_all_state[2], label='State Space=3000')
	plt.plot(iters_all_state[3], time_all_state[3], label='State Space=4000')
	plt.plot(iters_all_state[4], time_all_state[4], label='State Space=5000')
	plt.xlabel('Iteration #')
	plt.ylabel('CPU Time (s)')
	plt.title('CPU Time for PI')	
	plt.legend()
	plt.savefig('Forest_PI_CPUTime_StateSpace')

	plt.clf()
	plt.plot(iters_all_state[0], maxutil_all_state[0], label='State Space=1000')
	plt.plot(iters_all_state[1], maxutil_all_state[1], label='State Space=2000')
	plt.plot(iters_all_state[2], maxutil_all_state[2], label='State Space=3000')
	plt.plot(iters_all_state[3], maxutil_all_state[3], label='State Space=4000')
	plt.plot(iters_all_state[4], maxutil_all_state[4], label='State Space=5000')
	plt.xlabel('Iteration #')
	plt.ylabel('Max Utility')
	plt.title('MaxUtil/Iters for PI')	
	plt.legend()
	plt.savefig('Forest_PI_MaxUtil_StateSpace')

	plt.clf()
	plt.plot(iters_all_state[0], avgutil_all_state[0], label='State Space=1000')
	plt.plot(iters_all_state[1], avgutil_all_state[1], label='State Space=2000')
	plt.plot(iters_all_state[2], avgutil_all_state[2], label='State Space=3000')
	plt.plot(iters_all_state[3], avgutil_all_state[3], label='State Space=4000')
	plt.plot(iters_all_state[4], avgutil_all_state[4], label='State Space=5000')
	plt.xlabel('Iteration #')
	plt.ylabel('Average Utility')
	plt.title('AvgUtil/Iters for PI')	
	plt.legend()
	plt.savefig('Forest_PI_AvgUtil_StateSpace')

	plt.clf()
	plt.plot(iters_all_state[0], error_all_state[0], label='State Space=1000')
	plt.plot(iters_all_state[1], error_all_state[1], label='State Space=2000')
	plt.plot(iters_all_state[2], error_all_state[2], label='State Space=3000')
	plt.plot(iters_all_state[3], error_all_state[3], label='State Space=4000')
	plt.plot(iters_all_state[4], error_all_state[4], label='State Space=5000')
	plt.xlabel('Iteration #')
	plt.ylabel('Error')
	plt.title('Error/Iters for PI')	
	plt.legend()
	plt.savefig('Forest_PI_Error_StateSpace')		
				


def q_learning_forest():
	P, R = mdptoolbox.example.forest(1000)
	iters10000 = range(0,10000)
	iters500000 = range(0,500000)
	time = []
	reward_discount = []
	reward_epsilon = []
	reward_alpha = []
	epsilon = [0.15, 0.3, 0.45, 0.6, 0.75, 0.9]
	epsilon_decay = [.9, 0.92, 0.94, 0.96, 0.98]
	discount = [0.9, 0.92, 0.94, 0.96, 0.98]
	alpha = [0.2, 0.4, 0.6, 0.8]
	mean_dis = []

	iters_all_discount = []
	time_all_discount = []
	maxutil_all_discount = []
	avgutil_all_discount = []
	error_all_discount = []
	iters_all_epsilon = []
	time_all_epsilon = []
	maxutil_all_epsilon = []
	avgutil_all_epsilon = []
	error_all_epsilon = []	
	iters_all_alpha = []
	time_all_alpha = []
	maxutil_all_alpha = []
	avgutil_all_alpha = []
	error_all_alpha = []	
	iters_all_ed = []
	time_all_ed = []
	maxutil_all_ed = []
	avgutil_all_ed  = []
	error_all_ed = []		

	for d in discount:
		print("Discount Factor: " + str(d))
		q = mdp.QLearning(P, R, gamma=d, epsilon = 0.75, n_iter=500000, run_stat_frequency=1000)
		q.run()

		run_stats_dict = pd.DataFrame.from_dict(q.run_stats)
		print(run_stats_dict)
		iters = run_stats_dict['Iteration']
		maxutil = run_stats_dict['Max V']
		avgutil = run_stats_dict['Mean V']
		error = run_stats_dict['Error']		
		time = run_stats_dict['Time']

		#iters_all_discount.append(iters)
		iters_all_discount.append(range(1,iters.size+1))
		maxutil_all_discount.append(maxutil)
		avgutil_all_discount.append(avgutil)
		error_all_discount.append(error)
		time_all_discount.append(time)		

	# for e in epsilon:
	# 	print("Epsilon: " + str(e))
	# 	q = mdp.QLearning(P, R, gamma=0.98, epsilon=e, n_iter=500000, run_stat_frequency=1000)
	# 	q.run()
		
	# 	run_stats_dict = pd.DataFrame.from_dict(q.run_stats)
	# 	print(run_stats_dict)
	# 	iters = run_stats_dict['Iteration']
	# 	maxutil = run_stats_dict['Max V']
	# 	avgutil = run_stats_dict['Mean V']
	# 	error = run_stats_dict['Error']		
	# 	time = run_stats_dict['Time']

	# 	#iters_all_epsilon.append(iters)
	# 	iters_all_epsilon.append(range(1,iters.size+1))
	# 	maxutil_all_epsilon.append(maxutil)
	# 	avgutil_all_epsilon.append(avgutil)
	# 	error_all_epsilon.append(error)
	# 	time_all_epsilon.append(time)	

	# for a in alpha:
	# 	print("Alpha: " + str(a))
	# 	q = mdp.QLearning(P, R, gamma=0.98, epsilon=0.6, alpha=a, n_iter=500000, run_stat_frequency=1000)
	# 	q.run()
		
	# 	run_stats_dict = pd.DataFrame.from_dict(q.run_stats)
	# 	print(run_stats_dict)
	# 	iters = run_stats_dict['Iteration']
	# 	maxutil = run_stats_dict['Max V']
	# 	avgutil = run_stats_dict['Mean V']
	# 	error = run_stats_dict['Error']		
	# 	time = run_stats_dict['Time']

	# 	#iters_all_alpha.append(iters)
	# 	iters_all_alpha.append(range(1,iters.size+1))
	# 	maxutil_all_alpha.append(maxutil)
	# 	avgutil_all_alpha.append(avgutil)
	# 	error_all_alpha.append(error)
	# 	time_all_alpha.append(time)	

	# for ed in epsilon_decay:
	# 	print("Epsilon Decay: " + str(ed))
	# 	q = mdp.QLearning(P, R, gamma=0.98, epsilon=0.6, alpha=0.8, epsilon_decay=ed, n_iter=500000, run_stat_frequency=1000)
	# 	q.run()
		
	# 	run_stats_dict = pd.DataFrame.from_dict(q.run_stats)
	# 	print(run_stats_dict)
	# 	iters = run_stats_dict['Iteration']
	# 	maxutil = run_stats_dict['Max V']
	# 	avgutil = run_stats_dict['Mean V']
	# 	error = run_stats_dict['Error']		
	# 	time = run_stats_dict['Time']

	# 	#iters_all_ed.append(iters)
	# 	iters_all_ed.append(range(1,iters.size+1))
	# 	maxutil_all_ed.append(maxutil)
	# 	avgutil_all_ed.append(avgutil)
	# 	error_all_ed.append(error)
	# 	time_all_ed.append(time)			


	# plt.clf()
	# plt.plot(iters_all_discount[0], time_all_discount[0], label='DiscountFactor=0.90')
	# plt.plot(iters_all_discount[1], time_all_discount[1], label='DiscountFactor=0.92')
	# plt.plot(iters_all_discount[2], time_all_discount[2], label='DiscountFactor=0.94')
	# plt.plot(iters_all_discount[3], time_all_discount[3], label='DiscountFactor=0.96')
	# plt.plot(iters_all_discount[4], time_all_discount[4], label='DiscountFactor=0.98')
	# plt.xlabel('Episode #')
	# plt.ylabel('CPU Time (s)')
	# plt.title('CPU Time for QLearning')	
	# plt.legend()
	# plt.savefig('Forest_Q_CPUTime_DiscountFactor')

	# plt.clf()
	# plt.plot(iters_all_discount[0], maxutil_all_discount[0], label='DiscountFactor=0.90')
	# plt.plot(iters_all_discount[1], maxutil_all_discount[1], label='DiscountFactor=0.92')
	# plt.plot(iters_all_discount[2], maxutil_all_discount[2], label='DiscountFactor=0.94')
	# plt.plot(iters_all_discount[3], maxutil_all_discount[3], label='DiscountFactor=0.96')
	# plt.plot(iters_all_discount[4], maxutil_all_discount[4], label='DiscountFactor=0.98')
	# plt.xlabel('Episode #')
	# plt.ylabel('Max Utility')
	# plt.title('MaxUtil/Iters for QLearning')	
	# plt.legend()
	# plt.savefig('Forest_Q_MaxUtil_DiscountFactor')

	# plt.clf()
	# plt.plot(iters_all_discount[0], avgutil_all_discount[0], label='DiscountFactor=0.90')
	# plt.plot(iters_all_discount[1], avgutil_all_discount[1], label='DiscountFactor=0.92')
	# plt.plot(iters_all_discount[2], avgutil_all_discount[2], label='DiscountFactor=0.94')
	# plt.plot(iters_all_discount[3], avgutil_all_discount[3], label='DiscountFactor=0.96')
	# plt.plot(iters_all_discount[4], avgutil_all_discount[4], label='DiscountFactor=0.98')
	# plt.xlabel('Episode #')
	# plt.ylabel('Average Utility')
	# plt.title('AvgUtil/Iters for QLearning')	
	# plt.legend()
	# plt.savefig('Forest_Q_AvgUtil_DiscountFactor')


	# plt.clf()
	# plt.plot(iters_all_epsilon[0], time_all_epsilon[0], label='Epsilon=0.15')
	# plt.plot(iters_all_epsilon[1], time_all_epsilon[1], label='Epsilon=0.30')
	# plt.plot(iters_all_epsilon[2], time_all_epsilon[2], label='Epsilon=0.45')
	# plt.plot(iters_all_epsilon[3], time_all_epsilon[3], label='Epsilon=0.60')
	# plt.plot(iters_all_epsilon[4], time_all_epsilon[4], label='Epsilon=0.75')
	# plt.plot(iters_all_epsilon[5], time_all_epsilon[5], label='Epsilon=0.90')
	# plt.xlabel('Episode #')
	# plt.ylabel('CPU Time (s)')
	# plt.title('CPU Time for QLearning')	
	# plt.legend()
	# plt.savefig('Forest_Q_CPUTime_Epsilon')

	# plt.clf()
	# plt.plot(iters_all_epsilon[0], maxutil_all_epsilon[0], label='Epsilon=0.15')
	# plt.plot(iters_all_epsilon[1], maxutil_all_epsilon[1], label='Epsilon=0.30')
	# plt.plot(iters_all_epsilon[2], maxutil_all_epsilon[2], label='Epsilon=0.45')
	# plt.plot(iters_all_epsilon[3], maxutil_all_epsilon[3], label='Epsilon=0.60')
	# plt.plot(iters_all_epsilon[4], maxutil_all_epsilon[4], label='Epsilon=0.75')
	# plt.plot(iters_all_epsilon[5], maxutil_all_epsilon[5], label='Epsilon=0.90')
	# plt.xlabel('Episode #')
	# plt.ylabel('Max Utility')
	# plt.title('MaxUtil/Iters for QLearning')	
	# plt.legend()
	# plt.savefig('Forest_Q_MaxUtil_Epsilon')

	# plt.clf()
	# plt.plot(iters_all_epsilon[0], avgutil_all_epsilon[0], label='Epsilon=0.15')
	# plt.plot(iters_all_epsilon[1], avgutil_all_epsilon[1], label='Epsilon=0.30')
	# plt.plot(iters_all_epsilon[2], avgutil_all_epsilon[2], label='Epsilon=0.45')
	# plt.plot(iters_all_epsilon[3], avgutil_all_epsilon[3], label='Epsilon=0.60')
	# plt.plot(iters_all_epsilon[4], avgutil_all_epsilon[4], label='Epsilon=0.75')
	# plt.plot(iters_all_epsilon[5], avgutil_all_epsilon[5], label='Epsilon=0.90')
	# plt.xlabel('Episode #')
	# plt.ylabel('Average Utility')
	# plt.title('AvgUtil/Iters for QLearning')	
	# plt.legend()
	# plt.savefig('Forest_Q_AvgUtil_Epsilon')


	# plt.clf()
	# plt.plot(iters_all_alpha[0], time_all_alpha[0], label='Alpha=0.20')
	# plt.plot(iters_all_alpha[1], time_all_alpha[1], label='Alpha=0.40')
	# plt.plot(iters_all_alpha[2], time_all_alpha[2], label='Alpha=0.60')
	# plt.plot(iters_all_alpha[3], time_all_alpha[3], label='Alpha=0.80')
	# plt.xlabel('Iteration #')
	# plt.ylabel('CPU Time (s)')
	# plt.title('CPU Time for QLearning')	
	# plt.legend()
	# plt.savefig('Forest_Q_CPUTime_Alpha')

	# plt.clf()
	# plt.plot(iters_all_alpha[0], maxutil_all_alpha[0], label='Alpha=0.20')
	# plt.plot(iters_all_alpha[1], maxutil_all_alpha[1], label='Alpha=0.40')
	# plt.plot(iters_all_alpha[2], maxutil_all_alpha[2], label='Alpha=0.60')
	# plt.plot(iters_all_alpha[3], maxutil_all_alpha[3], label='Alpha=0.80')
	# plt.xlabel('Iteration #')
	# plt.ylabel('Max Utility')
	# plt.title('MaxUtil/Iters for QLearning')	
	# plt.legend()
	# plt.savefig('Forest_Q_MaxUtil_Alpha')

	# plt.clf()
	# plt.plot(iters_all_alpha[0], avgutil_all_alpha[0], label='Alpha=0.20')
	# plt.plot(iters_all_alpha[1], avgutil_all_alpha[1], label='Alpha=0.40')
	# plt.plot(iters_all_alpha[2], avgutil_all_alpha[2], label='Alpha=0.60')
	# plt.plot(iters_all_alpha[3], avgutil_all_alpha[3], label='Alpha=0.80')
	# plt.xlabel('Iteration #')
	# plt.ylabel('Average Utility')
	# plt.title('AvgUtil/Iters for QLearning')	
	# plt.legend()
	# plt.savefig('Forest_Q_AvgUtil_Alpha')


	# plt.clf()
	# plt.plot(iters_all_ed[0], time_all_ed[0], label='EpsilonDecay=0.90')
	# plt.plot(iters_all_ed[1], time_all_ed[1], label='EpsilonDecay=0.92')
	# plt.plot(iters_all_ed[2], time_all_ed[2], label='EpsilonDecay=0.94')
	# plt.plot(iters_all_ed[3], time_all_ed[3], label='EpsilonDecay=0.96')
	# plt.plot(iters_all_ed[4], time_all_ed[4], label='EpsilonDecay=0.98')
	# plt.xlabel('Episode #')
	# plt.ylabel('CPU Time (s)')
	# plt.title('CPU Time for QLearning')	
	# plt.legend()
	# plt.savefig('Forest_Q_CPUTime_EpsilonDecay')

	# plt.clf()
	# plt.plot(iters_all_ed[0], maxutil_all_ed[0], label='EpsilonDecay=0.90')
	# plt.plot(iters_all_ed[1], maxutil_all_ed[1], label='EpsilonDecay=0.92')
	# plt.plot(iters_all_ed[2], maxutil_all_ed[2], label='EpsilonDecay=0.94')
	# plt.plot(iters_all_ed[3], maxutil_all_ed[3], label='EpsilonDecay=0.96')
	# plt.plot(iters_all_ed[4], maxutil_all_ed[4], label='EpsilonDecay=0.98')
	# plt.xlabel('Episode #')
	# plt.ylabel('Max Utility')
	# plt.title('MaxUtil/Iters for QLearning')	
	# plt.legend()
	# plt.savefig('Forest_Q_MaxUtil_EpsilonDecay')

	# plt.clf()
	# plt.plot(iters_all_ed[0], avgutil_all_ed[0], label='EpsilonDecay=0.90')
	# plt.plot(iters_all_ed[1], avgutil_all_ed[1], label='EpsilonDecay=0.92')
	# plt.plot(iters_all_ed[2], avgutil_all_ed[2], label='EpsilonDecay=0.94')
	# plt.plot(iters_all_ed[3], avgutil_all_ed[3], label='EpsilonDecay=0.96')
	# plt.plot(iters_all_ed[4], avgutil_all_ed[4], label='EpsilonDecay=0.98')
	# plt.xlabel('Episode #')
	# plt.ylabel('Average Utility')
	# plt.title('AvgUtil/Iters for QLearning')	
	# plt.legend()
	# plt.savefig('Forest_Q_AvgUtil_EpsilonDecay')

def compare_frozen_lake():
	random.seed(0)
	fl4 = openai.OpenAI_MDPToolbox("FrozenLake-v0", s=4, render=True)
	vi = mdp.ValueIteration(fl4.P, fl4.R, 0.98)
	#vi.max_iter = 1000
	pi = mdp.PolicyIteration(fl4.P, fl4.R, 0.98)
	#pi.max_iter = 1000
	q = mdp.QLearning(fl4.P,fl4.R, gamma=0.98, epsilon=0.6, n_iter=500000, run_stat_frequency=1000)

	iters_all = []
	maxutil_all = []
	avgutil_all = []
	time_all = []
	episodes_all = []
	# iters = []	
	solvers = [vi, pi, q]
	for i in range(len(solvers)):
		print(i)
		# rew = range(1000000)
		solvers[i].run()
		print(solvers[i].policy)
		print(print_policy(solvers[i].policy))
		# if i !=2:
		# 	# for j in range(solvers[i].iter):
		# 	# 	rew[j] = solvers[i].run_stats[j]['Reward']
		# 	# rew.ffill()


		# if i==2:
			# for j in range(solvers[i].max_iter):
			# 	rew[j] = solvers[i].run_stats[j]['Reward']
		run_stats_dict = pd.DataFrame.from_dict(solvers[i].run_stats)
		iters = run_stats_dict['Iteration']
		maxutil = run_stats_dict['Max V']
		avgutil = run_stats_dict['Mean V']
		time = run_stats_dict['Time']
		
		if (i<2):
			iters_all.append(iters)
		if i==2:
			episodes_all = range(1,iters.size+1)
		maxutil_all.append(maxutil)
		avgutil_all.append(avgutil)
		time_all.append(time)
		#rewards_all.append(rew)



		# if i==2:
		# 	reward.append(solvers[i].rew)
		# else:
		# 	reward.append(np.mean(solves[i].V))

		# if i==2:
		# 	iters.append(1000000)
		# else:
		# 	iters.append(solvers[i].iter)


	#Compare values
	print('--- # Iterations to Converge --- ')
	print('VI: ' + str(iters_all[0].tail(1).item()))
	print('PI: ' + str(iters_all[1].tail(1).item()))
	print('Q: ' + str(episodes_all[-1]))
	print('-------------------------------- ')

	print('--------- # Max Util ----------- ')
	print('VI: ' + str(maxutil_all[0].tail(1).item()))
	print('PI: ' + str(maxutil_all[1].tail(1).item()))
	print('Q: ' + str(maxutil_all[2].tail(1).item()))
	print('-------------------------------- ')

	print('--------- # Avg Util ----------- ')
	print('VI: ' + str(avgutil_all[0].tail(1).item()))
	print('PI: ' + str(avgutil_all[1].tail(1).item()))
	print('Q: ' + str(avgutil_all[2].tail(1).item()))
	print('-------------------------------- ')

	print('--------- # CPU Time ----------- ')
	print('VI: ' + str(time_all[0].tail(1).item()))
	print('PI: ' + str(time_all[1].tail(1).item()))
	print('Q: ' + str(time_all[2].tail(1).item()))
	print('-------------------------------- ')

def compare_forest():
	random.seed(0)
	P, R = mdptoolbox.example.forest(1000)	
	vi = mdp.ValueIteration(P, R, 0.98)
	#vi.max_iter = 1000
	pi = mdp.PolicyIteration(P, R, 0.98)
	#pi.max_iter = 1000
	q = mdp.QLearning(P,R, gamma=0.98, epsilon=0.6, alpha=0.8, epsilon_decay=0.94, n_iter=500000, run_stat_frequency=1000)

	iters_all = []
	maxutil_all = []
	avgutil_all = []
	time_all = []
	episodes_all = []
	# iters = []	
	solvers = [vi, pi, q]
	for i in range(len(solvers)):
		print(i)
		# rew = range(1000000)
		solvers[i].run()
		print(len(solvers[i].policy))
		print(solvers[i].policy)
		# if i !=2:
		# 	# for j in range(solvers[i].iter):
		# 	# 	rew[j] = solvers[i].run_stats[j]['Reward']
		# 	# rew.ffill()


		# if i==2:
			# for j in range(solvers[i].max_iter):
			# 	rew[j] = solvers[i].run_stats[j]['Reward']
		run_stats_dict = pd.DataFrame.from_dict(solvers[i].run_stats)
		iters = run_stats_dict['Iteration']
		maxutil = run_stats_dict['Max V']
		avgutil = run_stats_dict['Mean V']
		time = run_stats_dict['Time']
		
		if (i<2):
			iters_all.append(iters)
		if i==2:
			episodes_all = range(1,iters.size+1)
		maxutil_all.append(maxutil)
		avgutil_all.append(avgutil)
		time_all.append(time)
		#rewards_all.append(rew)



		# if i==2:
		# 	reward.append(solvers[i].rew)
		# else:
		# 	reward.append(np.mean(solves[i].V))

		# if i==2:
		# 	iters.append(1000000)
		# else:
		# 	iters.append(solvers[i].iter)


	#Compare values
	print('--- # Iterations to Converge --- ')
	print('VI: ' + str(iters_all[0].tail(1).item()))
	print('PI: ' + str(iters_all[1].tail(1).item()))
	print('Q: ' + str(episodes_all[-1]))
	print('-------------------------------- ')

	print('--------- # Max Util ----------- ')
	print('VI: ' + str(maxutil_all[0].tail(1).item()))
	print('PI: ' + str(maxutil_all[1].tail(1).item()))
	print('Q: ' + str(maxutil_all[2].tail(1).item()))
	print('-------------------------------- ')

	print('--------- # Avg Util ----------- ')
	print('VI: ' + str(avgutil_all[0].tail(1).item()))
	print('PI: ' + str(avgutil_all[1].tail(1).item()))
	print('Q: ' + str(avgutil_all[2].tail(1).item()))
	print('-------------------------------- ')

	print('--------- # CPU Time ----------- ')
	print('VI: ' + str(time_all[0].tail(1).item()))
	print('PI: ' + str(time_all[1].tail(1).item()))
	print('Q: ' + str(time_all[2].tail(1).item()))
	print('-------------------------------- ')

	# plt.clf()
	# plt.plot(iters_all[0], util_all[0], label='Solver=ValueIteration')
	# plt.plot(iters_all[1], util_all[1], label='Solver=PolicyIteration')
	# plt.xlabel('Iteration #')
	# plt.ylabel('Max Utility')
	# plt.title('Max Utility/Iters Comparison')	
	# plt.legend()
	# plt.savefig('Forest_PIvsVIComparison_MaxUtil')	


	# plt.clf()
	# plt.plot(iters_all[2], util_all[2], label='Solver=QLearning')
	# plt.xlabel('Iteration #')
	# plt.ylabel('Max Utility')
	# plt.title('Max Utility/Iters Comparison')	
	# plt.legend()
	# plt.savefig('Forest_QLearningComparison_MaxUtil')	


	# plt.clf()
	# plt.plot(iters_all[0], time_all[0], label='Solver=ValueIteration')
	# plt.plot(iters_all[1], time_all[1], label='Solver=PolicyIteration')
	# plt.xlabel('Iteration #')
	# plt.ylabel('Compute Time (s)')
	# plt.title('Compute Time/Iters Comparison')	
	# plt.legend()
	# plt.savefig('Forest_PIvsVIComparison_ComputeTime')	

	# plt.clf()
	# plt.plot(iters_all[2], time_all[2], label='Solver=QLearning')
	# plt.xlabel('Iteration #')
	# plt.ylabel('Compute Time (s)')
	# plt.title('Compute Time/Iters Comparison')	
	# plt.legend()
	# plt.savefig('Forest_QLearningComparison_ComputeTime')

def main():
	#forest()
	#frozen_lake()
	#compare_forest()
	compare_frozen_lake()




if __name__ == "__main__":
	main()
	
