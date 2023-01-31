import mlrose
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import pandas as pd
from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.neural_network import MLPClassifier


# setup the randoms tate
RANDOM_STATE = 545510477

def TSP():
	'''
	https://readthedocs.org/projects/mlrose/downloads/pdf/stable/
	https://stackoverflow.com/questions/11886862/calculating-computational-time-and-memory-for-a-code-in-python
	'''
	iters = 501
	# Create list of distances between pairs of cities
	coords_list = [tuple(random.sample(range(1,10), 2)) for x in range(20)]
	# print(len(coords_list))
	# print(len(set(coords_list)))

	# coords_list=[(1, 1), (4, 2), (5, 2), (6, 4), (4, 4), (3, 6), (1, 5), (2, 3)]
	# Initialize fitness function object using dist_list
	# fitness_coords = mlrose.TravellingSales(coords = coords_list)
	# problem_fit = mlrose.DiscreteOpt(length = 8, fitness_fn = fitness_coords, maximize=True)
	problem_no_fit = mlrose.TSPOpt(length = 20, coords = coords_list, maximize=True)
	ga_fit = []
	sa_fit = []
	mimic_fit = []
	rhc_fit = []
	ga_time = []
	sa_time = []
	mimic_time = []
	rhc_time = []

	for i in range(1,iters):
		print(i)
		time_start_ga = time.time()
		best_state, best_fitness_ga = mlrose.genetic_alg(problem_no_fit, max_iters=i,random_state = 2)	
		time_elapsed_ga = (time.time() - time_start_ga)
		ga_time.append(time_elapsed_ga)
		ga_fit.append(best_fitness_ga)
		
		time_start_sa = time.time()
		best_state, best_fitness_sa = mlrose.simulated_annealing(problem_no_fit, max_iters=i,random_state = 2)	
		time_elapsed_sa = (time.time() - time_start_sa)
		sa_time.append(time_elapsed_sa)		
		sa_fit.append(best_fitness_sa)
		
		time_start_mimic = time.time()
		best_state, best_fitness_mimic = mlrose.mimic(problem_no_fit, max_iters=i,random_state = 2)	
		time_elapsed_mimic = (time.time() - time_start_mimic)
		mimic_time.append(time_elapsed_mimic)			
		mimic_fit.append(best_fitness_mimic)
		
		time_start_rhc = time.time()
		best_state, best_fitness_rhc = mlrose.random_hill_climb(problem_no_fit, max_iters=i,random_state = 2)	
		time_elapsed_rhc = (time.time() - time_start_rhc)		
		rhc_time.append(time_elapsed_rhc)			
		rhc_fit.append(best_fitness_rhc)

	x = list(range(1,iters))
	plt.plot(x, ga_fit, label = "GA")
	plt.plot(x, sa_fit, label = "SA")
	plt.plot(x, mimic_fit, label = "MIMIC")
	plt.plot(x, rhc_fit, label = "RHC")
	plt.xlabel('Iters')
	# Set the y axis label of the current axis.
	plt.ylabel('Fitness function value')
	# Set a title of the current axes.
	plt.title('Fitness Function Evaluation for TSP')
	plt.legend()
	plt.savefig('TSP_FitnessFunc')
	plt.close()

	x = list(range(1,iters))
	plt.plot(x, ga_time, label = "GA")
	plt.plot(x, sa_time, label = "SA")
	plt.plot(x, mimic_time, label = "MIMIC")
	plt.plot(x, rhc_fit, label = "RHC")
	plt.xlabel('Iters')
	# Set the y axis label of the current axis.
	plt.ylabel('Computation Time (s)')
	# Set a title of the current axes.
	plt.title('Computation Time Evalutation for TSP')		
	plt.legend()
	plt.savefig('TSP_ComputeTime')
	plt.close()	

def continous_peaks():
	'''
	https://readthedocs.org/projects/mlrose/downloads/pdf/stable/
	https://stackoverflow.com/questions/11886862/calculating-computational-time-and-memory-for-a-code-in-python
	'''
	iters=1001
	problem_fit = mlrose.DiscreteOpt(length = 20,fitness_fn = mlrose.ContinuousPeaks(t_pct=0.15), maximize=True)
	
	ga_fit = []
	sa_fit = []
	mimic_fit = []
	rhc_fit = []
	ga_time = []
	sa_time = []
	mimic_time = []
	rhc_time = []

	for i in range(1,iters):
		print(i)
		time_start_ga = time.time()
		best_state, best_fitness_ga = mlrose.genetic_alg(problem_fit, max_iters=i,random_state = 2)	
		time_elapsed_ga = (time.time() - time_start_ga)
		ga_time.append(time_elapsed_ga)
		ga_fit.append(best_fitness_ga)
		
		time_start_sa = time.time()
		best_state, best_fitness_sa = mlrose.simulated_annealing(problem_fit, max_iters=i,random_state = 2)	
		time_elapsed_sa = (time.time() - time_start_sa)
		sa_time.append(time_elapsed_sa)		
		sa_fit.append(best_fitness_sa)
		
		time_start_mimic = time.time()
		best_state, best_fitness_mimic = mlrose.mimic(problem_fit, max_iters=i,random_state = 2)	
		time_elapsed_mimic = (time.time() - time_start_mimic)
		mimic_time.append(time_elapsed_mimic)			
		mimic_fit.append(best_fitness_mimic)
		
		time_start_rhc = time.time()
		best_state, best_fitness_rhc = mlrose.random_hill_climb(problem_fit, max_iters=i,random_state = 2)	
		time_elapsed_rhc = (time.time() - time_start_rhc)		
		rhc_time.append(time_elapsed_rhc)			
		rhc_fit.append(best_fitness_rhc)

	x = list(range(1,iters))
	plt.plot(x, ga_fit, label = "GA")
	plt.plot(x, sa_fit, label = "SA")
	plt.plot(x, mimic_fit, label = "MIMIC")
	plt.plot(x, rhc_fit, label = "RHC")
	plt.xlabel('Iters')
	# Set the y axis label of the current axis.
	plt.ylabel('Fitness function value')
	# Set a title of the current axes.
	plt.title('Fitness Function Evaluation for CP')
	plt.legend()
	plt.savefig('ContinuousPeaks_FitnessFunc')
	plt.close()

	x = list(range(1,iters))
	plt.plot(x, ga_time, label = "GA")
	plt.plot(x, sa_time, label = "SA")
	plt.plot(x, mimic_time, label = "MIMIC")
	plt.plot(x, rhc_time, label = "RHC")
	plt.xlabel('Iters')
	# Set the y axis label of the current axis.
	plt.ylabel('Computation Time (s)')
	# Set a title of the current axes.
	plt.title('Computation Time Evalutation for CP')		
	plt.legend()
	plt.savefig('ContinuousPeaks_ComputeTime')
	plt.close()	

def flip_flop():
	'''
	https://readthedocs.org/projects/mlrose/downloads/pdf/stable/
	https://stackoverflow.com/questions/11886862/calculating-computational-time-and-memory-for-a-code-in-python
	'''
	iters=1001
	problem_fit = mlrose.DiscreteOpt(length = 20,fitness_fn = mlrose.FlipFlop())
	
	ga_fit = []
	sa_fit = []
	mimic_fit = []
	rhc_fit = []
	ga_time = []
	sa_time = []
	mimic_time = []
	rhc_time = []

	for i in range(1,iters):
		print(i)
		time_start_ga = time.time()
		best_state, best_fitness_ga = mlrose.genetic_alg(problem_fit, max_iters=i,random_state = 2)	
		time_elapsed_ga = (time.time() - time_start_ga)
		ga_time.append(time_elapsed_ga)
		ga_fit.append(best_fitness_ga)
		
		time_start_sa = time.time()
		best_state, best_fitness_sa = mlrose.simulated_annealing(problem_fit, max_iters=i,random_state = 2)	
		time_elapsed_sa = (time.time() - time_start_sa)
		sa_time.append(time_elapsed_sa)		
		sa_fit.append(best_fitness_sa)
		
		time_start_mimic = time.time()
		best_state, best_fitness_mimic = mlrose.mimic(problem_fit, max_iters=i,random_state = 2)	
		time_elapsed_mimic = (time.time() - time_start_mimic)
		mimic_time.append(time_elapsed_mimic)			
		mimic_fit.append(best_fitness_mimic)
		
		time_start_rhc = time.time()
		best_state, best_fitness_rhc = mlrose.random_hill_climb(problem_fit, max_iters=i,random_state = 2)	
		time_elapsed_rhc = (time.time() - time_start_rhc)		
		rhc_time.append(time_elapsed_rhc)			
		rhc_fit.append(best_fitness_rhc)

	x = list(range(1,iters))
	plt.plot(x, ga_fit, label = "GA")
	plt.plot(x, sa_fit, label = "SA")
	plt.plot(x, mimic_fit, label = "MIMIC")
	plt.plot(x, rhc_fit, label = "RHC")
	plt.xlabel('Iters')
	# Set the y axis label of the current axis.
	plt.ylabel('Fitness function value')
	# Set a title of the current axes.
	plt.title('Fitness Function Evaluation for FlipFlop')
	plt.legend()
	plt.savefig('FlipFlop_FitnessFunc')
	plt.close()

	x = list(range(1,iters))
	plt.plot(x, ga_time, label = "GA")
	plt.plot(x, sa_time, label = "SA")
	plt.plot(x, mimic_time, label = "MIMIC")
	plt.plot(x, rhc_time, label = "RHC")
	plt.xlabel('Iters')
	# Set the y axis label of the current axis.
	plt.ylabel('Computation Time (s)')
	# Set a title of the current axes.
	plt.title('Computation Time Evalutation for FlipFlop')		
	plt.legend()
	plt.savefig('FlipFlop_ComputeTime')
	plt.close()

def flip_flop_ht():
	'''
	https://readthedocs.org/projects/mlrose/downloads/pdf/stable/
	https://stackoverflow.com/questions/11886862/calculating-computational-time-and-memory-for-a-code-in-python
	'''
	pct=[.1, .2, .3, .4, .5, .6, .7, .8, .9]
	problem_fit = mlrose.DiscreteOpt(length = 50,fitness_fn = mlrose.FlipFlop())
	
	ga_fit = []
	sa_fit = []
	mimic_fit = []
	rhc_fit = []
	ga_time = []
	sa_time = []
	mimic_time = []
	rhc_time = []

	for i in pct:
		print(i)
		best_state, best_fitness_mimic = mlrose.mimic(problem_fit, max_iters=200, keep_pct=i, random_state = 2)	
		mimic_fit.append(best_fitness_mimic)
		

	x = pct
	plt.plot(x, mimic_fit, label = "MIMIC")
	plt.xlabel('Proportion of Kept Samples')
	# Set the y axis label of the current axis.
	plt.ylabel('Fitness function value')
	# Set a title of the current axes.
	plt.title('Fitness Function Evaluation for FlipFlop vs Keep PCT')
	plt.legend()
	plt.savefig('FlipFlop_MIMICFitnessFuncPCT')
	plt.close()

def four_peaks():
	'''
	https://readthedocs.org/projects/mlrose/downloads/pdf/stable/
	https://stackoverflow.com/questions/11886862/calculating-computational-time-and-memory-for-a-code-in-python
	'''
	iters=1001
	problem_fit = mlrose.DiscreteOpt(length = 20,fitness_fn = mlrose.FourPeaks(t_pct=0.15))
	
	ga_fit = []
	sa_fit = []
	mimic_fit = []
	rhc_fit = []
	ga_time = []
	sa_time = []
	mimic_time = []
	rhc_time = []

	for i in range(1,iters):
		print(i)
		time_start_ga = time.time()
		best_state, best_fitness_ga = mlrose.genetic_alg(problem_fit, max_iters=i,random_state = 2)	
		time_elapsed_ga = (time.time() - time_start_ga)
		ga_time.append(time_elapsed_ga)
		ga_fit.append(best_fitness_ga)
		
		time_start_sa = time.time()
		best_state, best_fitness_sa = mlrose.simulated_annealing(problem_fit, max_iters=i,random_state = 2)	
		time_elapsed_sa = (time.time() - time_start_sa)
		sa_time.append(time_elapsed_sa)		
		sa_fit.append(best_fitness_sa)
		
		time_start_mimic = time.time()
		best_state, best_fitness_mimic = mlrose.mimic(problem_fit, max_iters=i,random_state = 2)	
		time_elapsed_mimic = (time.time() - time_start_mimic)
		mimic_time.append(time_elapsed_mimic)			
		mimic_fit.append(best_fitness_mimic)
		
		time_start_rhc = time.time()
		best_state, best_fitness_rhc = mlrose.random_hill_climb(problem_fit, max_iters=i,random_state = 2)	
		time_elapsed_rhc = (time.time() - time_start_rhc)		
		rhc_time.append(time_elapsed_rhc)			
		rhc_fit.append(best_fitness_rhc)

	x = list(range(1,iters))
	plt.plot(x, ga_fit, label = "GA")
	plt.plot(x, sa_fit, label = "SA")
	plt.plot(x, mimic_fit, label = "MIMIC")
	plt.plot(x, rhc_fit, label = "RHC")
	plt.xlabel('Iters')
	# Set the y axis label of the current axis.
	plt.ylabel('Fitness function value')
	# Set a title of the current axes.
	plt.title('Fitness Function Evaluation for FourPeaks')
	plt.legend()
	plt.savefig('FourPeaks_FitnessFunc')
	plt.close()

	x = list(range(1,iters))
	plt.plot(x, ga_time, label = "GA")
	plt.plot(x, sa_time, label = "SA")
	plt.plot(x, mimic_time, label = "MIMIC")
	plt.plot(x, rhc_time, label = "RHC")
	plt.xlabel('Iters')
	# Set the y axis label of the current axis.
	plt.ylabel('Computation Time (s)')
	# Set a title of the current axes.
	plt.title('Computation Time Evalutation for FourPeaks')		
	plt.legend()
	plt.savefig('FourPeaks_ComputeTime')
	plt.close()	

def four_peaks():
	'''
	https://readthedocs.org/projects/mlrose/downloads/pdf/stable/
	https://stackoverflow.com/questions/11886862/calculating-computational-time-and-memory-for-a-code-in-python
	'''
	iters=4001
	problem_fit = mlrose.DiscreteOpt(length = 20,fitness_fn = mlrose.FourPeaks(t_pct=0.15))
	
	ga_fit = []
	sa_fit = []
	mimic_fit = []
	rhc_fit = []
	ga_time = []
	sa_time = []
	mimic_time = []
	rhc_time = []

	for i in range(1,iters):
		print(i)
		time_start_ga = time.time()
		best_state, best_fitness_ga = mlrose.genetic_alg(problem_fit, max_iters=i,random_state = 2)	
		time_elapsed_ga = (time.time() - time_start_ga)
		ga_time.append(time_elapsed_ga)
		ga_fit.append(best_fitness_ga)
		
		time_start_sa = time.time()
		best_state, best_fitness_sa = mlrose.simulated_annealing(problem_fit, max_iters=i,random_state = 2)	
		time_elapsed_sa = (time.time() - time_start_sa)
		sa_time.append(time_elapsed_sa)		
		sa_fit.append(best_fitness_sa)
		
		time_start_mimic = time.time()
		best_state, best_fitness_mimic = mlrose.mimic(problem_fit, max_iters=i,random_state = 2)	
		time_elapsed_mimic = (time.time() - time_start_mimic)
		mimic_time.append(time_elapsed_mimic)			
		mimic_fit.append(best_fitness_mimic)
		
		time_start_rhc = time.time()
		best_state, best_fitness_rhc = mlrose.random_hill_climb(problem_fit, max_iters=i,random_state = 2)	
		time_elapsed_rhc = (time.time() - time_start_rhc)		
		rhc_time.append(time_elapsed_rhc)			
		rhc_fit.append(best_fitness_rhc)

	x = list(range(1,iters))
	plt.plot(x, ga_fit, label = "GA")
	plt.plot(x, sa_fit, label = "SA")
	plt.plot(x, mimic_fit, label = "MIMIC")
	plt.plot(x, rhc_fit, label = "RHC")
	plt.xlabel('Iters')
	# Set the y axis label of the current axis.
	plt.ylabel('Fitness function value')
	# Set a title of the current axes.
	plt.title('Fitness Function Evaluation for FourPeaks')
	plt.legend()
	plt.savefig('FourPeaks_FitnessFunc')
	plt.close()

	x = list(range(1,iters))
	plt.plot(x, ga_time, label = "GA")
	plt.plot(x, sa_time, label = "SA")
	plt.plot(x, mimic_time, label = "MIMIC")
	plt.plot(x, rhc_time, label = "RHC")
	plt.xlabel('Iters')
	# Set the y axis label of the current axis.
	plt.ylabel('Computation Time (s)')
	# Set a title of the current axes.
	plt.title('Computation Time Evalutation for FourPeaks')		
	plt.legend()
	plt.savefig('FourPeaks_ComputeTime')
	plt.close()

def one_max():
	'''
	https://readthedocs.org/projects/mlrose/downloads/pdf/stable/
	https://stackoverflow.com/questions/11886862/calculating-computational-time-and-memory-for-a-code-in-python
	'''
	iters=[500,1000,1500,2000,2500,3000]
	problem_fit = mlrose.DiscreteOpt(length = 20,fitness_fn = mlrose.OneMax())
	
	ga_fit = []
	sa_fit = []
	mimic_fit = []
	rhc_fit = []
	ga_time = []
	sa_time = []
	mimic_time = []
	rhc_time = []

	for i in iters:
		print(i)
		time_start_ga = time.time()
		best_state, best_fitness_ga = mlrose.genetic_alg(problem_fit, max_iters=i,random_state = 2)	
		time_elapsed_ga = (time.time() - time_start_ga)
		ga_time.append(time_elapsed_ga)
		ga_fit.append(best_fitness_ga)
		
		time_start_sa = time.time()
		best_state, best_fitness_sa = mlrose.simulated_annealing(problem_fit, max_iters=i,random_state = 2)	
		time_elapsed_sa = (time.time() - time_start_sa)
		sa_time.append(time_elapsed_sa)		
		sa_fit.append(best_fitness_sa)
		
		time_start_mimic = time.time()
		best_state, best_fitness_mimic = mlrose.mimic(problem_fit, max_iters=i,random_state = 2)	
		time_elapsed_mimic = (time.time() - time_start_mimic)
		mimic_time.append(time_elapsed_mimic)			
		mimic_fit.append(best_fitness_mimic)
		
		time_start_rhc = time.time()
		best_state, best_fitness_rhc = mlrose.random_hill_climb(problem_fit, max_iters=i,random_state = 2)	
		time_elapsed_rhc = (time.time() - time_start_rhc)		
		rhc_time.append(time_elapsed_rhc)			
		rhc_fit.append(best_fitness_rhc)

	x = iters
	plt.plot(x, ga_fit, label = "GA")
	plt.plot(x, sa_fit, label = "SA")
	plt.plot(x, mimic_fit, label = "MIMIC")
	plt.plot(x, rhc_fit, label = "RHC")
	plt.xlabel('Iters')
	# Set the y axis label of the current axis.
	plt.ylabel('Fitness function value')
	# Set a title of the current axes.
	plt.title('Fitness Function Evaluation for OneMax')
	plt.legend()
	plt.savefig('OneMax_FitnessFunc')
	plt.close()

	x = iters
	plt.plot(x, ga_time, label = "GA")
	plt.plot(x, sa_time, label = "SA")
	plt.plot(x, mimic_time, label = "MIMIC")
	plt.plot(x, rhc_time, label = "RHC")
	plt.xlabel('Iters')
	# Set the y axis label of the current axis.
	plt.ylabel('Computation Time (s)')
	# Set a title of the current axes.
	plt.title('Computation Time Evalutation for OneMax')		
	plt.legend()
	plt.savefig('OneMax_ComputeTime')
	plt.close()

# def opt_weights(X, Y):

# 	f1_train = []
# 	f1_test = []
# 	for i in iter:
# 		clf = MLPClassifier(random_state=RANDOM_STATE, hidden_layer_sizes=30, activation='relu', solver='sgd').fit(X, Y)		estimator.fit(X_train, y_train)
# 		Y_pred_train = clf.predict(X_train)
# 		Y_pred_test = clf.predict(X_test)
# 		f1_train.append(f1_score(y_train, Y_pred_train))
# 		f1_test.append(f1_score(y_test, Y_pred_test))


def rhc_opt(X_train,Y_train, X_test, Y_test):
	'''
	https://readthedocs.org/projects/mlrose/downloads/pdf/stable/
	'''
	iters = [1,50,100,150,200,250,300]
	restarts = [0,25,50]
	rhc_train_0_f1 = []
	rhc_train_25_f1 = []
	rhc_train_50_f1 = []

	rhc_train_0_acc = []
	rhc_train_25_acc = []
	rhc_train_50_acc = []

	rhc_test_0_f1 = []
	rhc_test_25_f1 = []
	rhc_test_50_f1 = []

	rhc_test_0_acc = []
	rhc_test_25_acc = []
	rhc_test_50_acc = []	

	rhc_f1_test = []
	rhc_acc_test = []
	for j in restarts:
		for i in iters:
			print(str(j)+","+str(i))
			# Initialize neural network object and fit object
			rhc = mlrose.NeuralNetwork(hidden_nodes = [30], activation = 'relu',algorithm = 'random_hill_climb', max_iters = i,bias = True, early_stopping=True,restarts=j, random_state = 2)
			rhc.fit(X_train,Y_train)
			
			if (j==0):
				# Predict labels for train set and assess accuracy
				y_train_pred = rhc.predict(X_train)
				y_train_f1 = f1_score(Y_train, y_train_pred)
				rhc_train_0_f1.append(y_train_f1)
				rhc_train_0_acc.append(accuracy_score(Y_train, y_train_pred))
				print(rhc_train_0_f1)
				print(rhc_train_0_acc)
				y_test_pred = rhc.predict(X_test)
				y_test_f1 = f1_score(Y_test, y_test_pred)
				rhc_test_0_f1.append(y_test_f1)
				rhc_test_0_acc.append(accuracy_score(Y_test, y_test_pred))

			elif (j==25):
				# Predict labels for train set and assess accuracy
				y_train_pred = rhc.predict(X_train)
				y_train_f1 = f1_score(Y_train, y_train_pred)
				rhc_train_25_f1.append(y_train_f1)
				rhc_train_25_acc.append(accuracy_score(Y_train, y_train_pred))
				print(rhc_train_25_f1)
				print(rhc_train_25_acc)
				y_test_pred = rhc.predict(X_test)
				y_test_f1 = f1_score(Y_test, y_test_pred)
				rhc_test_25_f1.append(y_test_f1)
				rhc_test_25_acc.append(accuracy_score(Y_test, y_test_pred))

			elif (j==50):
				# Predict labels for train set and assess accuracy
				y_train_pred = rhc.predict(X_train)
				y_train_f1 = f1_score(Y_train, y_train_pred)
				rhc_train_50_f1.append(y_train_f1)
				rhc_train_50_acc.append(accuracy_score(Y_train, y_train_pred))
				print(rhc_train_50_f1)
				print(rhc_train_50_acc)				
				y_test_pred = rhc.predict(X_test)
				y_test_f1 = f1_score(Y_test, y_test_pred)
				rhc_test_50_f1.append(y_test_f1)
				rhc_test_50_acc.append(accuracy_score(Y_test, y_test_pred))					

	x = iters
	plt.plot(x, rhc_train_0_f1, label = "Restarts-0")
	plt.plot(x, rhc_train_25_f1, label = "Restarts-25")
	plt.plot(x, rhc_train_50_f1, label = "Restarts-50")
	plt.xlabel('Iters')
	# Set the y axis label of the current axis.
	plt.ylabel('F1 Score')
	# Set a title of the current axes.
	plt.title('Hyperparam Tuning for RHC, Training Set')		
	plt.legend()
	plt.savefig('HyperRHC_NNTrain_F1')
	plt.close()

	x = iters
	plt.plot(x, rhc_train_0_acc, label = "Restarts-0")
	plt.plot(x, rhc_train_25_acc, label = "Restarts-25")
	plt.plot(x, rhc_train_50_acc, label = "Restarts-50")
	plt.xlabel('Iters')
	# Set the y axis label of the current axis.
	plt.ylabel('Accuracy Score')
	# Set a title of the current axes.
	plt.title('Hyperparam Tuning for RHC, Training Set')		
	plt.legend()
	plt.savefig('HyperRHC_NNTrain_Acc')
	plt.close()	

	x = iters
	plt.plot(x, rhc_test_0_f1, label = "Restarts-0")
	plt.plot(x, rhc_test_25_f1, label = "Restarts-25")
	plt.plot(x, rhc_test_50_f1, label = "Restarts-50")
	plt.xlabel('Iters')
	# Set the y axis label of the current axis.
	plt.ylabel('F1 Score')
	# Set a title of the current axes.
	plt.title('Hyperparam Tuning for RHC, Test Set')		
	plt.legend()
	plt.savefig('HyperRHC_NNTest_F1')
	plt.close()	

	x = iters
	plt.plot(x, rhc_test_0_acc, label = "Restarts-0")
	plt.plot(x, rhc_test_25_acc, label = "Restarts-25")
	plt.plot(x, rhc_test_50_acc, label = "Restarts-50")
	plt.xlabel('Iters')
	# Set the y axis label of the current axis.
	plt.ylabel('Accuracy Score')
	# Set a title of the current axes.
	plt.title('Hyperparam Tuning for RHC, Test Set')		
	plt.legend()
	plt.savefig('HyperRHC_NNTest_Acc')
	plt.close()			

	# x = list(range(1,iters))
	# plt.plot(x, rhc_f1_train, label = "Training")
	# plt.plot(x, rhc_f1_test, label = "Testing")
	# plt.xlabel('Iters')
	# # Set the y axis label of the current axis.
	# plt.ylabel('Computation Time (s)')
	# # Set a title of the current axes.
	# plt.title('Computation Time Evaluation for RHC')		
	# plt.legend()
	# plt.savefig('RHC_NNOpt_CompTime')
	# plt.close()			

def sa_opt(X_train,Y_train, X_test, Y_test):
	'''
	https://readthedocs.org/projects/mlrose/downloads/pdf/stable/
	'''
	iters = [1,500,1000,1500,2000,2500,3000,3500,4000]
	schedule=[mlrose.GeomDecay(), mlrose.ArithDecay(), mlrose.ExpDecay()]

	sa_train_geom_f1 = []
	sa_train_arith_f1 = []
	sa_train_exp_f1 = []

	sa_test_geom_f1 = []
	sa_test_arith_f1 = []
	sa_test_exp_f1= []

	sa_train_geom_acc = []
	sa_train_arith_acc = []
	sa_train_exp_acc = []

	sa_test_geom_acc = []
	sa_test_arith_acc= []
	sa_test_exp_acc = []			
	for j in range(len(schedule)):
		for i in iters:
			print(str(j)+','+str(i))
			# Initialize neural network object and fit object
			sa = mlrose.NeuralNetwork(hidden_nodes = [30], activation = 'relu',algorithm = 'simulated_annealing', max_iters = i,bias = True, early_stopping=True, schedule=schedule[j], random_state = 2)
			sa.fit(X_train,Y_train)
			
			if (j==0):
				# Predict labels for train set and assess accuracy
				y_train_pred = sa.predict(X_train)
				# y_train_prec = precision_score(Y_train, y_train_pred)
				# sa_train_geom_prec.append(y_train_prec)
				# y_train_recall = recall_score(Y_train, y_train_pred)
				# sa_train_geom_recall.append(y_train_recall)
				y_train_f1 = f1_score(Y_train, y_train_pred)
				sa_train_geom_f1.append(y_train_f1)
				print(sa_train_geom_f1)
				y_train_acc = accuracy_score(Y_train, y_train_pred)
				sa_train_geom_acc.append(y_train_acc)
				print(sa_train_geom_acc)												

				y_test_pred = sa.predict(X_test)
				# y_test_prec = precision_score(Y_test, y_test_pred)
				# sa_test_geom_prec.append(y_test_prec)
				# y_test_recall = recall_score(Y_test, y_test_pred)
				# sa_test_geom_recall.append(y_test_recall)
				y_test_f1 = f1_score(Y_test, y_test_pred)
				sa_test_geom_f1.append(y_test_f1)
				y_test_acc = accuracy_score(Y_test, y_test_pred)
				sa_test_geom_acc.append(y_test_acc)

			if (j==1):
				# Predict labels for train set and assess accuracy
				y_train_pred = sa.predict(X_train)
				y_train_f1 = f1_score(Y_train, y_train_pred)
				sa_train_arith_f1.append(y_train_f1)
				sa_train_arith_acc.append(accuracy_score(Y_train, y_train_pred))
				print(sa_train_arith_f1)
				print(sa_train_arith_acc)

				y_test_pred = sa.predict(X_test)
				y_test_f1 = f1_score(Y_test, y_test_pred)
				sa_test_arith_f1.append(y_test_f1)
				sa_test_arith_acc.append(accuracy_score(Y_test, y_test_pred))
			if (j==2):
				# Predict labels for train set and assess accuracy
				y_train_pred = sa.predict(X_train)
				y_train_f1 = f1_score(Y_train, y_train_pred)
				sa_train_exp_f1.append(y_train_f1)
				sa_train_exp_acc.append(accuracy_score(Y_train, y_train_pred))
				print(sa_train_exp_f1)
				print(sa_train_exp_acc)				

				y_test_pred = sa.predict(X_test)
				y_test_f1 = f1_score(Y_test, y_test_pred)
				sa_test_exp_f1.append(y_test_f1)
				sa_test_exp_acc.append(accuracy_score(Y_test, y_test_pred))								

	# x=iters
	# plt.plot(x, sa_train_geom_prec, label = "Schedule-GeomDecay")
	# # plt.plot(x, sa_train_arith, label = "Schedule-ArithDecay")
	# # plt.plot(x, sa_train_exp, label = "Schedule-ExpDecay")
	# plt.xlabel('Iters')
	# # Set the y axis label of the current axis.
	# plt.ylabel('Precision Score')
	# # Set a title of the current axes.
	# plt.title('Hyperparam Tuning for SA, Training Set')		
	# plt.legend()
	# plt.savefig('HyperSA_NNTrain_Prec')
	# plt.close()

	# x=iters
	# plt.plot(x, sa_train_geom_recall, label = "Schedule-GeomDecay")
	# # plt.plot(x, sa_train_arith, label = "Schedule-ArithDecay")
	# # plt.plot(x, sa_train_exp, label = "Schedule-ExpDecay")
	# plt.xlabel('Iters')
	# # Set the y axis label of the current axis.
	# plt.ylabel('Recall Score')
	# # Set a title of the current axes.
	# plt.title('Hyperparam Tuning for SA, Training Set')		
	# plt.legend()
	# plt.savefig('HyperSA_NNTrain_Recall')
	# plt.close()

	x=iters
	plt.plot(x, sa_train_geom_f1, label = "Schedule-GeomDecay")
	plt.plot(x, sa_train_arith_f1, label = "Schedule-ArithDecay")
	plt.plot(x, sa_train_exp_f1, label = "Schedule-ExpDecay")
	plt.xlabel('Iters')
	# Set the y axis label of the current axis.
	plt.ylabel('F1 Score')
	# Set a title of the current axes.
	plt.title('Hyperparam Tuning for SA, Training Set')		
	plt.legend()
	plt.savefig('HyperSA_NNTrain_F1')
	plt.close()

	x=iters
	plt.plot(x, sa_train_geom_acc, label = "Schedule-GeomDecay")
	plt.plot(x, sa_train_arith_acc, label = "Schedule-ArithDecay")
	plt.plot(x, sa_train_exp_acc, label = "Schedule-ExpDecay")
	plt.xlabel('Iters')
	# Set the y axis label of the current axis.
	plt.ylabel('Accuracy Score')
	# Set a title of the current axes.
	plt.title('Hyperparam Tuning for SA, Training Set')		
	plt.legend()
	plt.savefig('HyperSA_NNTrain_Acc')
	plt.close()			

	x=iters
	plt.plot(x, sa_test_geom_f1, label = "Schedule-GeomDecay")
	plt.plot(x, sa_test_arith_f1, label = "Schedule-ArithDecay")
	plt.plot(x, sa_test_exp_f1, label = "Schedule-ExpDecay")
	plt.xlabel('Iters')
	# Set the y axis label of the current axis.
	plt.ylabel('F1 Score')
	# Set a title of the current axes.
	plt.title('Hyperparam Tuning for SA, Training Set')		
	plt.legend()
	plt.savefig('HyperSA_NNTest_F1')
	plt.close()

	x=iters
	plt.plot(x, sa_test_geom_acc, label = "Schedule-GeomDecay")
	plt.plot(x, sa_test_arith_acc, label = "Schedule-ArithDecay")
	plt.plot(x, sa_test_exp_acc, label = "Schedule-ExpDecay")
	plt.xlabel('Iters')
	# Set the y axis label of the current axis.
	plt.ylabel('Accuracy Score')
	# Set a title of the current axes.
	plt.title('Hyperparam Tuning for SA, Training Set')		
	plt.legend()
	plt.savefig('HyperSA_NNTest_Acc')
	plt.close()	

	# x=iters
	# plt.plot(x, sa_test_geom_f1, label = "Schedule-GeomDecay")
	# plt.plot(x, sa_test_arith_f1, label = "Schedule-ArithDecay")
	# plt.plot(x, sa_test_exp_f1, label = "Schedule-ExpDecay")
	# plt.xlabel('Iters')
	# # Set the y axis label of the current axis.
	# plt.ylabel('Precision Score')
	# # Set a title of the current axes.
	# plt.title('Hyperparam Tuning for SA, Testing Set, Precision')		
	# plt.legend()
	# plt.savefig('HyperSA_NNTest')
	# plt.close()	

	# x = list(range(1,iters))
	# plt.plot(x, rhc_f1_train, label = "Training")
	# plt.plot(x, rhc_f1_test, label = "Testing")
	# plt.xlabel('Iters')
	# # Set the y axis label of the current axis.
	# plt.ylabel('Computation Time (s)')
	# # Set a title of the current axes.
	# plt.title('Computation Time Evaluation for RHC')		
	# plt.legend()
	# plt.savefig('RHC_NNOpt_CompTime')
	# plt.close()

def ga_opt(X_train,Y_train, X_test, Y_test):
	'''
	https://readthedocs.org/projects/mlrose/downloads/pdf/stable/
	'''

	iters = [1,50,100,150,200,250,300]
	pop=[100,300]
	m = [0.1, 0.3, 0.5, 0.7, 0.9]
	ga_train_100_1 = []
	ga_train_100_3 = []
	ga_train_100_5 = []
	ga_train_100_7 = []
	ga_train_100_9 = []
	ga_train_300_1 = []
	ga_train_300_3 = []
	ga_train_300_5 = []
	ga_train_300_7 = []
	ga_train_300_9 = []

	ga_train_100_1_acc = []
	ga_train_100_3_acc = []
	ga_train_100_5_acc = []
	ga_train_100_7_acc = []
	ga_train_100_9_acc = []
	ga_train_300_1_acc = []
	ga_train_300_3_acc = []
	ga_train_300_5_acc = []
	ga_train_300_7_acc = []
	ga_train_300_9_acc = []

	sa_f1_train = []
	ga_f1_train = []

	ga_test_100_1 = []
	ga_test_100_3 = []
	ga_test_100_5 = []
	ga_test_100_7 = []
	ga_test_100_9 = []
	ga_test_300_1 = []
	ga_test_300_3 = []
	ga_test_300_5 = []
	ga_test_300_7 = []
	ga_test_300_9 = []

	ga_test_100_1_acc = []
	ga_test_100_3_acc = []
	ga_test_100_5_acc = []
	ga_test_100_7_acc = []
	ga_test_100_9_acc = []
	ga_test_300_1_acc = []
	ga_test_300_3_acc = []
	ga_test_300_5_acc = []
	ga_test_300_7_acc = []
	ga_test_300_9_acc = []

	sa_f1_test = []
	ga_f1_test = []
	for j in pop:
		for k in m:
			for i in iters:
				print(str(j) + "-", str(k) + "," + str(i))
				# Initialize neural network object and fit object
				ga = mlrose.NeuralNetwork(hidden_nodes = [30], activation = 'relu',algorithm = 'genetic_alg', max_iters = i,bias = True, early_stopping=True,pop_size=j,mutation_prob=k, random_state = 2)
				ga.fit(X_train,Y_train)
					
			# 		if (i==100):
				# Predict labels for train set and assess accuracy
				
				if(j==100 and k==0.1):
					y_train_pred = ga.predict(X_train)
					y_train_f1 = f1_score(Y_train, y_train_pred)
					ga_train_100_1.append(y_train_f1)
					print(ga_train_100_1)
					ga_train_100_1_acc.append(accuracy_score(Y_train, y_train_pred))
					print(ga_train_100_1_acc)
					y_test_pred = ga.predict(X_test)
					y_test_f1 = f1_score(Y_test, y_test_pred)
					ga_test_100_1.append(y_test_f1)
					ga_test_100_1_acc.append(accuracy_score(Y_test, y_test_pred))
				if(j==100 and k==0.3):
					y_train_pred = ga.predict(X_train)
					y_train_f1 = f1_score(Y_train, y_train_pred)
					ga_train_100_3.append(y_train_f1)
					print(ga_train_100_3)
					ga_train_100_3_acc.append(accuracy_score(Y_train, y_train_pred))
					print(ga_train_100_3_acc)
					y_test_pred = ga.predict(X_test)
					y_test_f1 = f1_score(Y_test, y_test_pred)
					ga_test_100_3.append(y_test_f1)
					ga_test_100_3_acc.append(accuracy_score(Y_test, y_test_pred))
				if(j==100 and k==0.5):
					y_train_pred = ga.predict(X_train)
					y_train_f1 = f1_score(Y_train, y_train_pred)
					ga_train_100_5.append(y_train_f1)
					print(ga_train_100_5)
					ga_train_100_5_acc.append(accuracy_score(Y_train, y_train_pred))
					print(ga_train_100_5_acc)
					y_test_pred = ga.predict(X_test)
					y_test_f1 = f1_score(Y_test, y_test_pred)
					ga_test_100_5.append(y_test_f1)
					ga_test_100_5_acc.append(accuracy_score(Y_test, y_test_pred))
				if(j==100 and k==0.7):
					y_train_pred = ga.predict(X_train)
					y_train_f1 = f1_score(Y_train, y_train_pred)
					ga_train_100_7.append(y_train_f1)
					print(ga_train_100_7)
					ga_train_100_7_acc.append(accuracy_score(Y_train, y_train_pred))
					print(ga_train_100_7_acc)
					y_test_pred = ga.predict(X_test)
					y_test_f1 = f1_score(Y_test, y_test_pred)
					ga_test_100_7.append(y_test_f1)
					ga_test_100_7_acc.append(accuracy_score(Y_test, y_test_pred))
				if(j==100 and k==0.9):
					y_train_pred = ga.predict(X_train)
					y_train_f1 = f1_score(Y_train, y_train_pred)
					ga_train_100_9.append(y_train_f1)
					print(ga_train_100_9)
					ga_train_100_9_acc.append(accuracy_score(Y_train, y_train_pred))
					print(ga_train_100_9_acc)
					y_test_pred = ga.predict(X_test)
					y_test_f1 = f1_score(Y_test, y_test_pred)
					ga_test_100_9.append(y_test_f1)	
					ga_test_100_9_acc.append(accuracy_score(Y_test, y_test_pred))
				if(j==300 and k==0.1):
					y_train_pred = ga.predict(X_train)
					y_train_f1 = f1_score(Y_train, y_train_pred)
					ga_train_300_1.append(y_train_f1)
					print(ga_train_300_1)
					ga_train_300_1_acc.append(accuracy_score(Y_train, y_train_pred))
					print(ga_train_300_1_acc)
					y_test_pred = ga.predict(X_test)
					y_test_f1 = f1_score(Y_test, y_test_pred)
					ga_test_300_1.append(y_test_f1)	
					ga_test_300_1_acc.append(accuracy_score(Y_test, y_test_pred))
				if(j==300 and k==0.3):
					y_train_pred = ga.predict(X_train)
					y_train_f1 = f1_score(Y_train, y_train_pred)
					ga_train_300_3.append(y_train_f1)
					print(ga_train_300_3)
					ga_train_300_3_acc.append(accuracy_score(Y_train, y_train_pred))
					print(ga_train_300_3_acc)
					y_test_pred = ga.predict(X_test)
					y_test_f1 = f1_score(Y_test, y_test_pred)
					ga_test_300_3.append(y_test_f1)	
					ga_test_300_3_acc.append(accuracy_score(Y_test, y_test_pred))
				if(j==300 and k==0.5):
					y_train_pred = ga.predict(X_train)
					y_train_f1 = f1_score(Y_train, y_train_pred)
					ga_train_300_5.append(y_train_f1)
					print(ga_train_300_5)
					ga_train_300_5_acc.append(accuracy_score(Y_train, y_train_pred))
					print(ga_train_300_5_acc)
					y_test_pred = ga.predict(X_test)
					y_test_f1 = f1_score(Y_test, y_test_pred)
					ga_test_300_5.append(y_test_f1)	
					ga_test_300_5_acc.append(accuracy_score(Y_test, y_test_pred))
				if(j==300 and k==0.7):
					y_train_pred = ga.predict(X_train)
					y_train_f1 = f1_score(Y_train, y_train_pred)
					ga_train_300_7.append(y_train_f1)
					print(ga_train_300_7)
					ga_train_300_7_acc.append(accuracy_score(Y_train, y_train_pred))
					print(ga_train_300_7_acc)
					y_test_pred = ga.predict(X_test)
					y_test_f1 = f1_score(Y_test, y_test_pred)
					ga_test_300_7.append(y_test_f1)	
					ga_test_300_7_acc.append(accuracy_score(Y_test, y_test_pred))
				if(j==300 and k==0.9):
					y_train_pred = ga.predict(X_train)
					y_train_f1 = f1_score(Y_train, y_train_pred)
					ga_train_300_9.append(y_train_f1)
					print(ga_train_300_9)
					ga_train_300_9_acc.append(accuracy_score(Y_train, y_train_pred))
					print(ga_train_300_9_acc)
					y_test_pred = ga.predict(X_test)
					y_test_f1 = f1_score(Y_test, y_test_pred)
					ga_test_300_9.append(y_test_f1)	
					ga_test_300_9_acc.append(accuracy_score(Y_test, y_test_pred))																																												
			# 		if(i==300):
			# 			# Predict labels for train set and assess accuracy
			# 			y_train_pred = rhc.predict(X_train)
			# 			y_train_f1 = f1_score(Y_train, y_train_pred)
			# 			rhc_f1_train_300.append(y_train_f1)
			# 			y_test_pred = rhc.predict(X_test)
			# 			y_test_f1 = f1_score(Y_test, y_test_pred)
			# 			rhc_f1_test_300.append(y_test_f1)
			# 		if(i==500):
			# 			# Predict labels for train set and assess accuracy
			# 			y_train_pred = rhc.predict(X_train)
			# 			y_train_f1 = f1_score(Y_train, y_train_pred)
			# 			rhc_f1_train_500.append(y_train_f1)
			# 			y_test_pred = rhc.predict(X_test)
			# 			y_test_f1 = f1_score(Y_test, y_test_pred)
			# 			rhc_f1_test_500.append(y_test_f1)				


	x = iters
	plt.plot(x, ga_train_100_1, label = "Pop100-MProb0.1")
	plt.plot(x, ga_train_100_3, label = "Pop100-MProb0.3")
	plt.plot(x, ga_train_100_5, label = "Pop100-MProb0.5")
	plt.plot(x, ga_train_100_7, label = "Pop100-MProb0.7")
	plt.plot(x, ga_train_100_9, label = "Pop100-MProb0.9")
	plt.plot(x, ga_train_300_1, label = "Pop300-MProb0.1")
	plt.plot(x, ga_train_300_3, label = "Pop300-MProb0.3")
	plt.plot(x, ga_train_300_5, label = "Pop300-MProb0.5")
	plt.plot(x, ga_train_300_7, label = "Pop300-MProb0.7")
	plt.plot(x, ga_train_300_9, label = "Pop300-MProb0.9")
	# plt.plot(x, rhc_f1_train_100, label = "Training-Pop300")
	# plt.plot(x, rhc_f1_test, label = "Testing-Pop300")
	# plt.plot(x, rhc_f1_train_100, label = "Training-Pop500")
	# plt.plot(x, rhc_f1_test, label = "Testing-Pop500")		
	plt.xlabel('Iters')
	# Set the y axis label of the current axis.
	plt.ylabel('F1 Score')
	# Set a title of the current axes.
	plt.title('Hyperparam Tuning for GA, Training Set')		
	plt.legend()
	plt.savefig('HyperGA_NNTrain_F1')
	plt.close()

	x = iters
	plt.plot(x, ga_test_100_1, label = "Pop100-MProb0.1")
	plt.plot(x, ga_test_100_3, label = "Pop100-MProb0.3")
	plt.plot(x, ga_test_100_5, label = "Pop100-MProb0.5")
	plt.plot(x, ga_test_100_7, label = "Pop100-MProb0.7")
	plt.plot(x, ga_test_100_9, label = "Pop100-MProb0.9")
	plt.plot(x, ga_test_300_1, label = "Pop300-MProb0.1")
	plt.plot(x, ga_test_300_3, label = "Pop300-MProb0.3")
	plt.plot(x, ga_test_300_5, label = "Pop300-MProb0.5")
	plt.plot(x, ga_test_300_7, label = "Pop300-MProb0.7")
	plt.plot(x, ga_test_300_9, label = "Pop300-MProb0.9")
	# plt.plot(x, rhc_f1_train_100, label = "Training-Pop300")
	# plt.plot(x, rhc_f1_test, label = "Testing-Pop300")
	# plt.plot(x, rhc_f1_train_100, label = "Training-Pop500")
	# plt.plot(x, rhc_f1_test, label = "Testing-Pop500")		
	plt.xlabel('Iters')
	# Set the y axis label of the current axis.
	plt.ylabel('F1 Score')
	# Set a title of the current axes.
	plt.title('Hyperparam Tuning for GA, Testing Set')		
	plt.legend()
	plt.savefig('HyperGA_NNTest_F1')
	plt.close()
	# x = list(range(1,iters))
	# plt.plot(x, rhc_f1_train, label = "Training")
	# plt.plot(x, rhc_f1_test, label = "Testing")
	# plt.xlabel('Iters')
	# # Set the y axis label of the current axis.
	# plt.ylabel('Computation Time (s)')
	# # Set a title of the current axes.
	# plt.title('Computation Time Evaluation for RHC')		
	# plt.legend()
	# plt.savefig('RHC_NNOpt_CompTime')
	# plt.close()

	x = iters
	plt.plot(x, ga_train_100_1_acc, label = "Pop100-MProb0.1")
	plt.plot(x, ga_train_100_3_acc, label = "Pop100-MProb0.3")
	plt.plot(x, ga_train_100_5_acc, label = "Pop100-MProb0.5")
	plt.plot(x, ga_train_100_7_acc, label = "Pop100-MProb0.7")
	plt.plot(x, ga_train_100_9_acc, label = "Pop100-MProb0.9")
	plt.plot(x, ga_train_300_1_acc, label = "Pop300-MProb0.1")
	plt.plot(x, ga_train_300_3_acc, label = "Pop300-MProb0.3")
	plt.plot(x, ga_train_300_5_acc, label = "Pop300-MProb0.5")
	plt.plot(x, ga_train_300_7_acc, label = "Pop300-MProb0.7")
	plt.plot(x, ga_train_300_9_acc, label = "Pop300-MProb0.9")
	# plt.plot(x, rhc_f1_train_100, label = "Training-Pop300")
	# plt.plot(x, rhc_f1_test, label = "Testing-Pop300")
	# plt.plot(x, rhc_f1_train_100, label = "Training-Pop500")
	# plt.plot(x, rhc_f1_test, label = "Testing-Pop500")		
	plt.xlabel('Iters')
	# Set the y axis label of the current axis.
	plt.ylabel('Accuracy Score')
	# Set a title of the current axes.
	plt.title('Hyperparam Tuning for GA, Training Set')		
	plt.legend()
	plt.savefig('HyperGA_NNTrain_Acc')
	plt.close()

	x = iters
	plt.plot(x, ga_test_100_1_acc, label = "Pop100-MProb0.1")
	plt.plot(x, ga_test_100_3_acc, label = "Pop100-MProb0.3")
	plt.plot(x, ga_test_100_5_acc, label = "Pop100-MProb0.5")
	plt.plot(x, ga_test_100_7_acc, label = "Pop100-MProb0.7")
	plt.plot(x, ga_test_100_9_acc, label = "Pop100-MProb0.9")
	plt.plot(x, ga_test_300_1_acc, label = "Pop300-MProb0.1")
	plt.plot(x, ga_test_300_3_acc, label = "Pop300-MProb0.3")
	plt.plot(x, ga_test_300_5_acc, label = "Pop300-MProb0.5")
	plt.plot(x, ga_test_300_7_acc, label = "Pop300-MProb0.7")
	plt.plot(x, ga_test_300_9_acc, label = "Pop300-MProb0.9")
	# plt.plot(x, rhc_f1_train_100, label = "Training-Pop300")
	# plt.plot(x, rhc_f1_test, label = "Testing-Pop300")
	# plt.plot(x, rhc_f1_train_100, label = "Training-Pop500")
	# plt.plot(x, rhc_f1_test, label = "Testing-Pop500")		
	plt.xlabel('Iters')
	# Set the y axis label of the current axis.
	plt.ylabel('Accuracy Score')
	# Set a title of the current axes.
	plt.title('Hyperparam Tuning for GA, Testing Set')		
	plt.legend()
	plt.savefig('HyperGA_NNTest_Acc')
	plt.close()
	# x = list(range(1,iters))
	# plt.plot(x, rhc_f1_train, label = "Training")
	# plt.plot(x, rhc_f1_test, label = "Testing")
	# plt.xlabel('Iters')
	# # Set the y axis label of the current axis.
	# plt.ylabel('Computation Time (s)')
	# # Set a title of the current axes.
	# plt.title('Computation Time Evaluation for RHC')		
	# plt.legend()
	# plt.savefig('RHC_NNOpt_CompTime')
	# plt.close()	

def opt_weights(X_train,Y_train, X_test, Y_test):
	'''
	https://readthedocs.org/projects/mlrose/downloads/pdf/stable/
	'''
	iters=1001
	rhc_f1_train = []
	sa_f1_train = []
	ga_f1_train = []

	rhc_f1_test = []
	sa_f1_test = []
	ga_f1_test = []
	for i in range(1,iters):
		# Initialize neural network object and fit object
		rhc = mlrose.NeuralNetwork(hidden_nodes = [30], activation = 'relu',algorithm = 'random_hill_climb', max_iters = i,bias = True,random_state = 2)
		rhc.fit(X_train,Y_train)
		
		sa = mlrose.NeuralNetwork(hidden_nodes = [30], activation = 'relu',algorithm = 'simulated_annealing', max_iters = i,bias = True,random_state = 2)
		sa.fit(X_train,Y_train)

		ga = mlrose.NeuralNetwork(hidden_nodes = [30], activation = 'relu',algorithm = 'genetic_alg', max_iters = i,bias = True,random_state = 2)
		ga.fit(X_train,Y_train)

		# Predict labels for train set and assess accuracy
		y_train_pred = rhc.predict(X_train)
		y_train_pred = rhc.predict(X_train)
		y_train_pred = rhc.predict(X_train)
		y_train_f1 = f1_score(Y_train, y_train_pred)
		rhc_f1_train.append(y_train_f1)
		y_test_pred = rhc.predict(X_test)
		y_test_f1 = f1_score(Y_test, y_test_pred)
		f1_test.append(y_test_f1)

	x = list(range(1,iters))
	plt.plot(x, f1_train, label = "RHC")
	plt.xlabel('Iters')
	# Set the y axis label of the current axis.
	plt.ylabel('F1 Score')
	# Set a title of the current axes.
	plt.title('F1 Score for Training')		
	plt.legend()
	plt.savefig('RHC_NNOpt')
	plt.close()	

def compare_algos(X_train, Y_train, X_test, Y_test):
	iters = [500,1000,1500,2000,2500,3000]

	rhc_f1_train = []
	rhc_compute = []
	sa_f1_train = []
	sa_compute = []
	ga_f1_train = []
	ga_compute = []
	backprop_f1_train = []
	backprop_compute = []

	rhc_f1_test = []
	sa_f1_test = []
	ga_f1_test = []	
	backprop_f1_test = []

	for i in iters:

		rhc = mlrose.NeuralNetwork(hidden_nodes = [30], activation = 'relu',algorithm = 'random_hill_climb', early_stopping=True, max_iters = 1000,bias = True,random_state = 2)
		time_start_rhc = time.time()
		rhc.fit(X_train,Y_train)
		time_elapsed_rhc = (time.time() - time_start_rhc)
		rhc_compute.append(time_elapsed_rhc)

		y_train_pred = rhc.predict(X_train)
		y_train_f1 = f1_score(Y_train, y_train_pred)
		rhc_f1_train.append(y_train_f1)
		y_test_pred = rhc.predict(X_test)
		y_test_f1 = f1_score(Y_test, y_test_pred)
		rhc_f1_test.append(y_test_f1)		
		
		sa = mlrose.NeuralNetwork(hidden_nodes = [30], activation = 'relu',algorithm = 'simulated_annealing', schedule=mlrose.ArithDecay(), early_stopping=True, max_iters = 1000,bias = True,random_state = 2)
		time_start_sa = time.time()	
		sa.fit(X_train,Y_train)
		time_elapsed_sa = (time.time() - time_start_sa)	
		sa_compute.append(time_elapsed_sa)

		y_train_pred = sa.predict(X_train)
		y_train_f1 = f1_score(Y_train, y_train_pred)
		sa_f1_train.append(y_train_f1)
		y_test_pred = sa.predict(X_test)
		y_test_f1 = f1_score(Y_test, y_test_pred)
		sa_f1_test.append(y_test_f1)	

		ga = mlrose.NeuralNetwork(hidden_nodes = [30], activation = 'relu',algorithm = 'genetic_alg', mutation_prob=0.9, max_iters = 1000, early_stopping=True, bias = True,random_state = 2)
		time_start_ga = time.time()	
		ga.fit(X_train,Y_train)
		time_elapsed_ga = (time.time() - time_start_ga)	
		ga_compute.append(time_elapsed_ga)

		y_train_pred = ga.predict(X_train)
		y_train_f1 = f1_score(Y_train, y_train_pred)
		ga_f1_train.append(y_train_f1)
		y_test_pred = ga.predict(X_test)
		y_test_f1 = f1_score(Y_test, y_test_pred)
		ga_f1_test.append(y_test_f1)			

		backprop = MLPClassifier(random_state=2, hidden_layer_sizes=30, activation='relu', solver='sgd')
		time_start_backprop = time.time()	
		backprop.fit(X_train, Y_train)
		time_elapsed_backprop = (time.time() - time_start_backprop)		
		sa_compute.append(time_elapsed_sa)

		y_train_pred = backprop.predict(X_train)
		y_train_f1 = f1_score(Y_train, y_train_pred)
		backprop_f1_train.append(y_train_f1)
		y_test_pred = backprop.predict(X_test)
		y_test_f1 = f1_score(Y_test, y_test_pred)
		backprop_f1_test.append(y_test_f1)	

	x = iters
	plt.plot(x, rhc_f1_train, label = "RHC")
	plt.plot(x, sa_f1_train, label = "SA")
	plt.plot(x, ga_f1_train, label = "GA")
	plt.plot(x, backprop_f1_train, label = "Backprop")
	# plt.plot(x, rhc_f1_train_100, label = "Training-Pop300")
	# plt.plot(x, rhc_f1_test, label = "Testing-Pop300")
	# plt.plot(x, rhc_f1_train_100, label = "Training-Pop500")
	# plt.plot(x, rhc_f1_test, label = "Testing-Pop500")		
	plt.xlabel('Iters')
	# Set the y axis label of the current axis.
	plt.ylabel('F1 Score')
	# Set a title of the current axes.
	plt.title('F1 Score Comparison for Opt Algos')		
	plt.legend()
	plt.savefig('GA_NNOpt_F1')
	plt.close()

	# x = list(range(1,iters))
	# plt.plot(x, rhc_f1_train, label = "Training")
	# plt.plot(x, rhc_f1_test, label = "Testing")
	# plt.xlabel('Iters')
	# # Set the y axis label of the current axis.
	# plt.ylabel('Computation Time (s)')
	# # Set a title of the current axes.
	# plt.title('Computation Time Evaluation for RHC')		
	# plt.legend()
	# plt.savefig('RHC_NNOpt_CompTime')
	# plt.close()

def compare_algos(X_train, Y_train, X_test, Y_test):
	iters = [1,500,1000,1500,2000,2500,3000,3500,4000]

	rhc_f1_train = []
	rhc_acc_train = []
	rhc_compute = []
	sa_f1_train = []
	sa_acc_train = []
	sa_compute = []
	ga_f1_train = []
	ga_acc_train = []
	ga_compute = []
	backprop_f1_train = []
	backprop_acc_train = []
	backprop_compute = []

	rhc_f1_test = []
	rhc_acc_test = []
	sa_f1_test = []
	sa_acc_test = []
	ga_f1_test = []
	ga_acc_test = []		
	backprop_f1_test = []
	backprop_acc_test = []

	# for i in iters:
	# 	print(i)

	rhc = mlrose.NeuralNetwork(hidden_nodes = [30], activation = 'relu',algorithm = 'random_hill_climb', early_stopping=True, max_iters = 4000,bias = True,random_state = 2)
	time_start_rhc = time.time()
	print("RHC start")
	rhc.fit(X_train,Y_train)
	time_elapsed_rhc = (time.time() - time_start_rhc)
	rhc_compute.append(time_elapsed_rhc)

	y_train_pred = rhc.predict(X_train)
	y_train_f1 = f1_score(Y_train, y_train_pred)
	rhc_f1_train.append(y_train_f1)
	print(rhc_f1_train)
	rhc_acc_train.append(accuracy_score(Y_train, y_train_pred))
	print(rhc_acc_train)
	y_test_pred = rhc.predict(X_test)
	y_test_f1 = f1_score(Y_test, y_test_pred)
	rhc_f1_test.append(y_test_f1)
	rhc_acc_test.append(accuracy_score(Y_test, y_test_pred))		
	
	sa = mlrose.NeuralNetwork(hidden_nodes = [30], activation = 'relu',algorithm = 'simulated_annealing', schedule=mlrose.ExpDecay(), early_stopping=True, max_iters = 4000,bias = True,random_state = 2)
	time_start_sa = time.time()	
	print("SA start")
	sa.fit(X_train,Y_train)
	time_elapsed_sa = (time.time() - time_start_sa)	
	sa_compute.append(time_elapsed_sa)

	y_train_pred = sa.predict(X_train)
	y_train_f1 = f1_score(Y_train, y_train_pred)
	sa_f1_train.append(y_train_f1)
	print(sa_f1_train)
	sa_acc_train.append(accuracy_score(Y_train, y_train_pred))
	print(sa_acc_train)
	y_test_pred = sa.predict(X_test)
	y_test_f1 = f1_score(Y_test, y_test_pred)
	sa_f1_test.append(y_test_f1)
	sa_acc_test.append(accuracy_score(Y_test, y_test_pred))	

	ga = mlrose.NeuralNetwork(hidden_nodes = [30], activation = 'relu',algorithm = 'genetic_alg', pop_size=300,mutation_prob=0.5, max_iters = 4000, early_stopping=True, bias = True,random_state = 2)
	time_start_ga = time.time()	
	print("GA start")
	ga.fit(X_train,Y_train)
	time_elapsed_ga = (time.time() - time_start_ga)	
	ga_compute.append(time_elapsed_ga)

	y_train_pred = ga.predict(X_train)
	y_train_f1 = f1_score(Y_train, y_train_pred)
	ga_f1_train.append(y_train_f1)
	print(ga_f1_train)
	ga_acc_train.append(accuracy_score(Y_train, y_train_pred))
	print(ga_acc_train)
	y_test_pred = ga.predict(X_test)
	y_test_f1 = f1_score(Y_test, y_test_pred)
	ga_f1_test.append(y_test_f1)	
	ga_acc_test.append(accuracy_score(Y_test, y_test_pred))		

	backprop = MLPClassifier(random_state=2, hidden_layer_sizes=30, activation='relu', solver='sgd',max_iter=4000)
	time_start_backprop = time.time()	
	print("BP start")
	backprop.fit(X_train, Y_train)
	time_elapsed_backprop = (time.time() - time_start_backprop)		
	backprop_compute.append(time_elapsed_sa)

	y_train_pred = backprop.predict(X_train)
	y_train_f1 = f1_score(Y_train, y_train_pred)
	backprop_f1_train.append(y_train_f1)
	print(backprop_f1_train)
	backprop_acc_train.append(accuracy_score(Y_train, y_train_pred))
	print(backprop_acc_train)
	y_test_pred = backprop.predict(X_test)
	y_test_f1 = f1_score(Y_test, y_test_pred)
	backprop_f1_test.append(y_test_f1)	
	backprop_acc_test.append(accuracy_score(Y_test, y_test_pred))

	print('----TRAIN_F1-----')
	print('RHC: ')
	print(rhc_f1_train)
	print('SA: ')
	print(sa_f1_train)
	print('GA: ')
	print(ga_f1_train)
	print('BP: ')
	print(backprop_f1_train)

	print('----TEST F1-----')
	print('RHC: ')
	print(rhc_f1_test)
	print('SA: ')
	print(sa_f1_test)
	print('GA: ')
	print(ga_f1_test)
	print('BP: ')
	print(backprop_f1_test)	

	print('----TRAIN_ACC-----')
	print('RHC: ')
	print(rhc_acc_train)
	print('SA: ')
	print(sa_acc_train)
	print('GA: ')
	print(ga_acc_train)
	print('BP: ')
	print(backprop_acc_train)

	print('----TEST ACC-----')
	print('RHC: ')
	print(rhc_acc_test)
	print('SA: ')
	print(sa_acc_test)
	print('GA: ')
	print(ga_acc_test)
	print('BP: ')
	print(backprop_acc_test)		

	print('----COMPUTE TIME-----')
	print('RHC: ')
	print(rhc_compute)
	print('SA: ')
	print(sa_compute)
	print('GA: ')
	print(ga_compute)
	print('BP: ')
	print(backprop_compute)			


	# x = iters
	# plt.plot(x, rhc_f1_train, label = "RHC")
	# plt.plot(x, sa_f1_train, label = "SA")
	# plt.plot(x, ga_f1_train, label = "GA")
	# plt.plot(x, backprop_f1_train, label = "Backprop")
	# # plt.plot(x, rhc_f1_train_100, label = "Training-Pop300")
	# # plt.plot(x, rhc_f1_test, label = "Testing-Pop300")
	# # plt.plot(x, rhc_f1_train_100, label = "Training-Pop500")
	# # plt.plot(x, rhc_f1_test, label = "Testing-Pop500")		
	# plt.xlabel('Iters')
	# # Set the y axis label of the current axis.
	# plt.ylabel('F1 Score')
	# # Set a title of the current axes.
	# plt.title('Algo Performance for Training Set')		
	# plt.legend()
	# plt.savefig('CompareAlgos_Train_F1')
	# plt.close()

	# x = iters
	# plt.plot(x, rhc_acc_train, label = "RHC")
	# plt.plot(x, sa_acc_train, label = "SA")
	# plt.plot(x, ga_acc_train, label = "GA")
	# plt.plot(x, backprop_acc_train, label = "Backprop")
	# # plt.plot(x, rhc_f1_train_100, label = "Training-Pop300")
	# # plt.plot(x, rhc_f1_test, label = "Testing-Pop300")
	# # plt.plot(x, rhc_f1_train_100, label = "Training-Pop500")
	# # plt.plot(x, rhc_f1_test, label = "Testing-Pop500")		
	# plt.xlabel('Iters')
	# # Set the y axis label of the current axis.
	# plt.ylabel('Accuracy Score')
	# # Set a title of the current axes.
	# plt.title('Algo Performance for Training Set')		
	# plt.legend()
	# plt.savefig('CompareAlgos_Train_Acc')
	# plt.close()

	# x = iters
	# plt.plot(x, rhc_f1_test, label = "RHC")
	# plt.plot(x, sa_f1_test, label = "SA")
	# plt.plot(x, ga_f1_test, label = "GA")
	# plt.plot(x, backprop_f1_test, label = "Backprop")
	# # plt.plot(x, rhc_f1_train_100, label = "Training-Pop300")
	# # plt.plot(x, rhc_f1_test, label = "Testing-Pop300")
	# # plt.plot(x, rhc_f1_train_100, label = "Training-Pop500")
	# # plt.plot(x, rhc_f1_test, label = "Testing-Pop500")		
	# plt.xlabel('Iters')
	# # Set the y axis label of the current axis.
	# plt.ylabel('F1 Score')
	# # Set a title of the current axes.
	# plt.title('Algo Performance for Testing Set')		
	# plt.legend()
	# plt.savefig('CompareAlgos_Test_F1')
	# plt.close()

	# x = iters
	# plt.plot(x, rhc_acc_test, label = "RHC")
	# plt.plot(x, sa_acc_test, label = "SA")
	# plt.plot(x, ga_acc_test, label = "GA")
	# plt.plot(x, backprop_acc_test, label = "Backprop")
	# # plt.plot(x, rhc_f1_train_100, label = "Training-Pop300")
	# # plt.plot(x, rhc_f1_test, label = "Testing-Pop300")
	# # plt.plot(x, rhc_f1_train_100, label = "Training-Pop500")
	# # plt.plot(x, rhc_f1_test, label = "Testing-Pop500")		
	# plt.xlabel('Iters')
	# # Set the y axis label of the current axis.
	# plt.ylabel('Accuracy Score')
	# # Set a title of the current axes.
	# plt.title('Algo Performance for Testing Set')		
	# plt.legend()
	# plt.savefig('CompareAlgos_Test_Acc')
	# plt.close()			

	# x = iters
	# plt.plot(x, rhc_compute, label = "RHC")
	# plt.plot(x, sa_compute, label = "SA")
	# plt.plot(x, ga_compute, label = "GA")
	# plt.plot(x, backprop_compute, label = "Backprop")
	# # plt.plot(x, rhc_f1_train_100, label = "Training-Pop300")
	# # plt.plot(x, rhc_f1_test, label = "Testing-Pop300")
	# # plt.plot(x, rhc_f1_train_100, label = "Training-Pop500")
	# # plt.plot(x, rhc_f1_test, label = "Testing-Pop500")		
	# plt.xlabel('Iters')
	# # Set the y axis label of the current axis.
	# plt.ylabel('Computation Time (s)')
	# # Set a title of the current axes.
	# plt.title('Algo Training Computation Time')		
	# plt.legend()
	# plt.savefig('CompareAlgos_ComputeTime_F1')
	# plt.close()		

	# x = list(range(1,iters))
	# plt.plot(x, rhc_f1_train, label = "Training")
	# plt.plot(x, rhc_f1_test, label = "Testing")
	# plt.xlabel('Iters')
	# # Set the y axis label of the current axis.
	# plt.ylabel('Computation Time (s)')
	# # Set a title of the current axes.
	# plt.title('Computation Time Evaluation for RHC')		
	# plt.legend()
	# plt.savefig('RHC_NNOpt_CompTime')
	# plt.close()

def main():
	# pData = pd.read_pickle('EEGEyeData.pkl')
	# pData['Class'] = pData['Class'].astype('category').cat.codes
	# X_pData = pData.iloc[:,:-1]
	# # X_pData = X_pData.replace('n',0)
	# # X_pData = X_pData.replace('y',1)
	# normalized_X_pData=(X_pData-X_pData.min())/(X_pData.max()-X_pData.min())
	# # X_pData = X_pData.replace('b',0)
	# # X_pData = X_pData.replace('o',1)
	# # X_pData = X_pData.replace('x',2)
	# y_pData = pData.iloc[:,-1]
	# X_train_pData, X_test_pData, y_train_pData, y_test_pData = train_test_split(normalized_X_pData, y_pData, test_size=0.3, random_state=RANDOM_STATE)
	
	# sa_opt(X_train_pData, y_train_pData, X_test_pData, y_test_pData)

	eData = pd.read_pickle('FlightDelays.pkl')

	X_eData = eData.iloc[:,:-1]
	X_eData['Airline'] = X_eData['Airline'].astype('category').cat.codes
	X_eData['Flight'] = X_eData['Flight'].astype('category').cat.codes
	X_eData['AirportFrom'] = X_eData['AirportFrom'].astype('category').cat.codes
	X_eData['AirportTo'] = X_eData['AirportTo'].astype('category').cat.codes
	normalized_X_eData=(X_eData-X_eData.min())/(X_eData.max()-X_eData.min())
	'''
	had to label encode categorical data and normalize
	'''
	y_eData = eData.iloc[:,-1]
	X_train_eData, X_test_eData, y_train_eData, y_test_eData = train_test_split(normalized_X_eData, y_eData, test_size=0.3, random_state=RANDOM_STATE)
	
	compare_algos(X_train_eData, y_train_eData, X_test_eData, y_test_eData)

	# display_metrics("Logistic Regression",logistic_regression_pred(X_train,Y_train),Y_train)
	# display_metrics("SVM",svm_pred(X_train,Y_train),Y_train)
	# display_metrics("Decision Tree",decisionTree_pred(X_train,Y_train),Y_train)
	

if __name__ == "__main__":
	main()
	
