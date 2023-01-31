import mlrose
import numpy as np
import matplotlib.pyplot as plt
import time
import random



def TSP_old():
	'''
	https://readthedocs.org/projects/mlrose/downloads/pdf/stable/
	https://stackoverflow.com/questions/11886862/calculating-computational-time-and-memory-for-a-code-in-python
	'''
	iters = [100,200,300,400,500,600,700,800,900,1000]
	# Create list of distances between pairs of cities
	coords_list = [tuple(random.sample(range(1,10), 2)) for x in range(100)]
	# print(len(coords_list))
	# print(len(set(coords_list)))

	# coords_list=[(1, 1), (4, 2), (5, 2), (6, 4), (4, 4), (3, 6), (1, 5), (2, 3)]
	# Initialize fitness function object using dist_list
	# fitness_coords = mlrose.TravellingSales(coords = coords_list)
	# problem_fit = mlrose.DiscreteOpt(length = 8, fitness_fn = fitness_coords, maximize=True)
	problem_no_fit = mlrose.TSPOpt(length = 100, coords = coords_list, maximize=True)
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

	x = iters
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

	x = iters
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

def flip_flop_old():
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

def flip_flop():
	'''
	https://readthedocs.org/projects/mlrose/downloads/pdf/stable/
	https://stackoverflow.com/questions/11886862/calculating-computational-time-and-memory-for-a-code-in-python
	'''
	iters=[1,100,200,300,400,500,600,700]
	problem_fit = mlrose.DiscreteOpt(length = 100,fitness_fn = mlrose.OneMax())
	
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
		best_state, best_fitness_ga = mlrose.genetic_alg(problem_fit, max_iters=i,random_state = 2, pop_size=100, mutation_prob=0.7)	
		time_elapsed_ga = (time.time() - time_start_ga)
		ga_time.append(time_elapsed_ga)
		ga_fit.append(best_fitness_ga)
		
		time_start_sa = time.time()
		best_state, best_fitness_sa = mlrose.simulated_annealing(problem_fit, max_iters=i,random_state = 2, schedule=mlrose.GeomDecay())	
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
	plt.title('Fitness Function Evaluation for FlipFlop')
	plt.legend()
	plt.savefig('FlipFlop_FitnessFunc')
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
	plt.title('Computation Time Evalutation for FlipFlop')		
	plt.legend()
	plt.savefig('FlipFlop_ComputeTime')
	plt.close()	

def four_peaks_old():
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
	iters=[1,200,400,600,800,1000,1200,1400]
	problem_fit = mlrose.DiscreteOpt(length = 100,fitness_fn = mlrose.FourPeaks(t_pct=0.15))
	
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
		best_state, best_fitness_ga = mlrose.genetic_alg(problem_fit, max_iters=i,random_state = 2, pop_size=100, mutation_prob=0.3)	
		time_elapsed_ga = (time.time() - time_start_ga)
		ga_time.append(time_elapsed_ga)
		ga_fit.append(best_fitness_ga)
		
		time_start_sa = time.time()
		best_state, best_fitness_sa = mlrose.simulated_annealing(problem_fit, max_iters=i,random_state = 2, schedule=mlrose.ExpDecay())	
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
	plt.title('Fitness Function Evaluation for FourPeaks')
	plt.legend()
	plt.savefig('FourPeaks_FitnessFunc')
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
	plt.title('Computation Time Evalutation for FourPeaks')		
	plt.legend()
	plt.savefig('FourPeaks_ComputeTime')
	plt.close()

def one_max():
	'''
	https://readthedocs.org/projects/mlrose/downloads/pdf/stable/
	https://stackoverflow.com/questions/11886862/calculating-computational-time-and-memory-for-a-code-in-python
	'''
	iters=[1,100,200,300,400]
	problem_fit = mlrose.DiscreteOpt(length = 150,fitness_fn = mlrose.OneMax())
	
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
		best_state, best_fitness_ga = mlrose.genetic_alg(problem_fit, max_iters=i,random_state = 2, pop_size=300, mutation_prob=0.1)	
		time_elapsed_ga = (time.time() - time_start_ga)
		ga_time.append(time_elapsed_ga)
		ga_fit.append(best_fitness_ga)
		
		time_start_sa = time.time()
		best_state, best_fitness_sa = mlrose.simulated_annealing(problem_fit, max_iters=i,random_state = 2, schedule=mlrose.ExpDecay())	
		time_elapsed_sa = (time.time() - time_start_sa)
		sa_time.append(time_elapsed_sa)		
		sa_fit.append(best_fitness_sa)
		
		time_start_mimic = time.time()
		best_state, best_fitness_mimic = mlrose.mimic(problem_fit, max_iters=i,random_state = 2, pop_size=100, keep_pct=0.7)	
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

def one_max_ga():
	'''
	https://readthedocs.org/projects/mlrose/downloads/pdf/stable/
	https://stackoverflow.com/questions/11886862/calculating-computational-time-and-memory-for-a-code-in-python
	'''
	iters=[1,100,200,300,400,500,600,700,800,900,1000]
	pop = [100,300]
	m = [0.1, 0.3, 0.5, 0.7, 0.9]
	problem_fit = mlrose.DiscreteOpt(length = 100,fitness_fn = mlrose.OneMax())
	
	ga_fit_100_1 = []
	ga_fit_100_3 = []
	ga_fit_100_5 = []
	ga_fit_100_7 = []
	ga_fit_100_9 = []
	ga_fit_300_1 = []
	ga_fit_300_3 = []
	ga_fit_300_5 = []
	ga_fit_300_7 = []
	ga_fit_300_9 = []	
	sa_fit = []
	mimic_fit = []
	rhc_fit = []
	ga_time = []
	sa_time = []
	mimic_time = []
	rhc_time = []

	for j in pop:
		for k in m:
			for i in iters:
				print(str(j) + "-", str(k) + "," + str(i))
				time_start_ga = time.time()
				best_state, best_fitness_ga = mlrose.genetic_alg(problem_fit, max_iters=i, pop_size=j, mutation_prob=k, random_state = 2)	
				time_elapsed_ga = (time.time() - time_start_ga)
				
				if (j==100 and k==0.1):
					ga_fit_100_1.append(best_fitness_ga)
				elif (j==100 and k==0.3):
					ga_fit_100_3.append(best_fitness_ga)
				elif (j==100 and k==0.5):
					ga_fit_100_5.append(best_fitness_ga)
				elif (j==100 and k==0.7):
					ga_fit_100_7.append(best_fitness_ga)
				elif (j==100 and k==0.9):
					ga_fit_100_9.append(best_fitness_ga)
				elif (j==300 and k==0.1):
					ga_fit_300_1.append(best_fitness_ga)
				elif (j==300 and k==0.3):
					ga_fit_300_3.append(best_fitness_ga)
				elif (j==300 and k==0.5):
					ga_fit_300_5.append(best_fitness_ga)	
				elif (j==300 and k==0.7):
					ga_fit_300_7.append(best_fitness_ga)
				elif (j==300 and k==0.9):
					ga_fit_300_9.append(best_fitness_ga)														

				# ga_time.append(time_elapsed_ga)
				# ga_fit.append(best_fitness_ga)
		
	x = iters
	plt.plot(x, ga_fit_100_1, label = "Pop100-MProb0.1")
	plt.plot(x, ga_fit_100_3, label = "Pop100-MProb0.3")
	plt.plot(x, ga_fit_100_5, label = "Pop100-MProb0.5")
	plt.plot(x, ga_fit_100_7, label = "Pop100-MProb0.7")
	plt.plot(x, ga_fit_100_9, label = "Pop100-MProb0.9")
	plt.plot(x, ga_fit_300_1, label = "Pop300-MProb0.1")
	plt.plot(x, ga_fit_300_3, label = "Pop300-MProb0.3")
	plt.plot(x, ga_fit_300_5, label = "Pop300-MProb0.5")
	plt.plot(x, ga_fit_300_7, label = "Pop300-MProb0.7")
	plt.plot(x, ga_fit_300_9, label = "Pop300-MProb0.9")

	plt.xlabel('Iters')
	# Set the y axis label of the current axis.
	plt.ylabel('Fitness function value')
	# Set a title of the current axes.
	plt.title('Hyperparam Tuning GA, OneMax Problem')
	plt.legend()
	plt.savefig('OneMax_GAHyper')
	plt.close()

	# x = iters
	# plt.plot(x, ga_time, label = "GA")
	# plt.plot(x, sa_time, label = "SA")
	# plt.plot(x, mimic_time, label = "MIMIC")
	# plt.plot(x, rhc_time, label = "RHC")
	# plt.xlabel('Iters')
	# # Set the y axis label of the current axis.
	# plt.ylabel('Computation Time (s)')
	# # Set a title of the current axes.
	# plt.title('Computation Time Evalutation for OneMax')		
	# plt.legend()
	# plt.savefig('OneMax_ComputeTime')
	# plt.close()

def one_max_mimic():
	'''
	https://readthedocs.org/projects/mlrose/downloads/pdf/stable/
	https://stackoverflow.com/questions/11886862/calculating-computational-time-and-memory-for-a-code-in-python
	'''
	iters=[1,100,200,300,400,500,600,700,800,900,1000]
	pop = [100,300]
	keep = [0.1, 0.3, 0.5, 0.7, 0.9]
	problem_fit = mlrose.DiscreteOpt(length = 100,fitness_fn = mlrose.OneMax())
	
	ga_fit_100_1 = []
	ga_fit_100_3 = []
	ga_fit_100_5 = []
	ga_fit_100_7 = []
	ga_fit_100_9 = []
	ga_fit_300_1 = []
	ga_fit_300_3 = []
	ga_fit_300_5 = []
	ga_fit_300_7 = []
	ga_fit_300_9 = []	
	sa_fit = []
	mimic_fit = []
	rhc_fit = []
	ga_time = []
	sa_time = []
	mimic_time = []
	rhc_time = []

	for j in pop:
		for k in keep:
			for i in iters:
				print(str(j) + "-", str(k) + "," + str(i))
				time_start_ga = time.time()
				best_state, best_fitness_ga = mlrose.mimic(problem_fit, max_iters=i, pop_size=j, keep_pct=k, random_state = 2)	
				time_elapsed_ga = (time.time() - time_start_ga)
				
				if (j==100 and k==0.1):
					ga_fit_100_1.append(best_fitness_ga)
				elif (j==100 and k==0.3):
					ga_fit_100_3.append(best_fitness_ga)
				elif (j==100 and k==0.5):
					ga_fit_100_5.append(best_fitness_ga)
				elif (j==100 and k==0.7):
					ga_fit_100_7.append(best_fitness_ga)
				elif (j==100 and k==0.9):
					ga_fit_100_9.append(best_fitness_ga)
				elif (j==300 and k==0.1):
					ga_fit_100_1.append(best_fitness_ga)
				elif (j==300 and k==0.3):
					ga_fit_300_3.append(best_fitness_ga)
				elif (j==300 and k==0.5):
					ga_fit_300_5.append(best_fitness_ga)	
				elif (j==300 and k==0.7):
					ga_fit_300_7.append(best_fitness_ga)
				elif (j==300 and k==0.9):
					ga_fit_300_9.append(best_fitness_ga)														

				# ga_time.append(time_elapsed_ga)
				# ga_fit.append(best_fitness_ga)
		
	x = iters
	plt.plot(x, ga_fit_100_1, label = "Pop100-Keep0.1")
	plt.plot(x, ga_fit_100_3, label = "Pop100-Keep0.3")
	plt.plot(x, ga_fit_100_5, label = "Pop100-Keep0.5")
	plt.plot(x, ga_fit_100_7, label = "Pop100-Keep0.7")
	plt.plot(x, ga_fit_100_9, label = "Pop100-Keep0.9")
	plt.plot(x, ga_fit_300_1, label = "Pop300-Keep0.1")
	plt.plot(x, ga_fit_300_3, label = "Pop300-Keep0.3")
	plt.plot(x, ga_fit_300_5, label = "Pop300-Keep0.5")
	plt.plot(x, ga_fit_300_7, label = "Pop300-Keep0.7")
	plt.plot(x, ga_fit_300_9, label = "Pop300-Keep0.9")

	plt.xlabel('Iters')
	# Set the y axis label of the current axis.
	plt.ylabel('Fitness function value')
	# Set a title of the current axes.
	plt.title('Hyperparam Tuning MIMIC, OneMax Problem')
	plt.legend()
	plt.savefig('OneMax_MIMICHyper')
	plt.close()

	# x = iters
	# plt.plot(x, ga_time, label = "GA")
	# plt.plot(x, sa_time, label = "SA")
	# plt.plot(x, mimic_time, label = "MIMIC")
	# plt.plot(x, rhc_time, label = "RHC")
	# plt.xlabel('Iters')
	# # Set the y axis label of the current axis.
	# plt.ylabel('Computation Time (s)')
	# # Set a title of the current axes.
	# plt.title('Computation Time Evalutation for OneMax')		
	# plt.legend()
	# plt.savefig('OneMax_ComputeTime')
	# plt.close()


def one_max_sa():
	'''
	https://readthedocs.org/projects/mlrose/downloads/pdf/stable/
	https://stackoverflow.com/questions/11886862/calculating-computational-time-and-memory-for-a-code-in-python
	'''
	iters=[1,100,200,300,400,500,600,700,800,900,1000]
	schedule=[mlrose.GeomDecay(), mlrose.ArithDecay(), mlrose.ExpDecay()]
	problem_fit = mlrose.DiscreteOpt(length = 100,fitness_fn = mlrose.OneMax())
	
	ga_fit_geom = []
	ga_fit_arith = []
	ga_fit_exp = []
	sa_fit = []
	mimic_fit = []
	rhc_fit = []
	ga_time = []
	sa_time = []
	mimic_time = []
	rhc_time = []

	for j in range(0,len(schedule)):
		for i in iters:
			print(str(j) + "," + str(i))
			time_start_ga = time.time()
			best_state, best_fitness_ga = mlrose.simulated_annealing(problem_fit, max_iters=i, schedule=schedule[j], random_state = 2)	
			time_elapsed_ga = (time.time() - time_start_ga)
			
			if (j==0):
				ga_fit_geom.append(best_fitness_ga)
			elif (j==1):
				ga_fit_arith.append(best_fitness_ga)
			elif (j==2):
				ga_fit_exp.append(best_fitness_ga)														

			# ga_time.append(time_elapsed_ga)
			# ga_fit.append(best_fitness_ga)
		
	x = iters
	plt.plot(x, ga_fit_geom, label = "Schedule-GeomDecay")
	plt.plot(x, ga_fit_arith, label = "Schedule-ArithDecay")
	plt.plot(x, ga_fit_exp, label = "Schedule-ExpDecay")

	plt.xlabel('Iters')
	# Set the y axis label of the current axis.
	plt.ylabel('Fitness function value')
	# Set a title of the current axes.
	plt.title('Hyperparam Tuning SA, OneMax Problem')
	plt.legend()
	plt.savefig('OneMax_SAHyper')
	plt.close()	

def four_peaks_ga():
	'''
	https://readthedocs.org/projects/mlrose/downloads/pdf/stable/
	https://stackoverflow.com/questions/11886862/calculating-computational-time-and-memory-for-a-code-in-python
	'''
	iters=[1,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300]
	pop = [100,300]
	m = [0.1, 0.3, 0.5, 0.7, 0.9]
	problem_fit = mlrose.DiscreteOpt(length = 100,fitness_fn = mlrose.FourPeaks())
	
	ga_fit_100_1 = []
	ga_fit_100_3 = []
	ga_fit_100_5 = []
	ga_fit_100_7 = []
	ga_fit_100_9 = []
	ga_fit_300_1 = []
	ga_fit_300_3 = []
	ga_fit_300_5 = []
	ga_fit_300_7 = []
	ga_fit_300_9 = []	
	sa_fit = []
	mimic_fit = []
	rhc_fit = []
	ga_time = []
	sa_time = []
	mimic_time = []
	rhc_time = []

	for j in pop:
		for k in m:
			for i in iters:
				print(str(j) + "-", str(k) + "," + str(i))
				time_start_ga = time.time()
				best_state, best_fitness_ga = mlrose.genetic_alg(problem_fit, max_iters=i, pop_size=j, mutation_prob=k, random_state = 2)	
				time_elapsed_ga = (time.time() - time_start_ga)
				
				if (j==100 and k==0.1):
					ga_fit_100_1.append(best_fitness_ga)
				elif (j==100 and k==0.3):
					ga_fit_100_3.append(best_fitness_ga)
				elif (j==100 and k==0.5):
					ga_fit_100_5.append(best_fitness_ga)
				elif (j==100 and k==0.7):
					ga_fit_100_7.append(best_fitness_ga)
				elif (j==100 and k==0.9):
					ga_fit_100_9.append(best_fitness_ga)
				elif (j==300 and k==0.1):
					ga_fit_300_1.append(best_fitness_ga)
				elif (j==300 and k==0.3):
					ga_fit_300_3.append(best_fitness_ga)
				elif (j==300 and k==0.5):
					ga_fit_300_5.append(best_fitness_ga)	
				elif (j==300 and k==0.7):
					ga_fit_300_7.append(best_fitness_ga)
				elif (j==300 and k==0.9):
					ga_fit_300_9.append(best_fitness_ga)														

				# ga_time.append(time_elapsed_ga)
				# ga_fit.append(best_fitness_ga)
		
	x = iters
	plt.plot(x, ga_fit_100_1, label = "Pop100-MProb0.1")
	plt.plot(x, ga_fit_100_3, label = "Pop100-MProb0.3")
	plt.plot(x, ga_fit_100_5, label = "Pop100-MProb0.5")
	plt.plot(x, ga_fit_100_7, label = "Pop100-MProb0.7")
	plt.plot(x, ga_fit_100_9, label = "Pop100-MProb0.9")
	plt.plot(x, ga_fit_300_1, label = "Pop300-MProb0.1")
	plt.plot(x, ga_fit_300_3, label = "Pop300-MProb0.3")
	plt.plot(x, ga_fit_300_5, label = "Pop300-MProb0.5")
	plt.plot(x, ga_fit_300_7, label = "Pop300-MProb0.7")
	plt.plot(x, ga_fit_300_9, label = "Pop300-MProb0.9")

	plt.xlabel('Iters')
	# Set the y axis label of the current axis.
	plt.ylabel('Fitness function value')
	# Set a title of the current axes.
	plt.title('Hyperparam Tuning GA, FourPeaks Problem')
	plt.legend()
	plt.savefig('FourPeaks_GAHyper')
	plt.close()

	# x = iters
	# plt.plot(x, ga_time, label = "GA")
	# plt.plot(x, sa_time, label = "SA")
	# plt.plot(x, mimic_time, label = "MIMIC")
	# plt.plot(x, rhc_time, label = "RHC")
	# plt.xlabel('Iters')
	# # Set the y axis label of the current axis.
	# plt.ylabel('Computation Time (s)')
	# # Set a title of the current axes.
	# plt.title('Computation Time Evalutation for OneMax')		
	# plt.legend()
	# plt.savefig('OneMax_ComputeTime')
	# plt.close()

def four_peaks_mimic():
	'''
	https://readthedocs.org/projects/mlrose/downloads/pdf/stable/
	https://stackoverflow.com/questions/11886862/calculating-computational-time-and-memory-for-a-code-in-python
	'''
	print('Four Peaks')
	iters=[1,200,400,600,800,1000]
	pop = [100,200,300,400,500,600,700,800,900,1000]
	keep = [0.1, 0.3, 0.5, 0.7, 0.9]
	problem_fit = mlrose.DiscreteOpt(length = 100,fitness_fn = mlrose.FourPeaks())
	
	# ga_fit_100_1 = []
	# ga_fit_100_3 = []
	# ga_fit_100_5 = []
	# ga_fit_100_7 = []
	# ga_fit_100_9 = []
	ga_fit_100_1 = []
	ga_fit_100_3 = []
	ga_fit_100_5 = []
	ga_fit_100_7 = []
	ga_fit_100_9 = []
	ga_fit_300_1 = []
	ga_fit_300_3 = []
	ga_fit_300_5 = []
	ga_fit_300_7 = []
	ga_fit_300_9 = []
	ga_fit_k=[]	
	ga_fit_pop = []
	sa_fit = []
	mimic_fit = []
	rhc_fit = []
	ga_time = []
	sa_time = []
	mimic_time = []
	rhc_time = []

	for k in keep:
		# for i in iters:
		print(str(k))
		time_start_ga = time.time()
		best_state, best_fitness_ga = mlrose.mimic(problem_fit, max_iters=1000, keep_pct=k, random_state = 2)	
		time_elapsed_ga = (time.time() - time_start_ga)
		ga_fit_k.append(best_fitness_ga)
		# if (k==0.1):
		# 	ga_fit_100_1.append(best_fitness_ga)
		# elif (k==0.3):
		# 	ga_fit_100_3.append(best_fitness_ga)
		# elif (k==0.5):
		# 	ga_fit_100_5.append(best_fitness_ga)
		# elif (k==0.7):
		# 	ga_fit_100_7.append(best_fitness_ga)
		# elif (k==0.9):
		# 	ga_fit_100_9.append(best_fitness_ga)													

			# ga_time.append(time_elapsed_ga)
			# ga_fit.append(best_fitness_ga)

	x = keep
	plt.plot(x, ga_fit_k, label = "MIMIC")
	# plt.plot(x, ga_fit_100_3, label = "Keep0.3")
	# plt.plot(x, ga_fit_100_5, label = "Keep0.5")
	# plt.plot(x, ga_fit_100_7, label = "Keep0.7")
	# plt.plot(x, ga_fit_100_9, label = "Keep0.9")

	plt.xlabel('Keep PCT')
	# Set the y axis label of the current axis.
	plt.ylabel('Fitness function value')
	# Set a title of the current axes.
	plt.title('Fitness Function Evaluation for FourPeaks vs Keep PCT')
	plt.legend()
	plt.savefig('TSP_MIMICHyper_KeepPCT')
	plt.close()

	for p in pop:
		# for i in iters:
		print(str(p))
		time_start_ga = time.time()
		best_state, best_fitness_ga = mlrose.mimic(problem_fit, max_iters=500, pop_size=p, random_state = 2)	
		time_elapsed_ga = (time.time() - time_start_ga)
		ga_fit_pop.append(best_fitness_ga)

	x = pop_size
	plt.plot(x, ga_fit_pop, label = "MIMIC")
	# plt.plot(x, ga_fit_100_3, label = "Keep0.3")
	# plt.plot(x, ga_fit_100_5, label = "Keep0.5")
	# plt.plot(x, ga_fit_100_7, label = "Keep0.7")
	# plt.plot(x, ga_fit_100_9, label = "Keep0.9")

	plt.xlabel('Population Size')
	# Set the y axis label of the current axis.
	plt.ylabel('Fitness function value')
	# Set a title of the current axes.
	plt.title('Fitness Function Evaluation for FourPeaks vs Population Size')
	plt.legend()
	plt.savefig('FourPeaks_MIMICHyper_PopSize')
	plt.close()		

	# x = iters
	# plt.plot(x, ga_time, label = "GA")
	# plt.plot(x, sa_time, label = "SA")
	# plt.plot(x, mimic_time, label = "MIMIC")
	# plt.plot(x, rhc_time, label = "RHC")
	# plt.xlabel('Iters')
	# # Set the y axis label of the current axis.
	# plt.ylabel('Computation Time (s)')
	# # Set a title of the current axes.
	# plt.title('Computation Time Evalutation for OneMax')		
	# plt.legend()
	# plt.savefig('OneMax_ComputeTime')
	# plt.close()


def four_peaks_sa():
	'''
	https://readthedocs.org/projects/mlrose/downloads/pdf/stable/
	https://stackoverflow.com/questions/11886862/calculating-computational-time-and-memory-for-a-code-in-python
	'''
	iters=[1,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300]
	schedule=[mlrose.GeomDecay(), mlrose.ArithDecay(), mlrose.ExpDecay()]
	problem_fit = mlrose.DiscreteOpt(length = 100,fitness_fn = mlrose.FourPeaks())
	
	ga_fit_geom = []
	ga_fit_arith = []
	ga_fit_exp = []
	sa_fit = []
	mimic_fit = []
	rhc_fit = []
	ga_time = []
	sa_time = []
	mimic_time = []
	rhc_time = []

	for j in range(0,len(schedule)):
		for i in iters:
			print(str(j) + "," + str(i))
			time_start_ga = time.time()
			best_state, best_fitness_ga = mlrose.simulated_annealing(problem_fit, max_iters=i, schedule=schedule[j], random_state = 2)	
			time_elapsed_ga = (time.time() - time_start_ga)
			
			if (j==0):
				ga_fit_geom.append(best_fitness_ga)
			elif (j==1):
				ga_fit_arith.append(best_fitness_ga)
			elif (j==2):
				ga_fit_exp.append(best_fitness_ga)														

			# ga_time.append(time_elapsed_ga)
			# ga_fit.append(best_fitness_ga)
		
	x = iters
	plt.plot(x, ga_fit_geom, label = "Schedule-GeomDecay")
	plt.plot(x, ga_fit_arith, label = "Schedule-ArithDecay")
	plt.plot(x, ga_fit_exp, label = "Schedule-ExpDecay")

	plt.xlabel('Iters')
	# Set the y axis label of the current axis.
	plt.ylabel('Fitness function value')
	# Set a title of the current axes.
	plt.title('Hyperparam Tuning SA, FourPeaks Problem')
	plt.legend()
	plt.savefig('FourPeaks_SAHyper')
	plt.close()	

def flip_flop_ga():
	'''
	https://readthedocs.org/projects/mlrose/downloads/pdf/stable/
	https://stackoverflow.com/questions/11886862/calculating-computational-time-and-memory-for-a-code-in-python
	'''
	iters=[1,100,200,300,400,500,600,700]
	pop = [100,300]
	m = [0.1, 0.3, 0.5, 0.7, 0.9]
	problem_fit = mlrose.DiscreteOpt(length = 100,fitness_fn = mlrose.FlipFlop())
	
	ga_fit_100_1 = []
	ga_fit_100_3 = []
	ga_fit_100_5 = []
	ga_fit_100_7 = []
	ga_fit_100_9 = []
	ga_fit_300_1 = []
	ga_fit_300_3 = []
	ga_fit_300_5 = []
	ga_fit_300_7 = []
	ga_fit_300_9 = []	
	sa_fit = []
	mimic_fit = []
	rhc_fit = []
	ga_time = []
	sa_time = []
	mimic_time = []
	rhc_time = []

	for j in pop:
		for k in m:
			for i in iters:
				print(str(j) + "-", str(k) + "," + str(i))
				time_start_ga = time.time()
				best_state, best_fitness_ga = mlrose.genetic_alg(problem_fit, max_iters=i, pop_size=j, mutation_prob=k, random_state = 2)	
				time_elapsed_ga = (time.time() - time_start_ga)
				
				if (j==100 and k==0.1):
					ga_fit_100_1.append(best_fitness_ga)
				elif (j==100 and k==0.3):
					ga_fit_100_3.append(best_fitness_ga)
				elif (j==100 and k==0.5):
					ga_fit_100_5.append(best_fitness_ga)
				elif (j==100 and k==0.7):
					ga_fit_100_7.append(best_fitness_ga)
				elif (j==100 and k==0.9):
					ga_fit_100_9.append(best_fitness_ga)
				elif (j==300 and k==0.1):
					ga_fit_300_1.append(best_fitness_ga)
				elif (j==300 and k==0.3):
					ga_fit_300_3.append(best_fitness_ga)
				elif (j==300 and k==0.5):
					ga_fit_300_5.append(best_fitness_ga)	
				elif (j==300 and k==0.7):
					ga_fit_300_7.append(best_fitness_ga)
				elif (j==300 and k==0.9):
					ga_fit_300_9.append(best_fitness_ga)														

				# ga_time.append(time_elapsed_ga)
				# ga_fit.append(best_fitness_ga)
		
	x = iters
	plt.plot(x, ga_fit_100_1, label = "Pop100-MProb0.1")
	plt.plot(x, ga_fit_100_3, label = "Pop100-MProb0.3")
	plt.plot(x, ga_fit_100_5, label = "Pop100-MProb0.5")
	plt.plot(x, ga_fit_100_7, label = "Pop100-MProb0.7")
	plt.plot(x, ga_fit_100_9, label = "Pop100-MProb0.9")
	plt.plot(x, ga_fit_300_1, label = "Pop300-MProb0.1")
	plt.plot(x, ga_fit_300_3, label = "Pop300-MProb0.3")
	plt.plot(x, ga_fit_300_5, label = "Pop300-MProb0.5")
	plt.plot(x, ga_fit_300_7, label = "Pop300-MProb0.7")
	plt.plot(x, ga_fit_300_9, label = "Pop300-MProb0.9")

	plt.xlabel('Iters')
	# Set the y axis label of the current axis.
	plt.ylabel('Fitness function value')
	# Set a title of the current axes.
	plt.title('Hyperparam Tuning GA, FlipFlop Problem')
	plt.legend()
	plt.savefig('FlipFlop_GAHyper')
	plt.close()

	# x = iters
	# plt.plot(x, ga_time, label = "GA")
	# plt.plot(x, sa_time, label = "SA")
	# plt.plot(x, mimic_time, label = "MIMIC")
	# plt.plot(x, rhc_time, label = "RHC")
	# plt.xlabel('Iters')
	# # Set the y axis label of the current axis.
	# plt.ylabel('Computation Time (s)')
	# # Set a title of the current axes.
	# plt.title('Computation Time Evalutation for OneMax')		
	# plt.legend()
	# plt.savefig('OneMax_ComputeTime')
	# plt.close()

def flip_flop_mimic():
	'''
	https://readthedocs.org/projects/mlrose/downloads/pdf/stable/
	https://stackoverflow.com/questions/11886862/calculating-computational-time-and-memory-for-a-code-in-python
	'''
	iters=[1,100,200,300,400,500,600,700]
	pop = [200]
	keep = [0.1, 0.3, 0.5, 0.7, 0.9]
	problem_fit = mlrose.DiscreteOpt(length = 100,fitness_fn = mlrose.FlipFlop())
	
	# ga_fit_100_1 = []
	# ga_fit_100_3 = []
	# ga_fit_100_5 = []
	# ga_fit_100_7 = []
	# ga_fit_100_9 = []
	ga_fit_300_1 = []
	ga_fit_300_3 = []
	ga_fit_300_5 = []
	ga_fit_300_7 = []
	ga_fit_300_9 = []	
	sa_fit = []
	mimic_fit = []
	rhc_fit = []
	ga_time = []
	sa_time = []
	mimic_time = []
	rhc_time = []

	for j in pop:
		for k in keep:
			for i in iters:
				print(str(j) + "-", str(k) + "," + str(i))
				time_start_ga = time.time()
				best_state, best_fitness_ga = mlrose.mimic(problem_fit, max_iters=i, keep_pct=k, random_state = 2)	
				time_elapsed_ga = (time.time() - time_start_ga)
				
				# if (j==100 and k==0.1):
				# 	ga_fit_100_1.append(best_fitness_ga)
				# elif (j==100 and k==0.3):
				# 	ga_fit_100_3.append(best_fitness_ga)
				# elif (j==100 and k==0.5):
				# 	ga_fit_100_5.append(best_fitness_ga)
				# elif (j==100 and k==0.7):
				# 	ga_fit_100_7.append(best_fitness_ga)
				# elif (j==100 and k==0.9):
				# 	ga_fit_100_9.append(best_fitness_ga)
				if(k==0.1):
					ga_fit_300_1.append(best_fitness_ga)
				elif (k==0.3):
					ga_fit_300_3.append(best_fitness_ga)
				elif (k==0.5):
					ga_fit_300_5.append(best_fitness_ga)	
				elif (k==0.7):
					ga_fit_300_7.append(best_fitness_ga)
				elif (k==0.9):
					ga_fit_300_9.append(best_fitness_ga)														

				# ga_time.append(time_elapsed_ga)
				# ga_fit.append(best_fitness_ga)
		
	x = iters
	# plt.plot(x, ga_fit_100_1, label = "Pop100-Keep0.1")
	# plt.plot(x, ga_fit_100_3, label = "Pop100-Keep0.3")
	# plt.plot(x, ga_fit_100_5, label = "Pop100-Keep0.5")
	# plt.plot(x, ga_fit_100_7, label = "Pop100-Keep0.7")
	# plt.plot(x, ga_fit_100_9, label = "Pop100-Keep0.9")
	plt.plot(x, ga_fit_300_1, label = "Keep0.1")
	plt.plot(x, ga_fit_300_3, label = "Keep0.3")
	plt.plot(x, ga_fit_300_5, label = "Keep0.5")
	plt.plot(x, ga_fit_300_7, label = "Keep0.7")
	plt.plot(x, ga_fit_300_9, label = "Keep0.9")

	plt.xlabel('Iters')
	# Set the y axis label of the current axis.
	plt.ylabel('Fitness function value')
	# Set a title of the current axes.
	plt.title('Hyperparam Tuning MIMIC, FlipFlop Problem')
	plt.legend()
	plt.savefig('FlipFlop_MIMICHyper')
	plt.close()

	# x = iters
	# plt.plot(x, ga_time, label = "GA")
	# plt.plot(x, sa_time, label = "SA")
	# plt.plot(x, mimic_time, label = "MIMIC")
	# plt.plot(x, rhc_time, label = "RHC")
	# plt.xlabel('Iters')
	# # Set the y axis label of the current axis.
	# plt.ylabel('Computation Time (s)')
	# # Set a title of the current axes.
	# plt.title('Computation Time Evalutation for OneMax')		
	# plt.legend()
	# plt.savefig('OneMax_ComputeTime')
	# plt.close()


def flip_flop_sa():
	'''
	https://readthedocs.org/projects/mlrose/downloads/pdf/stable/
	https://stackoverflow.com/questions/11886862/calculating-computational-time-and-memory-for-a-code-in-python
	'''
	iters=[1,100,200,300,400,500,600,700]
	schedule=[mlrose.GeomDecay(), mlrose.ArithDecay(), mlrose.ExpDecay()]
	problem_fit = mlrose.DiscreteOpt(length = 100,fitness_fn = mlrose.FlipFlop())
	
	ga_fit_geom = []
	ga_fit_arith = []
	ga_fit_exp = []
	sa_fit = []
	mimic_fit = []
	rhc_fit = []
	ga_time = []
	sa_time = []
	mimic_time = []
	rhc_time = []

	for j in range(0,len(schedule)):
		for i in iters:
			print(str(j) + "," + str(i))
			time_start_ga = time.time()
			best_state, best_fitness_ga = mlrose.simulated_annealing(problem_fit, max_iters=i, schedule=schedule[j], random_state = 2)	
			time_elapsed_ga = (time.time() - time_start_ga)
			
			if (j==0):
				ga_fit_geom.append(best_fitness_ga)
			elif (j==1):
				ga_fit_arith.append(best_fitness_ga)
			elif (j==2):
				ga_fit_exp.append(best_fitness_ga)														

			# ga_time.append(time_elapsed_ga)
			# ga_fit.append(best_fitness_ga)
		
	x = iters
	plt.plot(x, ga_fit_geom, label = "Schedule-GeomDecay")
	plt.plot(x, ga_fit_arith, label = "Schedule-ArithDecay")
	plt.plot(x, ga_fit_exp, label = "Schedule-ExpDecay")

	plt.xlabel('Iters')
	# Set the y axis label of the current axis.
	plt.ylabel('Fitness function value')
	# Set a title of the current axes.
	plt.title('Hyperparam Tuning SA, FlipFlop Problem')
	plt.legend()
	plt.savefig('FlipFlop_SAHyper')
	plt.close()	

def TSP_sa(coords_list):
	'''
	https://readthedocs.org/projects/mlrose/downloads/pdf/stable/
	https://stackoverflow.com/questions/11886862/calculating-computational-time-and-memory-for-a-code-in-python
	'''
	iters=[1,100,200,300,400,500]
	schedule=[mlrose.GeomDecay(), mlrose.ArithDecay(), mlrose.ExpDecay()]
	# coords_list = [tuple(random.sample(range(1,50), 2)) for x in range(100)]
	fitness_coords = mlrose.TravellingSales(coords=coords_list)
	# print(len(coords_list))
	# print(len(set(coords_list)))

	# coords_list=[(1, 1), (4, 2), (5, 2), (6, 4), (4, 4), (3, 6), (1, 5), (2, 3)]
	# Initialize fitness function object using dist_list
	# fitness_coords = mlrose.TravellingSales(coords = coords_list)
	# problem_fit = mlrose.DiscreteOpt(length = 8, fitness_fn = fitness_coords, maximize=True)
	problem_fit = mlrose.TSPOpt(length = 100, fitness_fn = fitness_coords, maximize=True)	
	
	ga_fit_geom = []
	ga_fit_arith = []
	ga_fit_exp = []
	sa_fit = []
	mimic_fit = []
	rhc_fit = []
	ga_time = []
	sa_time = []
	mimic_time = []
	rhc_time = []

	for j in range(0,len(schedule)):
		for i in iters:
			print(str(j) + "," + str(i))
			time_start_ga = time.time()
			best_state, best_fitness_ga = mlrose.simulated_annealing(problem_fit, max_iters=i, schedule=schedule[j], random_state = 2)	
			time_elapsed_ga = (time.time() - time_start_ga)
			
			if (j==0):
				ga_fit_geom.append(best_fitness_ga)
			elif (j==1):
				ga_fit_arith.append(best_fitness_ga)
			elif (j==2):
				ga_fit_exp.append(best_fitness_ga)														

			# ga_time.append(time_elapsed_ga)
			# ga_fit.append(best_fitness_ga)
		
	x = iters
	plt.plot(x, ga_fit_geom, label = "Schedule-GeomDecay")
	plt.plot(x, ga_fit_exp, label = "Schedule-ArithDecay")
	plt.plot(x, ga_fit_exp, label = "Schedule-ExpDecay")

	plt.xlabel('Iters')
	# Set the y axis label of the current axis.
	plt.ylabel('Fitness function value')
	# Set a title of the current axes.
	plt.title('Hyperparam Tuning SA, TSP Problem')
	plt.legend()
	plt.savefig('TSP_SAHyper')
	plt.close()	

def TSP_ga(coords_list):
	'''
	https://readthedocs.org/projects/mlrose/downloads/pdf/stable/
	https://stackoverflow.com/questions/11886862/calculating-computational-time-and-memory-for-a-code-in-python
	'''
	iters=[1,100,200,300,400,500]
	pop = [100,300]
	m = [0.1, 0.3, 0.5, 0.7, 0.9]
	# coords_list = [tuple(random.sample(range(1,50), 2)) for x in range(100)]
	fitness_coords = mlrose.TravellingSales(coords=coords_list)
	# print(len(coords_list))
	# print(len(set(coords_list)))

	# coords_list=[(1, 1), (4, 2), (5, 2), (6, 4), (4, 4), (3, 6), (1, 5), (2, 3)]
	# Initialize fitness function object using dist_list
	# fitness_coords = mlrose.TravellingSales(coords = coords_list)
	# problem_fit = mlrose.DiscreteOpt(length = 8, fitness_fn = fitness_coords, maximize=True)
	problem_fit = mlrose.TSPOpt(length = 100, fitness_fn = fitness_coords, maximize=True)		
	ga_fit_100_1 = []
	ga_fit_100_3 = []
	ga_fit_100_5 = []
	ga_fit_100_7 = []
	ga_fit_100_9 = []
	ga_fit_300_1 = []
	ga_fit_300_3 = []
	ga_fit_300_5 = []
	ga_fit_300_7 = []
	ga_fit_300_9 = []	
	sa_fit = []
	mimic_fit = []
	rhc_fit = []
	ga_time = []
	sa_time = []
	mimic_time = []
	rhc_time = []

	for j in pop:
		for k in m:
			for i in iters:
				print(str(j) + "-", str(k) + "," + str(i))
				time_start_ga = time.time()
				best_state, best_fitness_ga = mlrose.genetic_alg(problem_fit, max_iters=i, pop_size=j, mutation_prob=k, random_state = 2)	
				time_elapsed_ga = (time.time() - time_start_ga)
				
				print(best_fitness_ga)
				if (j==100 and k==0.1):
					ga_fit_100_1.append(best_fitness_ga)
				elif (j==100 and k==0.3):
					ga_fit_100_3.append(best_fitness_ga)
				elif (j==100 and k==0.5):
					ga_fit_100_5.append(best_fitness_ga)
				elif (j==100 and k==0.7):
					ga_fit_100_7.append(best_fitness_ga)
				elif (j==100 and k==0.9):
					ga_fit_100_9.append(best_fitness_ga)
				elif (j==300 and k==0.1):
					ga_fit_300_1.append(best_fitness_ga)
				elif (j==300 and k==0.3):
					ga_fit_300_3.append(best_fitness_ga)
				elif (j==300 and k==0.5):
					ga_fit_300_5.append(best_fitness_ga)	
				elif (j==300 and k==0.7):
					ga_fit_300_7.append(best_fitness_ga)
				elif (j==300 and k==0.9):
					ga_fit_300_9.append(best_fitness_ga)														

				# ga_time.append(time_elapsed_ga)
				# ga_fit.append(best_fitness_ga)
		
	x = iters
	plt.plot(x, ga_fit_100_1, label = "Pop100-MProb0.1")
	plt.plot(x, ga_fit_100_3, label = "Pop100-MProb0.3")
	plt.plot(x, ga_fit_100_5, label = "Pop100-MProb0.5")
	plt.plot(x, ga_fit_100_7, label = "Pop100-MProb0.7")
	plt.plot(x, ga_fit_100_9, label = "Pop100-MProb0.9")
	plt.plot(x, ga_fit_300_1, label = "Pop300-MProb0.1")
	plt.plot(x, ga_fit_300_3, label = "Pop300-MProb0.3")
	plt.plot(x, ga_fit_300_5, label = "Pop300-MProb0.5")
	plt.plot(x, ga_fit_300_7, label = "Pop300-MProb0.7")
	plt.plot(x, ga_fit_300_9, label = "Pop300-MProb0.9")

	plt.xlabel('Iters')
	# Set the y axis label of the current axis.
	plt.ylabel('Fitness function value')
	# Set a title of the current axes.
	plt.title('Hyperparam Tuning GA, TSP Problem')
	plt.legend()
	plt.savefig('TSP_GAHyper')
	plt.close()

	# x = iters
	# plt.plot(x, ga_time, label = "GA")
	# plt.plot(x, sa_time, label = "SA")
	# plt.plot(x, mimic_time, label = "MIMIC")
	# plt.plot(x, rhc_time, label = "RHC")
	# plt.xlabel('Iters')
	# # Set the y axis label of the current axis.
	# plt.ylabel('Computation Time (s)')
	# # Set a title of the current axes.
	# plt.title('Computation Time Evalutation for OneMax')		
	# plt.legend()
	# plt.savefig('OneMax_ComputeTime')
	# plt.close()	

def TSP_mimic(coords_list):
	'''
	https://readthedocs.org/projects/mlrose/downloads/pdf/stable/
	https://stackoverflow.com/questions/11886862/calculating-computational-time-and-memory-for-a-code-in-python
	'''
	print('TSP')
	iters=[1,100,200,300,400,500]
	keep = [0.1, 0.3, 0.5, 0.7, 0.9]
	pop = [100,200,300,400,500,600,700,800]
	# coords_list = [tuple(random.sample(range(1,50), 2)) for x in range(100)]
	fitness_coords = mlrose.TravellingSales(coords=coords_list)
	# print(len(coords_list))
	# print(len(set(coords_list)))

	# coords_list=[(1, 1), (4, 2), (5, 2), (6, 4), (4, 4), (3, 6), (1, 5), (2, 3)]
	# Initialize fitness function object using dist_list
	# fitness_coords = mlrose.TravellingSales(coords = coords_list)
	# problem_fit = mlrose.DiscreteOpt(length = 8, fitness_fn = fitness_coords, maximize=True)
	problem_fit = mlrose.TSPOpt(length = 100, fitness_fn = fitness_coords, maximize=True)		
	
	ga_fit_100_1 = []
	ga_fit_100_3 = []
	ga_fit_100_5 = []
	ga_fit_100_7 = []
	ga_fit_100_9 = []
	ga_fit_300_1 = []
	ga_fit_300_3 = []
	ga_fit_300_5 = []
	ga_fit_300_7 = []
	ga_fit_300_9 = []
	ga_fit_k=[]	
	ga_fit_pop = []
	sa_fit = []
	mimic_fit = []
	rhc_fit = []
	ga_time = []
	sa_time = []
	mimic_time = []
	rhc_time = []

	for k in keep:
		# for i in iters:
		print(str(k))
		time_start_ga = time.time()
		best_state, best_fitness_ga = mlrose.mimic(problem_fit, max_iters=500, keep_pct=k, random_state = 2)	
		time_elapsed_ga = (time.time() - time_start_ga)
		ga_fit_k.append(best_fitness_ga)
		print(ga_fit_k)
		# if (k==0.1):
		# 	ga_fit_100_1.append(best_fitness_ga)
		# elif (k==0.3):
		# 	ga_fit_100_3.append(best_fitness_ga)
		# elif (k==0.5):
		# 	ga_fit_100_5.append(best_fitness_ga)
		# elif (k==0.7):
		# 	ga_fit_100_7.append(best_fitness_ga)
		# elif (k==0.9):
		# 	ga_fit_100_9.append(best_fitness_ga)													

			# ga_time.append(time_elapsed_ga)
			# ga_fit.append(best_fitness_ga)

	x = keep
	plt.plot(x, ga_fit_k, label = "MIMIC")
	# plt.plot(x, ga_fit_100_3, label = "Keep0.3")
	# plt.plot(x, ga_fit_100_5, label = "Keep0.5")
	# plt.plot(x, ga_fit_100_7, label = "Keep0.7")
	# plt.plot(x, ga_fit_100_9, label = "Keep0.9")

	plt.xlabel('Keep PCT')
	# Set the y axis label of the current axis.
	plt.ylabel('Fitness function value')
	# Set a title of the current axes.
	plt.title('Fitness Function Evaluation for TSP vs Keep PCT')
	plt.legend()
	plt.savefig('TSP_MIMICHyper_KeepPCT')
	plt.close()


	for p in pop:
		# for i in iters:
		print(str(p))
		time_start_ga = time.time()
		best_state, best_fitness_ga = mlrose.mimic(problem_fit, max_iters=500, pop_size=p, random_state = 2)	
		time_elapsed_ga = (time.time() - time_start_ga)
		ga_fit_pop.append(best_fitness_ga)

	x = pop_size
	plt.plot(x, ga_fit_pop, label = "MIMIC")
	# plt.plot(x, ga_fit_100_3, label = "Keep0.3")
	# plt.plot(x, ga_fit_100_5, label = "Keep0.5")
	# plt.plot(x, ga_fit_100_7, label = "Keep0.7")
	# plt.plot(x, ga_fit_100_9, label = "Keep0.9")

	plt.xlabel('Population Size')
	# Set the y axis label of the current axis.
	plt.ylabel('Fitness function value')
	# Set a title of the current axes.
	plt.title('Fitness Function Evaluation for TSP vs Population Size')
	plt.legend()
	plt.savefig('TSP_MIMICHyper_PopSize')
	plt.close()	

	# x = iters
	# plt.plot(x, ga_time, label = "GA")
	# plt.plot(x, sa_time, label = "SA")
	# plt.plot(x, mimic_time, label = "MIMIC")
	# plt.plot(x, rhc_time, label = "RHC")
	# plt.xlabel('Iters')
	# # Set the y axis label of the current axis.
	# plt.ylabel('Computation Time (s)')
	# # Set a title of the current axes.
	# plt.title('Computation Time Evalutation for OneMax')		
	# plt.legend()
	# plt.savefig('OneMax_ComputeTime')
	# plt.close()


def TSP(coords_list):
	'''
	https://readthedocs.org/projects/mlrose/downloads/pdf/stable/
	https://stackoverflow.com/questions/11886862/calculating-computational-time-and-memory-for-a-code-in-python
	'''
	iters=[1,100,200,300,400,500]
	random.seed(2)
	# coords_list = [tuple(random.sample(range(1,50), 2)) for x in range(100)]
	fitness_coords = mlrose.TravellingSales(coords=coords_list)
	# fitness_coords = mlrose.CustomFitness(coords_list,"tsp")
	# print(len(coords_list))
	# print(len(set(coords_list)))

	# coords_list=[(1, 1), (4, 2), (5, 2), (6, 4), (4, 4), (3, 6), (1, 5), (2, 3)]
	# Initialize fitness function object using dist_list
	# fitness_coords = mlrose.TravellingSales(coords = coords_list)
	# problem_fit = mlrose.DiscreteOpt(length = 8, fitness_fn = fitness_coords, maximize=True)
	problem_fit = mlrose.TSPOpt(length = 100, fitness_fn = fitness_coords, maximize=True)	
	
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
		best_state, best_fitness_ga = mlrose.genetic_alg(problem_fit, max_iters=i,random_state = 2, pop_size=300, mutation_prob=0.1)	
		time_elapsed_ga = (time.time() - time_start_ga)
		ga_time.append(time_elapsed_ga)
		ga_fit.append(best_fitness_ga)
		print(ga_fit)
		
		time_start_sa = time.time()
		best_state, best_fitness_sa = mlrose.simulated_annealing(problem_fit, max_iters=i, schedule=mlrose.ExpDecay(), random_state = 2)	
		time_elapsed_sa = (time.time() - time_start_sa)
		sa_time.append(time_elapsed_sa)		
		sa_fit.append(best_fitness_sa)
		print(sa_fit)
		
		time_start_mimic = time.time()
		best_state, best_fitness_mimic = mlrose.mimic(problem_fit, max_iters=i,random_state = 2)	
		time_elapsed_mimic = (time.time() - time_start_mimic)
		mimic_time.append(time_elapsed_mimic)			
		mimic_fit.append(best_fitness_mimic)
		print(mimic_fit)
		
		time_start_rhc = time.time()
		best_state, best_fitness_rhc = mlrose.random_hill_climb(problem_fit, max_iters=i,random_state = 2)	
		time_elapsed_rhc = (time.time() - time_start_rhc)		
		rhc_time.append(time_elapsed_rhc)			
		rhc_fit.append(best_fitness_rhc)
		print(rhc_fit)

	x = iters
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

	x = iters
	plt.plot(x, ga_time, label = "GA")
	plt.plot(x, sa_time, label = "SA")
	plt.plot(x, mimic_time, label = "MIMIC")
	plt.plot(x, rhc_time, label = "RHC")
	plt.xlabel('Iters')
	# Set the y axis label of the current axis.
	plt.ylabel('Computation Time (s)')
	# Set a title of the current axes.
	plt.title('Computation Time Evalutation for TSP')		
	plt.legend()
	plt.savefig('TSP_ComputeTime')
	plt.close()	

def main():
	random.seed(1009)
	coords_list = [tuple(random.sample(range(1,50), 2)) for x in range(100)]
	

if __name__ == "__main__":
	main()
	
