This folder contains:

 * README.txt - This file as an overview of folder contents.
 * randomOp.py - Main file that generates all plots for Randomized Optimization Algorithms, evaluated on Optimization Problems. It contains APIs for hyperparameter tuning and fitness performance/computation time performance for the following algorithms:
	(1) Random Hill Climbing (RHC)
	(2) Simulated Annealing (SA)
	(3) Genetic Algorithms (GA)
	(4) MIMIC

applied to the following optimization problems:
	(1) Traveling Salesperson Problem (TSP)
	(2) Four Peaks Problem
	(3) Flip Flop Problem

 * NNweights.py - Reads FlightDelays.pkl file into training/test datasets for model training. Evaluates RHC, SA, and GA algorithms on optimizing NN weights, compared against back propagation technique used in Assignment 1. It also contains APIs for hyperparameter tuning of individual algorithms

 * FlightDelays.pkl - Pickle file for Flight Delays data

├── README.txt
├── randomOp.py
├── NNweights.py
└── FlightDelays.pkl

SUMMARY OF HOW TO RUN THE CODE
=======================

To run the code in this folder, first ensure that you are using Python3.6 or greater.

===PART 1: RANDOMIZED OPTIMIZATION ALGORITHMS AND PROBLEMS ===

You can generate hyperparameter tuning plots and curves for fitness evaluation/computation time evaluation for optimization algorithms listed above, using the following command:

python randomOp.py

Be sure to modify the main function based on which model you are interested in running. Some tips below:

* To generate hyperparameter tuning plots, invoke 
<PROBLEM>_<ALGORITHM>() API, e.g. flip_flop_mimic()

* To generate fitness evaluation plots and computation time evaluation plots, invoke
<PROBLEM>() API, e.g. four_peaks()

===PART 2: OPTIMAL WEIGHTS FOR NEURAL NETWORK ===

You can also generate hyperparameter tuning plots and f1/accuracy score curves for evaluating the optimization algorithms listed above, using the following command:

python NNweights.py

Be sure to modify the main function based on which model you are interested in running. Some tips below:

* To perform hyperparameter tuning for models using Flight Delay Dataset, invoke <ALGORITHM>_opt(X_train_eData, Y_train_eData, X_test_eData, Y_test_eData),
e.g. ga_opt()

* To generate f1/accuracy score curves and computation time evaluation, invoke
compare_algos(X_train_eData, Y_train_eData, X_test_eData, Y_test_eData)


REFERENCES:
Documentation for mlrose library: https://readthedocs.org/projects/mlrose/downloads/pdf/stable/

Calculating compute-time in Python: https://stackoverflow.com/questions/11886862/calculating-computational-time-and-memory-for-a-code-in-python

Unsupervised Learning Randomized Optimization Lecture: https://gatech.instructure.com/courses/159302/pages/ul-1-randomized-optimization?module_item_id=1427428
