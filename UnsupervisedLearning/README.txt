The code can be found at:
https://github.gatech.edu/arao338/CS7641_UnsupervisedLearningDimensionalityReduction.git

This folder contains:

 * README.txt - This file as an overview of folder contents.
 * ul.py - Main file that generates all plots for Unsupervised Learning & Dimensionality Reduction Experiments 1-5 on two datasets referenced in the paper:
	(1) Experiment 1: Perform clustering
	(2) Experiment 2: Perform dimensionality reduction
	(3) Experiment 3: Perform dimensionality reduction + clustering
	(4) Experiment 4: Perform dimensionality reduction + neural network training
	(5) Experiment 5: Perform clustering (as features) + neural network training

APIs for implementing the following clustering algorithms are provided:
	(1) K-Means
	(2) Expectation Maximization (Gaussian Mixture)

APIs for implementing the following dimensionality reduction techniques are provided:
	(1) Principal Component Analysis (PCA)
	(2) Independent Component Analysis (ICA)
	(3) Randomized Projections (RP)
	(4) Backward Elimination using KNN estimator (BE)

 * EEGEyeData.pkl - Pickle file for EEG Eye data
 * FlightDelays.pkl - Pickle file for Flight Delays data


├── README.txt
├── ul.py
├── EEGEyeData.pkl
└── FlightDelays.pkl

SUMMARY OF HOW TO RUN THE CODE
=======================

To run the code in this folder, first ensure that you are using Python3.6 or greater.

You can perform Experiments 1-5 for the two datasets using the following command:

python ul.py

Be sure to modify the main function based on which model you are interested in running. Some tips below:

===EXPERIMENT 1: CLUSTERING ===

* To generate clustering plots, invoke 

do_<ALGORITHM>(), e.g. do_kmeans()


===EXPERIMENT 2: DIMENSIONALITY REDUCTION ===

* To perform dimensionality reduction and generate hyperparameter tuning plots by # components (explained variance, kurtosis, reconstruction error, etc), invoke

do_<ALGORITHM>(), e.g. do_PCA()


===EXPERIMENT 3: DIMENSIONALITY REDUCTION + CLUSTERING ===

* To perform dimensionality reduction and pass the transformed data to clustering algorithms, invoke command from Experiment 2, then invoke

dimred_<ALGORITHM>(), e.g. dimred_kmeans()


===EXPERIMENT 4: DIMENSIONALITY REDUCTION + NEURAL NETWORK ===

* To generate plots for neural network performance after dimensionality reduction step, invoke command from Experiment 2, then invoke

gen_learning_curve()
plot_learning_curve()

===EXPERIMENT 5: CLUSTERING + NEURAL NETWORK ===

* To generate plots for neural network performance after using clusters as features, invoke

gen_learning_curve_clusters()
plot_learning_curve()


REFERENCES:

* K-Means Clustering: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

* Gaussian Mixture Clustering: https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html

* PCA Dimensionality Reduction: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

* ICA Dimensionality Reduction: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html

* Randomized Projections Dimensionality Reduction: https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.GaussianRandomProjection.html

* Backward Elimination Dimensionality Reduction: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html

* Silhouette Score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html

* Elbow Method: https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/

* Kurtosis: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kurtosis.html

* Plot learning curve code: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

* Calculating compute-time in Python: https://stackoverflow.com/questions/11886862/calculating-computational-time-and-memory-for-a-code-in-python


