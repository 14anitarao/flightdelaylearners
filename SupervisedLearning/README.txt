This folder contains:

 * README.txt - This file as an overview of folder contents.
 * models.py - Main file that generates all plots for supervised learning model evaluation. It contains APIs for hyperparameter tuning and learning curve generation for the following models:
	(1) Decision Tree Classifier
	(2) Multi-layer Perceptron Classifier
	(3) Gradient Boosting Classifier
	(4) Support Vector Classifier
	(5) KNN Classifier

 * pickeData.py - Converts datasets in CSV format to pickle format for faster loading from run to run. It contains code to convert the datasets used in this experiment:
	(1) FlightDelays
	(2) EEGEyeData
 * FlightDelays.pkl - Pickle file for Flight Delays data
 * EEGEyeData.pkl - Pickle file for EEG Eye data

├── README.txt
├── models.py
├── pickeData.py
├── FlightDelays.pkl
└── EEGEyeData.pkl

SUMMARY OF HOW TO RUN THE CODE
=======================

To run the code in this folder, first ensure that you are using Python3.6 or greater.


You can generate hyperparameter tuning plots and learning curves for models listed above using the following command:

python models.py

Be sure to modify the main function based on which model you are interested in running. Some tips below:

e.g. To perform hyperparameter tuning for models using EEG Dataset, invoke <MODEL>_tuning(X_train_pData, X_test_pData, Y_train_pData, Y_test_pData)

e.g. To plot learning curves for models using EEG Dataset, invoke gen_learning_curve(normalized_X_pData, y_pData)



You can generate your own picked data by using the following command:

python pickleData.py

Be sure to modify the main function based on the CSV file you would like to convert to pkl.


REFERENCES:
Decision Tree Classifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

Multi-layer Perceptron Classifier: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

Gradient Boosting Classifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html

Support Vector Classifier: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

KNN Classifier: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier.kneighbors
