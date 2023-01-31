import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *
from sklearn.model_selection import *
import sys
from time import time


# setup the randoms tate
RANDOM_STATE = 545510477

#input: X_train, Y_train
#output: Y_pred
def logistic_regression_pred(X_train, Y_train):
	#train a logistic regression classifier using X_train and Y_train. Use this to predict labels of X_train
	#use default params for the classifier
	clf = LogisticRegression(random_state=RANDOM_STATE).fit(X_train, Y_train)
	Y_pred = clf.predict(X_train)

	return Y_pred

#input: X_train, Y_train
#output: Y_pred
def decisionTree_pred_insample(X_train, Y_train):
	#train a logistic regression classifier using X_train and Y_train. Use this to predict labels of X_train
	#use max_depth as 5
	clf = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=5).fit(X_train, Y_train)
	Y_pred = clf.predict(X_train)

	'''
	Using GINI as criteria for splitting attributes
	Potential experiment: Try entropy?

	Potential experiment: Pruning by changing the max_depth. 
	The smaller you go, you will start to see overfitting. Plot where the overfitting begins.
	Overfitting: When in-sample error goes down, but out-sample error goes up
	'''

	return Y_pred

#input: X_train, Y_train and X_test
#output: Y_pred
def decisionTree_pred_outsample(X_train, Y_train, X_test):
	#TODO:train a logistic regression classifier using X_train and Y_train. Use this to predict labels of X_test
	#IMPORTANT: use max_depth as 5. Else your test cases might fail.
	clf = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=5).fit(X_train, Y_train)
	Y_pred = clf.predict(X_test)

	'''
	Using GINI as criteria for splitting attributes
	Potential experiment: Try entropy?

	Potential experiment: changing the max_depth Currently pre-pruning setting it to 5. 
	The smaller you go, you will start to see overfitting. Plot where the overfitting begins.
	Overfitting: When in-sample error goes down, but out-sample error goes up
	'''

	return Y_pred

#input: X_train, Y_train and X_test
#output: Y_pred
def decisionTree_tuning(X_train, X_test, Y_train, Y_test):
	#TODO:train a logistic regression classifier using X_train and Y_train. Use this to predict labels of X_test
	#IMPORTANT: use max_depth as 5. Else your test cases might fail.

	''' Code for hypertuning by max_depth. Max_Depth = 5 is best for FLIGHT'''
	''' Code for hypertuning by max_depth. Max_Depth = 8 is best for EEG'''

	# d = list(range(1,51))
	# error_train = []
	# error_test = []

	# for i in d:
	# 	clf = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=i).fit(X_train, Y_train)
	# 	Y_pred_train = clf.predict(X_train)
	# 	Y_pred_test = clf.predict(X_test)
	# 	error_train.append(1-accuracy_score(Y_train, Y_pred_train))
	# 	error_test.append(1-accuracy_score(Y_test, Y_pred_test))
	
	# plt.plot(d, error_train, label = "Training Error")
	# plt.plot(d, error_test, label = "Validation Error")
	# plt.xlabel('Max Depth')
	# # Set the y axis label of the current axis.
	# plt.ylabel('Error')
	# # Set a title of the current axes.
	# plt.title('Decision Tree Hypertuning of Max Depth Param')	
	# plt.legend()
	# plt.savefig('DT_TuningMaxDepth')

	''' Code for hypertuning by split criterion. Best is no difference for FLIGHT'''
	''' Code for hypertuning by split criterion. Best is entropy, by little for EEG'''

	# d = ['gini', 'entropy']
	# error_train = []
	# error_test = []

	# for i in d:
	# 	clf = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=5, criterion=i).fit(X_train, Y_train)
	# 	Y_pred_train = clf.predict(X_train)
	# 	Y_pred_test = clf.predict(X_test)		
	# 	error_train.append(1-accuracy_score(Y_train, Y_pred_train))
	# 	error_test.append(1-accuracy_score(Y_test, Y_pred_test))

	# print(error_train)
	# print(error_test)


	'''
	Using GINI as criteria for splitting attributes
	Potential experiment: Try entropy?

	Potential experiment: changing the max_depth Currently pre-pruning setting it to 5. 
	The smaller you go, you will start to see overfitting. Plot where the overfitting begins.
	Overfitting: When in-sample error goes down, but out-sample error goes up
	'''

#input: X_train, Y_train
#output: Y_pred
def mlp_pred_insample(X_train, Y_train):
	#train a logistic regression classifier using X_train and Y_train. Use this to predict labels of X_train
	#use max_depth as 5
	clf = MLPClassifier(random_state=RANDOM_STATE, max_iter=300).fit(X_train, Y_train)
	Y_pred = clf.predict(X_train)

	'''
	Using 100 hidden layers
	RELU activation
	Adam solver
	max_iter 300
	Potential experiment: Toggle the above params?
	'''

	return Y_pred

#input: X_train, Y_train and X_test
#output: Y_pred
def mlp_pred_outsample(X_train, Y_train, X_test):
	#TODO:train a logistic regression classifier using X_train and Y_train. Use this to predict labels of X_test
	#IMPORTANT: use max_depth as 5. Else your test cases might fail.
	clf = MLPClassifier(random_state=RANDOM_STATE, max_iter=300).fit(X_train, Y_train)
	Y_pred = clf.predict(X_test)

	'''
	Using 100 hidden layers
	RELU activation
	Adam solver
	max_iter 300
	Potential experiment: Toggle the above params?
	'''

	return Y_pred

#input: X_train, Y_train and X_test
#output: Y_pred
def mlp_tuning(X_train, X_test, Y_train, Y_test):
	#TODO:train a logistic regression classifier using X_train and Y_train. Use this to predict labels of X_test
	#IMPORTANT: use max_depth as 5. Else your test cases might fail.

	# ''' Code for hypertuning by hidden layers. hidden layers = 30 is best for FLIGHT'''
	# d = list(range(1,31))
	# error_train = []
	# error_test = []

	# for i in d:
	# 	print(i)
	# 	clf = MLPClassifier(random_state=RANDOM_STATE, hidden_layer_sizes=i, max_iter=100, activation='tanh', solver='sgd').fit(X_train, Y_train)
	# 	Y_pred_train = clf.predict(X_train)
	# 	Y_pred_test = clf.predict(X_test)
	# 	error_train.append(1-accuracy_score(Y_train, Y_pred_train))
	# 	error_test.append(1-accuracy_score(Y_test, Y_pred_test))
	
	# plt.plot(d, error_train, label = "Training Error")
	# plt.plot(d, error_test, label = "Validation Error")
	# plt.xlabel('# Hidden Layers')
	# # Set the y axis label of the current axis.
	# plt.ylabel('Error')
	# # Set a title of the current axes.
	# plt.title('MLP Hypertuning of # Hidden Layers')	
	# plt.legend()
	# plt.savefig('MLP_TuningHiddenLayer')
	

	''' Code for hypertuning by activation function. Best doesn't matter for FLIGHT'''
	''' Code for hypertuning by activation function. Best is relu for EEG'''

	# d = ['tanh','relu']
	# error_train = []
	# error_test = []

	# for i in d:
	# 	print(i)
	# 	clf = MLPClassifier(random_state=RANDOM_STATE, hidden_layer_sizes=30, activation=i, solver='sgd').fit(X_train, Y_train)
	# 	Y_pred_train = clf.predict(X_train)
	# 	Y_pred_test = clf.predict(X_test)
	# 	error_train.append(1-accuracy_score(Y_train, Y_pred_train))
	# 	error_test.append(1-accuracy_score(Y_test, Y_pred_test))
	
	# print(error_train)
	# print(error_test)

	# ''' Code for hypertuning by solver. Best is adam'''
	# d = ['sgd','adam']
	# error_train = []
	# error_test = []

	# for i in d:
	# 	print(i)
	# 	clf = MLPClassifier(random_state=RANDOM_STATE, hidden_layer_sizes=100, max_iter=300, activation='relu', solver=i).fit(X_train, Y_train)
	# 	Y_pred_train = clf.predict(X_train)
	# 	Y_pred_test = clf.predict(X_test)
	# 	error_train.append(1-accuracy_score(Y_train, Y_pred_train))
	# 	error_test.append(1-accuracy_score(Y_test, Y_pred_test))
	
	# print(error_train)
	# print(error_test)

	'''
	Using 100 hidden layers
	RELU activation
	Adam solver
	max_iter 200
	Potential experiment: Toggle the above params?
	'''

#input: X_train, Y_train
#output: Y_pred
def boost_pred_insample(X_train, Y_train):
	#train a logistic regression classifier using X_train and Y_train. Use this to predict labels of X_train
	#use max_depth as 5
	clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,max_depth=1, random_state=RANDOM_STATE).fit(X_train, Y_train)
	Y_pred = clf.predict(X_train)

	'''
	Using 100 n estimators (number of boosting stages to perform)
	learning rate is 0.1, the "shrinkage" of the trees at each iteration
	max depth 1, can be more aggressive about pruning because we are using boosting
	Potential experiment: Toggle the above params?
	'''

	return Y_pred

#input: X_train, Y_train and X_test
#output: Y_pred
def boost_pred_outsample(X_train, Y_train, X_test):
	#TODO:train a logistic regression classifier using X_train and Y_train. Use this to predict labels of X_test
	#IMPORTANT: use max_depth as 5. Else your test cases might fail.
	clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,max_depth=1, random_state=RANDOM_STATE).fit(X_train, Y_train)
	Y_pred = clf.predict(X_test)

	'''
	Using 100 n estimators (number of boosting stages to perform)
	learning rate is 0.1, the "shrinkage" of the trees at each iteration
	max depth 1, can be more aggressive about pruning because we are using boosting
	Potential experiment: Toggle the above params?
	'''

	return Y_pred

#input: X_train, Y_train and X_test
#output: Y_pred
def boost_tuning(X_train, X_test, Y_train, Y_test):
	#TODO:train a logistic regression classifier using X_train and Y_train. Use this to predict labels of X_test
	#IMPORTANT: use max_depth as 5. Else your test cases might fail.

	''' Code for hypertuning by num estimators. num iterators = 175 is best for FLIGHT'''
	''' Code for hypertuning by num estimators. num iterators = 150 is best'''

	# d = list(range(30,301))
	# error_train = []
	# error_test = []

	# for i in d:
	# 	print(i)
	# 	clf = GradientBoostingClassifier(n_estimators=i, learning_rate=0.1,max_depth=1, random_state=RANDOM_STATE).fit(X_train, Y_train)
	# 	Y_pred_train = clf.predict(X_train)
	# 	Y_pred_test = clf.predict(X_test)
	# 	error_train.append(1-accuracy_score(Y_train, Y_pred_train))
	# 	error_test.append(1-accuracy_score(Y_test, Y_pred_test))
	
	# plt.plot(d, error_train, label = "Training Error")
	# plt.plot(d, error_test, label = "Validation Error")
	# plt.xlabel('# Estimators')
	# # Set the y axis label of the current axis.
	# plt.ylabel('Error')
	# # Set a title of the current axes.
	# plt.title('Gradient Tree Boosting Hypertuning of # Estimators')	
	# plt.legend()
	# plt.savefig('Boost_TuningEstimators')

	''' Code for hypertuning by learning rate. Best is 0.4 for FLIGHT'''
	''' Code for hypertuning by learning rate. Best is 0.2 for FLIGHT'''

	# d = list(np.linspace(0,1,11))[1:]
	# error_train = []
	# error_test = []

	# for i in d:
	# 	print(i)
	# 	clf = GradientBoostingClassifier(n_estimators=100, learning_rate=i,max_depth=1, random_state=RANDOM_STATE).fit(X_train, Y_train)
	# 	Y_pred_train = clf.predict(X_train)
	# 	Y_pred_test = clf.predict(X_test)
	# 	error_train.append(1-accuracy_score(Y_train, Y_pred_train))
	# 	error_test.append(1-accuracy_score(Y_test, Y_pred_test))
	
	# plt.plot(d, error_train, label = "Training Error")
	# plt.plot(d, error_test, label = "Validation Error")
	# plt.xlabel('Learning Rate')
	# # Set the y axis label of the current axis.
	# plt.ylabel('Error')
	# # Set a title of the current axes.
	# plt.title('Gradient Tree Boosting Hypertuning of Learning Rate')	
	# plt.legend()
	# plt.savefig('Boost_TuningLearningRate')

	''' Code for hypertuning by max depth. Best is 1 for FLIGHT'''
	''' Code for hypertuning by max depth. Best is 3 for EEG'''


	# d = list(range(1,51))
	# error_train = []
	# error_test = []

	# for i in d:
	# 	print(i)
	# 	clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.2,max_depth=i, random_state=RANDOM_STATE).fit(X_train, Y_train)
	# 	Y_pred_train = clf.predict(X_train)
	# 	Y_pred_test = clf.predict(X_test)
	# 	error_train.append(1-accuracy_score(Y_train, Y_pred_train))
	# 	error_test.append(1-accuracy_score(Y_test, Y_pred_test))
	
	# plt.plot(d, error_train, label = "Training Error")
	# plt.plot(d, error_test, label = "Validation Error")
	# plt.xlabel('Max Depth')
	# # Set the y axis label of the current axis.
	# plt.ylabel('Error')
	# # Set a title of the current axes.
	# plt.title('Gradient Tree Boosting Hypertuning of Max Depth')	
	# plt.legend()
	# plt.savefig('Boost_TuningMaxDepth')



	'''
	Using 100 n estimators (number of boosting stages to perform)
	learning rate is 0.1, the "shrinkage" of the trees at each iteration
	max depth 1, can be more aggressive about pruning because we are using boosting
	Potential experiment: Toggle the above params?
	'''

#input: X_train, Y_train
#output: Y_pred
def svm_pred_insample(X_train, Y_train):
	#train a SVM classifier using X_train and Y_train. Use this to predict labels of X_train
	#use default params for the classifier
	clf = SVC(random_state=RANDOM_STATE, kernel='linear').fit(X_train, Y_train)
	Y_pred = clf.predict(X_train)

	'''
	Using linear kernel
	Required experiment: Toggle the kernel type, try at least one other
	'''	

	return Y_pred

#input: X_train, Y_train and X_test
#output: Y_pred
def svm_pred_outsample(X_train, Y_train, X_test):
	#TODO:train a SVM classifier using X_train and Y_train. Use this to predict labels of X_test
	#use default params for the classifier
	clf = SVC(random_state=RANDOM_STATE, kernel='linear').fit(X_train, Y_train)
	Y_pred = clf.predict(X_test)

	'''
	Using linear kernel
	Required experiment: Toggle the kernel type, try at least one other
	'''	

	return Y_pred

#input: X_train, Y_train and X_test
#output: Y_pred
def svm_tuning(X_train, X_test, Y_train, Y_test):
	#TODO:train a logistic regression classifier using X_train and Y_train. Use this to predict labels of X_test
	#IMPORTANT: use max_depth as 5. Else your test cases might fail.

	''' Code for hypertuning by hidden layer sizes. rbf is best for FLIGHT'''
	''' Code for hypertuning by hidden layer sizes. All are the same for EEG'''

	# d =['linear', 'rbf', 'sigmoid']
	# error_train = []
	# error_test = []

	# for i in d:
	# 	print(i)
	# 	clf = SVC(random_state=RANDOM_STATE, kernel=i).fit(X_train, Y_train)
	# 	Y_pred_train = clf.predict(X_train)
	# 	Y_pred_test = clf.predict(X_test)
	# 	error_train.append(1-accuracy_score(Y_train, Y_pred_train))
	# 	error_test.append(1-accuracy_score(Y_test, Y_pred_test))
	
	# print(error_train)
	# print(error_test)

	'''
	Using linear kernel
	Required experiment: Toggle the kernel type, try at least one other
	'''	

#input: X_train, Y_train
#output: Y_pred
def knn_pred_insample(X_train, Y_train):
	#train a SVM classifier using X_train and Y_train. Use this to predict labels of X_train
	#use default params for the classifier
	clf = KNeighborsClassifier(n_neighbors=5).fit(X_train, Y_train)	
	Y_pred = clf.predict(X_train)

	'''
	Using 5 nearest neighbors
	Required experiment: Toggle the params. smaller you go, more it should be overfit
	'''	

	return Y_pred

#input: X_train, Y_train and X_test
#output: Y_pred
def knn_pred_outsample(X_train, Y_train, X_test):
	#TODO:train a SVM classifier using X_train and Y_train. Use this to predict labels of X_test
	#use default params for the classifier
	clf = KNeighborsClassifier(n_neighbors=5).fit(X_train, Y_train)	
	Y_pred = clf.predict(X_test)

	'''
	Using 5 nearest neighbors
	Required experiment: Toggle the params. smaller you go, more it should be overfit
	'''	

	return Y_pred	

#input: X_train, Y_train and X_test
#output: Y_pred
def knn_tuning(X_train, X_test, Y_train, Y_test):
	#TODO:train a logistic regression classifier using X_train and Y_train. Use this to predict labels of X_test
	#IMPORTANT: use max_depth as 5. Else your test cases might fail.

	''' Code for hypertuning by hidden layer sizes. K = 30 is best for FLIGHT'''
	''' Code for hypertuning by hidden layer sizes. K = 20 is best for EEG'''

	d = list(range(1,51))
	error_train = []
	error_test = []

	for i in d:
		print(i)
		clf = KNeighborsClassifier(n_neighbors=i).fit(X_train, Y_train)	
		Y_pred_train = clf.predict(X_train)
		Y_pred_test = clf.predict(X_test)
		error_train.append(1-accuracy_score(Y_train, Y_pred_train))
		error_test.append(1-accuracy_score(Y_test, Y_pred_test))
	
	plt.plot(d, error_train, label = "Training Error")
	plt.plot(d, error_test, label = "Validation Error")
	plt.xlabel('K')
	# Set the y axis label of the current axis.
	plt.ylabel('Error')
	# Set a title of the current axes.
	plt.title('KNN Hypertuning of K Param')	
	plt.legend()
	plt.savefig('KNN_TuningK')	

	'''
	Using 5 nearest neighbors
	Required experiment: Toggle the params. smaller you go, more it should be overfit
	'''					

#input: Y_pred,Y_true
#output: accuracy, auc, precision, recall, f1-score
def classification_metrics(Y_pred, Y_true):
	#NOTE: It is important to provide the output in the same order
	
	# calculate accurcacy
	accuracy = accuracy_score(Y_true, Y_pred)
	# calculate auc 
	AUC = roc_auc_score(Y_true, Y_pred)
	# calculate precision
	precision = precision_score(Y_true, Y_pred)
	# calculate recall
	recall = recall_score(Y_true, Y_pred)
	# calculate f1-score
	f1score = f1_score(Y_true, Y_pred)	
	# RMSE
	# rmse = math.sqrt(((Y_true - Y_pred) ** 2).sum() / Y_true.shape[0])
	error_rate = 1-accuracy

	return accuracy,AUC,precision,recall,f1score,error_rate

#input: Name of classifier, predicted labels, actual labels
def display_metrics(classifierName,Y_pred,Y_true):
	print("______________________________________________")
	print(("Classifier: "+classifierName))
	acc, auc_, precision, recall, f1score, error_rate = classification_metrics(Y_pred,Y_true)
	print(("Accuracy: "+str(acc)))
	print(("AUC: "+str(auc_)))
	print(("Precision: "+str(precision)))
	print(("Recall: "+str(recall)))
	print(("F1-score: "+str(f1score)))
	print(("Error Rate: "+str(error_rate)))
	print("______________________________________________")
	print("")

def print_stats():

	print("----FlightDelays: In-Sample-----")

	# display_metrics("Logistic Regression",logistic_regression_pred(X_train,Y_train),Y_train)
	# display_metrics("Decision Tree",decisionTree_pred_insample(X_train_eData,y_train_eData),y_train_eData)
	# display_metrics("MLP",mlp_pred_insample(X_train_eData,y_train_eData),y_train_eData)
	# display_metrics("Boosting",boost_pred_insample(X_train_eData,y_train_eData),y_train_eData)
	# display_metrics("SVM",svm_pred_insample(X_train_eData,y_train_eData),y_train_eData)
	# display_metrics("KNN",knn_pred_insample(X_train_eData,y_train_eData),y_train_eData)

	print("-----Vote: In-Sample-----")
	display_metrics("Decision Tree",decisionTree_pred_insample(X_train_pData,y_train_pData),y_train_pData)
	display_metrics("MLP",mlp_pred_insample(X_train_pData,y_train_pData),y_train_pData)
	display_metrics("Boost",boost_pred_insample(X_train_pData,y_train_pData),y_train_pData)
	display_metrics("SVM",svm_pred_insample(X_train_pData,y_train_pData),y_train_pData)
	display_metrics("KNN",knn_pred_insample(X_train_pData,y_train_pData),y_train_pData)

	print("----FlightDelays: Out-Sample-----")

	# display_metrics("Logistic Regression",logistic_regression_pred(X_train,Y_train),Y_train)
	# display_metrics("Decision Tree",decisionTree_pred_outsample(X_train_eData,y_train_eData, X_test_eData),y_test_eData)
	# display_metrics("MLP",mlp_pred_outsample(X_train_eData,y_train_eData, X_test_eData),y_test_eData)
	# display_metrics("Boost",boost_pred_outsample(X_train_eData,y_train_eData, X_test_eData),y_test_eData)
	# display_metrics("SVM",svm_pred_outsample(X_train_eData,y_train_eData, X_test_eData),y_test_eData)
	# display_metrics("KNN",knn_pred_outsample(X_train_eData,y_train_eData, X_test_eData),y_test_eData)

	print("-----Vote: Out-Sample-----")
	display_metrics("Decision Tree",decisionTree_pred_outsample(X_train_pData,y_train_pData, X_test_pData),y_test_pData)
	display_metrics("MLP",mlp_pred_outsample(X_train_pData,y_train_pData, X_test_pData),y_test_pData)
	display_metrics("Boost",boost_pred_outsample(X_train_pData,y_train_pData, X_test_pData),y_test_pData)
	display_metrics("SVM",svm_pred_outsample(X_train_pData,y_train_pData, X_test_pData),y_test_pData)
	display_metrics("KNN",knn_pred_outsample(X_train_pData,y_train_pData, X_test_pData),y_test_pData)

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def iter_learning_curve(iter, X_train, X_test, y_train, y_test):
	accuracy_train = []
	accuracy_test = []
	for i in iter:
		estimator = MLPClassifier(random_state=RANDOM_STATE, max_iter=i)
		estimator.fit(X_train, y_train)
		Y_pred_train = estimator.predict(X_train)
		Y_pred_test = estimator.predict(X_test)
		accuracy_train.append(accuracy_score(y_train, Y_pred_train))
		accuracy_test.append(accuracy_score(y_test, Y_pred_test))

def gen_learning_curve(X, y):
	title = "Learning Curves (KNN)"
	# Cross validation with 100 iterations to get smoother mean test and train
	# score curves, each time with 20% data randomly selected as a validation set.
	cv = ShuffleSplit(n_splits=5,test_size=0.2, random_state=RANDOM_STATE)
	# estimator = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=8, criterion='entropy').fit(X, y)
	# estimator = MLPClassifier(random_state=RANDOM_STATE, hidden_layer_sizes=30, activation='relu', solver='sgd').fit(X, y)
	# estimator = GradientBoostingClassifier(n_estimators=150, learning_rate=0.2,max_depth=3, random_state=RANDOM_STATE).fit(X, y)
	# estimator = SVC(random_state=RANDOM_STATE, kernel='rbf').fit(X, y)
	estimator = KNeighborsClassifier(n_neighbors=20).fit(X, y)
	plot_learning_curve(estimator, title, X, y, cv=cv,n_jobs=4)
	# plt.plot(estimator.loss_curve_)
	plt.savefig('LearningCurves_KNN')
	# plt.show()

def main():
	eData = pd.read_pickle('FlightDelays.pkl')
	pData = pd.read_pickle('EEGEyeData.pkl')

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
	X_train_eData, X_test_eData, y_train_eData, y_test_eData = train_test_split(normalized_X_eData, y_eData, test_size=0.2, random_state=RANDOM_STATE)	

	pData['Class'] = pData['Class'].astype('category').cat.codes
	X_pData = pData.iloc[:,:-1]
	# X_pData = X_pData.replace('n',0)
	# X_pData = X_pData.replace('y',1)
	normalized_X_pData=(X_pData-X_pData.min())/(X_pData.max()-X_pData.min())
	# X_pData = X_pData.replace('b',0)
	# X_pData = X_pData.replace('o',1)
	# X_pData = X_pData.replace('x',2)
	y_pData = pData.iloc[:,-1]
	X_train_pData, X_test_pData, y_train_pData, y_test_pData = train_test_split(normalized_X_pData, y_pData, test_size=0.2, random_state=RANDOM_STATE)



	# mlp_tuning(X_train_eData, X_test_eData, y_train_eData, y_test_eData)
	# mlp_tuning(X_train_pData, X_test_pData, y_train_pData, y_test_pData)
	gen_learning_curve(X_train_pData, y_train_pData)

if __name__ == "__main__":
	main()
	
