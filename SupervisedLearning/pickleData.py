import pandas as pd
import pickle


def main():
	eData = pd.read_csv('FlightDelays.csv')
	eData.to_pickle("./FlightDelays.pkl")

	pData = pd.read_csv('EEGEyeData.csv')
	pData.to_pickle("./EEGEyeData.pkl")
	



	# display_metrics("Logistic Regression",logistic_regression_pred(X_train,Y_train),Y_train)
	# display_metrics("SVM",svm_pred(X_train,Y_train),Y_train)
	# display_metrics("Decision Tree",decisionTree_pred(X_train,Y_train),Y_train)
	

if __name__ == "__main__":
	main()
	
