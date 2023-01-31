import numpy as np
import matplotlib.pyplot as plt
import time
import random
import pandas as pd
from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.mixture import GaussianMixture
from sklearn.random_projection import GaussianRandomProjection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from scipy.stats import kurtosis
from copy import *

# setup the randoms tate
RANDOM_STATE = 545510477

def intra_cluster(X):
	# k = [2,3,4,5,6]
	# inertia = []
	# for i in k:
	# 	km = KMeans(n_clusters=i, init='k-means++', random_state=RANDOM_STATE)
	# 	#
	# 	# Fit the KMeans model
	# 	#
	# 	km.fit(X)
	# 	#
	# 	# Calculate Silhoutte Score
	# 	#
	# 	inertia.append(km.inertia_)
	# 	#
	# 	# Print the score
	# 	#
	# 	# print('Silhouetter Score: %.3f' % score)
	# x = k
	# plt.plot(x, inertia, label = "Inertia Score")
	# plt.xlabel('Clusters')
	# plt.ylabel('Inertia Scores')	
	# plt.title('Intra Cluster Density')		
	# plt.legend()
	# plt.savefig('KMeans_IntraCluster')
	# plt.close()	

	km = KMeans(n_clusters=2, init='k-means++', random_state=RANDOM_STATE)
	km.fit(X)
	print(km.inertia_)

def inter_cluster(X):
	k = [2,3,4,5,6]
	# scores = []
	# for i in k:
	# 	km = KMeans(n_clusters=i, init='k-means++', random_state=RANDOM_STATE)
	# 	#
	# 	# Fit the KMeans model
	# 	#
	# 	km.fit_predict(X)
	# 	#
	# 	# Calculate Silhoutte Score
	# 	#
	# 	scores.append(silhouette_score(X, km.labels_, metric='euclidean'))
	# 	#
	# 	# Print the score
	# 	#
	# 	# print('Silhouetter Score: %.3f' % score)
	# x = k
	# plt.plot(x, scores, label = "Silhouette score")
	# plt.xlabel('Clusters')
	# plt.ylabel('Silhouette Scores')	
	# plt.title('Inter Cluster Spacing')		
	# plt.legend()
	# plt.savefig('KMeans_InterCluster')
	# plt.close()
	km = KMeans(n_clusters=2, init='k-means++', random_state=RANDOM_STATE)		
	km.fit_predict(X)
	score = silhouette_score(X, km.labels_, metric='euclidean')
	print(score)



def plot_kmeans_cluster(X,Y, dimred):
	scaler = StandardScaler()
	scaled = scaler.fit_transform(X)
	scores_pca = dimred.transform(scaled)
	# scores_pca = np.delete(scores_pca, [0,1,3,4,6,8,9,11,12,13], 1)	

	km = KMeans(n_clusters=5, init='k-means++', random_state=RANDOM_STATE)
	km.fit(scores_pca)
	y_kmeans = km.predict(scores_pca)

	df = pd.DataFrame(scores_pca)
	df['cluster'] = y_kmeans

	clus0 = df[df['cluster']==0]
	clus1 = df[df['cluster']==1]
	clus2 = df[df['cluster']==2]	
	clus3 = df[df['cluster']==3]
	clus4 = df[df['cluster']==4]

	print(clus0)	

	plt.clf()
	plt.scatter(clus0[clus0.columns[1]], clus0[clus0.columns[2]], marker='o', color='r', label='Cluster0')
	plt.scatter(clus1[clus1.columns[1]], clus1[clus1.columns[2]], marker='o', color='g', label='Cluster1')
	plt.scatter(clus2[clus2.columns[1]], clus2[clus2.columns[2]], marker='o', color='b', label='Cluster2')
	plt.scatter(clus3[clus3.columns[1]], clus3[clus3.columns[2]], marker='o', color='y', label='Cluster3')
	plt.scatter(clus4[clus4.columns[1]], clus4[clus4.columns[2]], marker='o', color='m', label='Cluster4')
	plt.legend()
	plt.title('K-Means Clustering across 2 Features')
	plt.xlabel('Feature 2')
	plt.ylabel('Feature 3')
	plt.savefig('VizKMeans_PostBE_Flight')


def do_kmeans(X,Y):
	km = KMeans(n_clusters=2, init='k-means++', random_state=RANDOM_STATE)

	km.fit(X)
	y_kmeans = km.predict(X)


	df = pd.DataFrame({'K-Means Cluster': y_kmeans, 'True Classification Label': Y})
	pd.set_option('display.max_rows', None)
	
	print(len(df[df['K-Means Cluster']==0]))
	print(len(df[df['K-Means Cluster']==1]))
	print(len(df[df['K-Means Cluster']==2]))
	# correct = df[df['K-Means Cluster'] == df['True Classification Label']]
	# print(len(correct.index)/len(df.index))
	# print(len(df[df['K-Means Cluster'] == 0]))
	# print(len(df[df['K-Means Cluster'] == 1]))
	# print(len(df[df['K-Means Cluster'] == 2]))
	# print(len(df[(df['True Classification Label'] == 0) & (df['K-Means Cluster'] == 0)]))
	# print(len(df[(df['True Classification Label'] == 0) & (df['K-Means Cluster'] == 1)]))

	# print(len(df[(df['True Classification Label'] == 1) & (df['K-Means Cluster'] == 0)]))
	# print(len(df[(df['True Classification Label'] == 1) & (df['K-Means Cluster'] == 1)]))	
	# print(len(df[(df['K-Means Cluster'] == 0) & (df['True Classification Label'] == 1)]))

def do_gm(X,Y):
	gm = GaussianMixture(n_components=3, random_state=RANDOM_STATE, covariance_type='full', warm_start = True)
	# scaler = StandardScaler()
	# scaled = scaler.fit_transform(X)
	# scores_pca = dimred.transform(scaled)
	# scores_pca = np.delete(scores_pca, [0,1,3,4,6,8,9,11,12,13], 1)		

	gm.fit(X)	

	y_gm = gm.predict(X)

	df = pd.DataFrame({'GM Cluster': y_gm, 'True Classification Label': Y})
	pd.set_option('display.max_rows', None)
	print(len(df[df['GM Cluster']==0]))
	print(len(df[df['GM Cluster']==1]))
	print(len(df[df['GM Cluster']==2]))	
	# correct = df[df['GM Cluster'] == df['True Classification Label']]
	# print(len(correct.index)/len(df.index))	

	# gm = GaussianMixture(n_components=3, random_state=RANDOM_STATE, covariance_type='spherical', warm_start = True)
	# gm.fit(scores_pca)	

	# y_gm = gm.predict(scores_pca)

	# df = pd.DataFrame({'GM Cluster': y_gm, 'True Classification Label': Y})
	# pd.set_option('display.max_rows', None)
	# print(len(df[df['GM Cluster']==0]))
	# print(len(df[df['GM Cluster']==1]))
	# print(len(df[df['GM Cluster']==2]))		
	# correct = df[df['GM Cluster'] == df['True Classification Label']]
	# print(len(correct.index)/len(df.index))	

	# gm = GaussianMixture(n_components=3, random_state=RANDOM_STATE, covariance_type='diag', warm_start = True)
	# gm.fit(scores_pca)	

	# y_gm = gm.predict(scores_pca)

	# df = pd.DataFrame({'GM Cluster': y_gm, 'True Classification Label': Y})
	# pd.set_option('display.max_rows', None)
	# print(len(df[df['GM Cluster']==0]))
	# print(len(df[df['GM Cluster']==1]))
	# print(len(df[df['GM Cluster']==2]))		
	# correct = df[df['GM Cluster'] == df['True Classification Label']]
	# print(len(correct.index)/len(df.index))	

	# gm = GaussianMixture(n_components=3, random_state=RANDOM_STATE, covariance_type='tied', warm_start = True)
	# gm.fit(scores_pca)	

	# y_gm = gm.predict(scores_pca)

	# df = pd.DataFrame({'GM Cluster': y_gm, 'True Classification Label': Y})
	# pd.set_option('display.max_rows', None)
	# print(len(df[df['GM Cluster']==0]))
	# print(len(df[df['GM Cluster']==1]))
	# print(len(df[df['GM Cluster']==2]))		
	# correct = df[df['GM Cluster'] == df['True Classification Label']]
	# print(len(correct.index)/len(df.index))				
	
	# print(len(df[df['GM Cluster']==0]))
	# print(len(df[df['GM Cluster']==1]))
	# print(len(df[(df['True Classification Label'] == 1) & (df['GM Cluster'] == 0)]))
	# print(len(df[(df['True Classification Label'] == 1) & (df['GM Cluster'] == 1)]))

def dimred_gm(X,Y, dimred):
	gm = GaussianMixture(n_components=3, random_state=RANDOM_STATE, covariance_type='full', warm_start = True)
	scaler = StandardScaler()
	scaled = scaler.fit_transform(X)
	scores_pca = dimred.transform(scaled)
	scores_pca = np.delete(scores_pca, [0,1,3,4,6,8,9,11,12,13], 1)		

	gm.fit(scores_pca)	

	y_gm = gm.predict(scores_pca)
	print("==1==")
	df = pd.DataFrame({'GM Cluster': y_gm, 'True Classification Label': Y})
	pd.set_option('display.max_rows', None)
	# print(len(df[df['GM Cluster']==0]))
	# print(len(df[df['GM Cluster']==1]))
	# print(len(df[df['GM Cluster']==2]))	
	new_df = df.replace({'GM Cluster': 1}, 0)
	new_df = df.replace({'GM Cluster': 2}, 1)	

	correct = new_df[new_df['GM Cluster'] == new_df['True Classification Label']]
	print(len(correct.index)/len(df.index))	

	gm = GaussianMixture(n_components=3, random_state=RANDOM_STATE, covariance_type='spherical', warm_start = True)
	gm.fit(scores_pca)	

	y_gm = gm.predict(scores_pca)
	print("==2==")
	df = pd.DataFrame({'GM Cluster': y_gm, 'True Classification Label': Y})
	pd.set_option('display.max_rows', None)
	# print(len(df[df['GM Cluster']==0]))
	# print(len(df[df['GM Cluster']==1]))
	# print(len(df[df['GM Cluster']==2]))		
	correct = df[df['GM Cluster'] == df['True Classification Label']]
	print(len(correct.index)/len(df.index))	

	gm = GaussianMixture(n_components=3, random_state=RANDOM_STATE, covariance_type='diag', warm_start = True)
	gm.fit(scores_pca)	

	y_gm = gm.predict(scores_pca)
	print("==3==")
	df = pd.DataFrame({'GM Cluster': y_gm, 'True Classification Label': Y})
	pd.set_option('display.max_rows', None)
	# print(len(df[df['GM Cluster']==0]))
	# print(len(df[df['GM Cluster']==1]))
	# print(len(df[df['GM Cluster']==2]))		
	new_df = df.replace({'GM Cluster': 1}, 0)
	new_df = df.replace({'GM Cluster': 2}, 0)	

	correct = new_df[new_df['GM Cluster'] == new_df['True Classification Label']]
	print(len(correct.index)/len(df.index))	

	gm = GaussianMixture(n_components=3, random_state=RANDOM_STATE, covariance_type='tied', warm_start = True)
	gm.fit(scores_pca)	

	y_gm = gm.predict(scores_pca)
	print("==4==")
	df = pd.DataFrame({'GM Cluster': y_gm, 'True Classification Label': Y})
	pd.set_option('display.max_rows', None)
	# print(len(df[df['GM Cluster']==0]))
	# print(len(df[df['GM Cluster']==1]))
	# print(len(df[df['GM Cluster']==2]))		
	correct = df[df['GM Cluster'] == df['True Classification Label']]
	print(len(correct.index)/len(df.index))				
	
	# print(len(df[df['GM Cluster']==0]))
	# print(len(df[df['GM Cluster']==1]))
	# print(len(df[(df['True Classification Label'] == 1) & (df['GM Cluster'] == 0)]))
	# print(len(df[(df['True Classification Label'] == 1) & (df['GM Cluster'] == 1)]))

def dimred_kmeans(X,Y,dimred):
	scaler = StandardScaler()
	scaled = scaler.fit_transform(X)
	scores_pca = dimred.transform(scaled)
	# scores_pca = np.delete(scores_pca, [0,1,3,4,6,8,9,11,12,13], 1)		

	km = KMeans(n_clusters=5, init='k-means++', random_state=RANDOM_STATE)
	km.fit(scores_pca)
	y_kmeans = km.predict(scores_pca)
	df = pd.DataFrame({'K-Means Cluster': y_kmeans, 'True Classification Label': Y})
	pd.set_option('display.max_rows', None)
	# new_df = df.replace({'K-Means Cluster': 1}, 0)
	# new_df = df.replace({'K-Means Cluster': 2}, 1)

	print(len(df[df['K-Means Cluster']==0]))
	print(len(df[df['K-Means Cluster']==1]))
	print(len(df[df['K-Means Cluster']==2]))	
	print(len(df[df['K-Means Cluster']==3]))
	print(len(df[df['K-Means Cluster']==4]))

	print(len(df[(df['True Classification Label'] == 0) & (df['K-Means Cluster'] == 0)]))
	print(len(df[(df['True Classification Label'] == 0) & (df['K-Means Cluster'] == 1)]))
	print(len(df[(df['True Classification Label'] == 0) & (df['K-Means Cluster'] == 2)]))
	print(len(df[(df['True Classification Label'] == 0) & (df['K-Means Cluster'] == 3)]))
	print(len(df[(df['True Classification Label'] == 0) & (df['K-Means Cluster'] == 4)]))

	print(len(df[(df['True Classification Label'] == 1) & (df['K-Means Cluster'] == 0)]))
	print(len(df[(df['True Classification Label'] == 1) & (df['K-Means Cluster'] == 1)]))
	print(len(df[(df['True Classification Label'] == 1) & (df['K-Means Cluster'] == 2)]))
	print(len(df[(df['True Classification Label'] == 1) & (df['K-Means Cluster'] == 3)]))
	print(len(df[(df['True Classification Label'] == 1) & (df['K-Means Cluster'] == 4)]))		
	# correct = new_df[new_df['K-Means Cluster'] == new_df['True Classification Label']]
	# print(len(correct.index)/len(df.index))	

def dimred_wcss(X, y, dimred):
	scaler = StandardScaler()
	scaled = scaler.fit_transform(X)
	# pca = do_PCA(X, 5)
	scores_pca = dimred.transform(scaled)
	# scores_pca = np.delete(scores_pca, [0,1,3,4,6,8,9,11,12,13], 1)	
	# scores_pca = np.delete(scores_pca, [0,1,6], 1)	

	wcss(scores_pca)

def dimred_sil(X, y, dimred):
	scaler = StandardScaler()
	scaled = scaler.fit_transform(X)
	# pca = do_PCA(X, 5)
	scores_pca = dimred.transform(scaled)
	# scores_pca = np.delete(scores_pca, [0,1,3,4,6,8,9,11,12,13], 1)	
	# scores_pca = np.delete(scores_pca, [0,1,6], 1)	

	plot_sil(scores_pca)	


def do_sfs(X,y):
	scaler = StandardScaler()
	scaled = scaler.fit_transform(X)	
	knn = KNeighborsClassifier(n_neighbors=3)
	sfs = SequentialFeatureSelector(knn, n_features_to_select=6, direction='backward')
	sfs.fit(scaled, y)
	print(sfs.get_support())

	
	return sfs

def do_randproj(X, components):
	scaler = StandardScaler()
	scaled = scaler.fit_transform(X)
	num_features = scaled.shape[1]
	mse_rp_1 = []
	mse_rp_2 = []
	mse_rp_3 = []
	mse_rp_4 = []
	mse_rp_5 = []

	iters = 5
	grp = GaussianRandomProjection(n_components=6)
	grp.fit(scaled)

	# for i in range(iters):
	# 	for j in range(num_features):
	# 		grp = GaussianRandomProjection(n_components=j+1)
	# 		grp.fit(scaled)

	# 		transformed_data = grp.transform(scaled)
	# 		inverse_data = np.linalg.pinv(grp.components_.T)
	# 		reconstructed_data = transformed_data.dot(inverse_data)	

	# 		mse = (np.square(scaled - reconstructed_data)).mean()
	# 		if i==0:	
	# 			mse_rp_1.append(mse)
	# 		elif i==1:	
	# 			mse_rp_2.append(mse)	
	# 		elif i==2:	
	# 			mse_rp_3.append(mse)					
	# 		elif i==3:	
	# 			mse_rp_4.append(mse)	
	# 		elif i==4:	
	# 			mse_rp_5.append(mse)	

	# a = np.array([mse_rp_1, mse_rp_2, mse_rp_3, mse_rp_4, mse_rp_5])
	# mean = np.mean(a, axis=0)
	# var = np.std(a, axis=0)


	# plt.plot(range(1,num_features+1), mse_rp_1, marker='o', linestyle='--', label='Iter1')
	# plt.plot(range(1,num_features+1), mse_rp_2, marker='o', linestyle='--', label='Iter2')
	# plt.plot(range(1,num_features+1), mse_rp_3, marker='o', linestyle='--', label='Iter3')
	# plt.plot(range(1,num_features+1), mse_rp_4, marker='o', linestyle='--', label='Iter4')
	# plt.plot(range(1,num_features+1), mse_rp_5, marker='o', linestyle='--', label='Iter5')
	# plt.fill_between(range(1,num_features+1), mean+var, mean-var, color='grey', alpha=0.3)
	# plt.legend()
	# plt.title('Reconstruction Error by Number of Components')
	# plt.xlabel('Number of Components')
	# plt.ylabel('Reconstruction Error')
	# plt.savefig('doRP')	

	return grp	

def do_ICA(X, components):
	scaler = StandardScaler()
	scaled = scaler.fit_transform(X)
	num_features = scaled.shape[1]
	mean_kurt = []
	# for i in range(num_features):
	# 	ica = FastICA(n_components=i+1, random_state=RANDOM_STATE)
	# 	ica.fit(scaled)
	# 	mean_kurt.append(np.mean(kurtosis(ica.components_)+3))
	ica = FastICA(n_components=num_features, random_state=RANDOM_STATE)
	ica.fit(scaled)
	
	identity = np.identity(num_features)

	# print(kurtosis(ica.components_,fisher=False))
	# plt.plot(range(1,num_features+1), kurtosis(ica.components_,fisher=False), marker='o', linestyle='--')
	# plt.title('Kurtosis by Component')
	# plt.xlabel('Component')
	# plt.ylabel('Kurtosis')
	# plt.savefig('doICA')	

	return ica

def do_PCA(X, components):
	scaler = StandardScaler()
	scaled = scaler.fit_transform(X)
	num_features = scaled.shape[1]
	mse_pca =[]

	pca = PCA(n_components=5, random_state=RANDOM_STATE)
	pca.fit(scaled)
	# for i in range(num_features):
	# 	pca = PCA(n_components=i+1, random_state=RANDOM_STATE)
	# 	pca.fit(scaled)

	# 	transformed_data = pca.transform(scaled)
	# 	inverse_data = np.linalg.pinv(pca.components_.T)
	# 	reconstructed_data = transformed_data.dot(inverse_data)	

	# 	mse = (np.square(scaled - reconstructed_data)).mean()
	# 	mse_pca.append(mse)	
	# 	pca = PCA(n_components=i+1, random_state=RANDOM_STATE)

	# print(pca.explained_variance_ratio_)
	# # plt.figure(figsize=(10,8))
	# plt.plot(range(1,8), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
	# plt.title('Explained Variance by Components')
	# plt.xlabel('Number of Components')
	# plt.ylabel('Cumulative Explained Variance')
	# plt.savefig('doPCA')

	# plt.figure(figsize=(10,8))
	# plt.plot(range(1,num_features+1), mse_pca, marker='o', linestyle='--')
	# plt.title('Reconstruction Error by Number of Components')
	# plt.xlabel('Number of Components')
	# plt.ylabel('Reconstruction Error')
	# plt.savefig('doPCAReconstruct')	

	return pca


def do_PCA_kmeans(X, pca):
	pca = PCA(n_components=5)
	pca.fit(X)
	scores_pca = pca.transform(X)
	wcss(scores_pca)

def plot_sil(X):
	fig, ax = plt.subplots(2, 2, figsize=(15,8))
	for i in [2, 3, 4, 5]:
	    '''
	    Create KMeans instance for different number of clusters
	    '''
	    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=RANDOM_STATE)
	    q, mod = divmod(i, 2)
	    '''
	    Create SilhouetteVisualizer instance with KMeans instance
	    Fit the visualizer
	    '''
	    visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q-1][mod])
	    visualizer.fit(X)
	    visualizer.show(outpath="sil_viz.png") 	

def sil_score(X, dimred):
	scaler = StandardScaler()
	scaled = scaler.fit_transform(X)
	scores_pca = dimred.transform(scaled)
	# scores_pca = np.delete(scores_pca, [0,1,3,4,6,8,9,11,12,13], 1)

	km = KMeans(n_clusters=5, random_state=RANDOM_STATE)
	gm_full = GaussianMixture(n_components=5, random_state=RANDOM_STATE, covariance_type='full', warm_start = True)
	gm_sphere = GaussianMixture(n_components=5, random_state=RANDOM_STATE, covariance_type='spherical', warm_start = True)
	gm_diag = GaussianMixture(n_components=5, random_state=RANDOM_STATE, covariance_type='diag', warm_start = True)
	gm_tied = GaussianMixture(n_components=5, random_state=RANDOM_STATE, covariance_type='tied', warm_start = True)


	km.fit_predict(scores_pca)
	gm_full_labels = gm_full.fit_predict(scores_pca)
	gm_sphere_labels = gm_sphere.fit_predict(scores_pca)
	gm_diag_labels = gm_diag.fit_predict(scores_pca)
	gm_tied_labels = gm_tied.fit_predict(scores_pca)

	score = silhouette_score(scores_pca, km.labels_, metric='euclidean')
	print('K-Means Score: %.3f' % score)

	score = silhouette_score(scores_pca, gm_full_labels, metric='euclidean')
	print('GM-Full Score: %.3f' % score)

	score = silhouette_score(scores_pca, gm_sphere_labels, metric='euclidean')
	print('GM-Sphere Score: %.3f' % score)

	score = silhouette_score(scores_pca, gm_diag_labels, metric='euclidean')
	print('GM-Diag Score: %.3f' % score)

	score = silhouette_score(scores_pca, gm_tied_labels, metric='euclidean')
	print('GM-Tied Score: %.3f' % score)
			


def wcss(X):
	'''
	https://readthedocs.org/projects/mlrose/downloads/pdf/stable/
	'''
	wcss = []
	for i in range(1, 20):
	    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=RANDOM_STATE)
	    kmeans.fit(X)
	    wcss.append(kmeans.inertia_)
	plt.plot(range(1, 20), wcss)
	plt.title('Elbow Method')
	plt.xlabel('Number of clusters')
	plt.ylabel('WCSS')
	plt.savefig('WCSS')

def dimred_nn(X_train, y_train, X_test, y_test, dimred):
	scaler = StandardScaler()
	scaled_Xtrain = scaler.fit_transform(X_train)
	scores_Xtrain = dimred.transform(scaled_Xtrain)
	# scores_Xtrain = np.delete(scores_Xtrain, [0,1,3,4,6,8,9,11,12,13], 1)		

	scaled_Xtest = scaler.fit_transform(X_test)
	scores_Xtest = dimred.transform(scaled_Xtest)
	# scores_Xtest = np.delete(scores_Xtest, [0,1,3,4,6,8,9,11,12,13], 1)	

	# ''' Code for hypertuning by hidden layers. hidden layers = 30 is best for FLIGHT'''
	# d = list(range(1,31))
	# error_train = []
	# error_test = []

	# for i in d:
	# 	print(i)
	# 	clf = MLPClassifier(random_state=RANDOM_STATE, hidden_layer_sizes=i, max_iter=100, solver='sgd').fit(scores_Xtrain, y_train)
	# 	Y_pred_train = clf.predict(scores_Xtrain)
	# 	Y_pred_test = clf.predict(scores_Xtest)
	# 	error_train.append(1-accuracy_score(y_train, Y_pred_train))
	# 	error_test.append(1-accuracy_score(y_test, Y_pred_test))
	
	# plt.plot(d, error_train, label = "Training Error")
	# plt.plot(d, error_test, label = "Validation Error")
	# plt.xlabel('# Hidden Layers')
	# # Set the y axis label of the current axis.
	# plt.ylabel('Error')
	# # Set a title of the current axes.
	# plt.title('MLP Hypertuning of # Hidden Layers')	
	# plt.legend()
	# plt.savefig('MLP_TuningHiddenLayer')
	

	# ''' Code for hypertuning by activation function. Best doesn't matter for FLIGHT'''
	# ''' Code for hypertuning by activation function. Best is relu for EEG'''

	# d = ['tanh','relu']
	# error_train = []
	# error_test = []

	# for i in d:
	# 	print(i)
	# 	clf = MLPClassifier(random_state=RANDOM_STATE, hidden_layer_sizes=30, activation=i, solver='sgd').fit(scores_Xtrain, y_train)
	# 	Y_pred_train = clf.predict(scores_Xtrain)
	# 	Y_pred_test = clf.predict(scores_Xtest)
	# 	error_train.append(1-accuracy_score(y_train, Y_pred_train))
	# 	error_test.append(1-accuracy_score(y_test, Y_pred_test))
	
	# print(error_train)
	# print(error_test)

	f1_train_old = []
	f1_test_old = []

	# f1_train_new = []
	# f1_test_new = []	
	# for i in iters:
	# 	print(i)
	# 	clf_old = MLPClassifier(random_state=RANDOM_STATE, hidden_layer_sizes=30, max_iter=100, activation='relu',solver='sgd').fit(X_train, y_train)
	# 	Y_pred_train = clf_old.predict(X_train)
	# 	Y_pred_test = clf_old.predict(X_test)
	# 	f1_train_old.append(f1_score(y_train, Y_pred_train))
	# 	f1_test_old.append(f1_score(y_test, Y_pred_test))

	# 	clf_new = MLPClassifier(random_state=RANDOM_STATE, hidden_layer_sizes=30, max_iter=100, activation='relu',solver='sgd').fit(scores_Xtrain, y_train)
	# 	Y_pred_train = clf_new.predict(scores_Xtrain)
	# 	Y_pred_test = clf_new.predict(scores_Xtest)
	# 	f1_train_new.append(f1_score(y_train, Y_pred_train))
	# 	f1_test_new.append(f1_score(y_test, Y_pred_test))


	# plt.plot(iters, f1_train_old, marker='o', linestyle='--', label='NN_Train')
	# plt.plot(iters, f1_train_new, marker='o', linestyle='--', label='NN_Train_with_DR')
	# plt.plot(iters, f1_test_old, marker='o', linestyle='--', label='NN_Test')
	# plt.plot(iters, f1_test_new, marker='o', linestyle='--', label='NN_Test_with_DR')
	# plt.legend()
	# plt.title('MLP Performance Comparison with/without DR')
	# plt.xlabel('Iterations')
	# plt.ylabel('F1 Score')
	# plt.savefig('MLP_PerfComparison_DR')	

def plot_learning_curve(estimator_old, estimator_new, title, X, y, X_new, y_new, ylim=None, cv=None,
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
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator_old, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    # plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
    #                  train_scores_mean + train_scores_std, alpha=0.1,
    #                  color="r")
    # plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
    #                  test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator_new, X_new, y_new, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    # plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
    #                  train_scores_mean + train_scores_std, alpha=0.1,
    #                  color="r")
    # plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
    #                  test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="y",
             label="DR -- Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="b",
             label="DR -- Cross-validation score")    

    plt.legend(loc="best")
    return plt

def gen_learning_curve(X, y, dimred):
	scaler = StandardScaler()
	scaled_X = scaler.fit_transform(X)
	scores_X = dimred.transform(scaled_X)
	# scores_X = np.delete(scores_X, [0,1,6], 1)	

	
	title = "MLP Performance Comparison with/without BE"
	# Cross validation with 100 iterations to get smoother mean test and train
	# score curves, each time with 20% data randomly selected as a validation set.
	cv = ShuffleSplit(n_splits=5,test_size=0.2, random_state=RANDOM_STATE)
	# estimator = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=8, criterion='entropy').fit(X, y)
	
	estimator_old = MLPClassifier(random_state=RANDOM_STATE, hidden_layer_sizes=30, activation='relu', solver='sgd')
	time_start_old = time.time()
	estimator_old.fit(X, y)
	time_elapsed_old = (time.time() - time_start_old)

	estimator_new = MLPClassifier(random_state=RANDOM_STATE, hidden_layer_sizes=30, activation='relu', solver='sgd')
	time_start_new = time.time()
	estimator_new.fit(scores_X, y)
	time_elapsed_new = (time.time() - time_start_new)

	print('===Old Time===')
	print(time_elapsed_old)
	print('===New Time===')	
	print(time_elapsed_new)
	# estimator = GradientBoostingClassifier(n_estimators=150, learning_rate=0.2,max_depth=3, random_state=RANDOM_STATE).fit(X, y)
	# estimator = SVC(random_state=RANDOM_STATE, kernel='rbf').fit(X, y)
	# estimator = KNeighborsClassifier(n_neighbors=20).fit(X, y)
	plot_learning_curve(estimator_old, estimator_new, title, X, y, scores_X, y, cv=cv,n_jobs=4)

	# plt.plot(estimator.loss_curve_)
	plt.savefig('MLP_PerfComparison_BE')
	# plt.show()

def gen_learning_curve_cluster(X, y, clus):
	# scaler = StandardScaler()
	# scaled_X = scaler.fit_transform(X)
	# scores_X = dimred.transform(scaled_X)
	# scores_X = np.delete(scores_X, [0,1,6], 1)
	# km = KMeans(n_clusters=3, init='k-means++', random_state=RANDOM_STATE).fit(X)
	# gm_full = GaussianMixture(n_components=3, random_state=RANDOM_STATE, covariance_type='full', warm_start = True).fit(X)
	# gm_sphere = GaussianMixture(n_components=3, random_state=RANDOM_STATE, covariance_type='spherical', warm_start = True).fit(X)
	# gm_diag = GaussianMixture(n_components=3, random_state=RANDOM_STATE, covariance_type='diag', warm_start = True).fit(X)
	# gm_tied = GaussianMixture(n_components=3, random_state=RANDOM_STATE, covariance_type='tied', warm_start = True).fit(X)

	clus.fit(X)
	y_clus = clus.predict(X)
	X_new = deepcopy(X)
	X_new['cluster'] = y_clus

	clus0 = pd.DataFrame((X_new['cluster'] == 0)).astype(int)
	clus1 = pd.DataFrame((X_new['cluster'] == 1)).astype(int)
	clus2 = pd.DataFrame((X_new['cluster'] == 2)).astype(int)

	# y_gm_full = gm_full.predict(X)
	# y_gm_sphere = gm_sphere.predict(X)
	# y_gm_diag = gm_diag.predict(X)
	# y_gm_tied = gm_tied.predict(X)
	df_X = pd.DataFrame({'Cluster0':clus0['cluster'], 'Cluster1':clus1['cluster'], 'Cluster2':clus2['cluster']})
	pd.set_option('display.max_rows', None)


	title = "MLP Performance Comparison with/without K-Means Features"
	# Cross validation with 100 iterations to get smoother mean test and train
	# score curves, each time with 20% data randomly selected as a validation set.
	cv = ShuffleSplit(n_splits=5,test_size=0.2, random_state=RANDOM_STATE)
	# estimator = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=8, criterion='entropy').fit(X, y)
	
	estimator_old = MLPClassifier(random_state=RANDOM_STATE, hidden_layer_sizes=30, activation='relu', solver='sgd')
	time_start_old = time.time()
	estimator_old.fit(X, y)
	time_elapsed_old = (time.time() - time_start_old)

	estimator_new = MLPClassifier(random_state=RANDOM_STATE, hidden_layer_sizes=30, activation='relu', solver='sgd')
	time_start_new = time.time()
	estimator_new.fit(df_X, y)
	time_elapsed_new = (time.time() - time_start_new)

	print('===Old Time===')
	print(time_elapsed_old)
	print('===New Time===')	
	print(time_elapsed_new)
	# estimator = GradientBoostingClassifier(n_estimators=150, learning_rate=0.2,max_depth=3, random_state=RANDOM_STATE).fit(X, y)
	# estimator = SVC(random_state=RANDOM_STATE, kernel='rbf').fit(X, y)
	# estimator = KNeighborsClassifier(n_neighbors=20).fit(X, y)
	plot_learning_curve(estimator_old, estimator_new, title, X, y, df_X, y, cv=cv,n_jobs=4)

	# plt.plot(estimator.loss_curve_)
	plt.savefig('MLP_PerfComparison_KMeans')
	# plt.show()

def main():
	pData = pd.read_pickle('EEGEyeData.pkl')
	pData['Class'] = pData['Class'].astype('category').cat.codes
	X_pData = pData.iloc[:,:-1]
	# X_pData = X_pData.replace('n',0)
	# X_pData = X_pData.replace('y',1)
	normalized_X_pData=(X_pData-X_pData.min())/(X_pData.max()-X_pData.min())
	# X_pData = X_pData.replace('b',0)
	# X_pData = X_pData.replace('o',1)
	# X_pData = X_pData.replace('x',2)
	y_pData = pData.iloc[:,-1]
	X_train_pData, X_test_pData, y_train_pData, y_test_pData = train_test_split(normalized_X_pData, y_pData, test_size=0.3, random_state=RANDOM_STATE)
	
	# wcss(X_pData)
	# sil_score(X_train_pData)
	# plot_sil(X_pData)
	# do_gm(X_pData,y_pData)
	# do_kmeans(X_pData,y_pData)
	# intra_cluster(X_pData)
	# inter_cluster(X_pData)
	# do_PCA(X_pData, 5)
	# dimred = do_sfs(X_pData, y_pData)
	# dimred = do_ICA(X_pData,4)
	# dimred_kmeans(X_pData, y_pData, dimred)
	# dimred_wcss(X_pData, y_pData, dimred)
	# dimred_sil(X_pData, y_pData, dimred)
	# dimred_kmeans(X_pData, y_pData, dimred)
	# dimred_gm(X_pData, y_pData, dimred)
	# dimred = do_ICA(X_pData,4)
	# # dimred = do_sfs(X_eData, y_eData)
	# dimred_wcss(X_pData, y_pData, dimred)
	# dimred_sil(X_pData, y_pData, dimred)
	# sil_score(X_pData, dimred)
	# dimred_nn(X_train_pData, y_train_pData, X_test_pData, y_test_pData, dimred)
	# plot_kmeans_cluster(X_pData, y_pData, dimred)


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
	
	#do_PCA_kmeans(X_train_eData)
	# wcss(X_eData)
	# do_kmeans(X_eData,y_eData)
	# do_gm(X_eData, y_eData)
	# sil_score(X_eData)
	# plot_sil(X_eData)
	# display_metrics("Logistic Regression",logistic_regression_pred(X_train,Y_train),Y_train)
	# display_metrics("SVM",svm_pred(X_train,Y_train),Y_train)
	# display_metrics("Decision Tree",decisionTree_pred(X_train,Y_train),Y_train)
	# intra_cluster(X_eData)
	# inter_cluster(X_eData)
	# do_PCA(X_eData)
	# dimred_kmeans(X_eData)
	# dimred_sil(X_eData)
	# dimred = do_randproj(X_eData, 4)
	# dimred = do_sfs(X_eData,y_eData)
	# dimred = do_sfs(X_eData, y_eData)
	# dimred_wcss(X_eData, y_eData, dimred)
	# dimred_sil(X_eData, y_eData, dimred)
	# dimred_kmeans(X_eData, y_eData, dimred)
	# sil_score(X_eData, dimred)	
	# dimred_nn(X_train_eData, y_train_eData, X_test_eData, y_test_eData, dimred)
	# km = KMeans(n_clusters=3, init='k-means++', random_state=RANDOM_STATE)
	# gm_full = GaussianMixture(n_components=3, random_state=RANDOM_STATE, covariance_type='full', warm_start = True)
	# gm_sphere = GaussianMixture(n_components=3, random_state=RANDOM_STATE, covariance_type='spherical', warm_start = True)
	# gm_diag = GaussianMixture(n_components=3, random_state=RANDOM_STATE, covariance_type='diag', warm_start = True)
	# gm_tied = GaussianMixture(n_components=3, random_state=RANDOM_STATE, covariance_type='tied', warm_start = True)	
	# gen_learning_curve_cluster(X_eData, y_eData, km)
	# gen_learning_curve(X_eData, y_eData, dimred)
	# plot_kmeans_cluster(X_eData, y_eData, dimred)

if __name__ == "__main__":
	main()
	
