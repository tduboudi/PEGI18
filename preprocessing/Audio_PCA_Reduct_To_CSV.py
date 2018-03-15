#!/usr/bin/python3.6

import os
import json
import sys
import numpy as np
np.set_printoptions(threshold=np.nan)
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def loadFiles(directory):
	current = 0
	names = []
	indices = []
	namedIndices = []
	prevStart = 0
	for file in os.listdir(directory):
		print("Processing file : " + file)
		data = pd.read_csv(os.path.join(directory, file), sep=",")
		dataArray = np.array(data.values)
		features = dataArray[:,1:-1]
		labels = dataArray[:,-1]
		if(current != 0):
			prevStart = np.size(Y)
			X = np.concatenate((X,features))
			Y = np.concatenate((Y,labels))
			#if(current == 6):
			#	break
		else:
			first = False
			X = features
			Y = labels
		current+=1
		print(file[0:-4])
		names.append(file[0:-4])
		indices.append([prevStart,np.size(Y)])
		namedIndices.append([file[0:-4], prevStart, np.size(Y)] )	
		print("Added : " , dataArray.shape , ", Total : (" , X.shape , "," , Y.shape , ")")
		Indices = np.array(namedIndices)
	return (X,Y,Indices)

def PCAAndScaler(X, n):
	scaler = StandardScaler()
	XReduit = scaler.fit_transform(X) # On calcul la version centrée réduite de X
	pca = PCA(n_components=n)
	X_pca = pca.fit_transform(XReduit) # On transforme par l'ACP
	ratio = pca.explained_variance_ratio_
	print(ratio)
	return X_pca

DevFeatures, DevLabels, DevNamedIndices = loadFiles("./DevSetComplete/complete/")
TestFeatures, TestLabels, TestNamedIndices  = loadFiles("./TestSetComplete/complete/")

DevFeatures = np.nan_to_num(DevFeatures)
TestFeatures = np.nan_to_num(TestFeatures)

DevFeatures = DevFeatures.astype(float)
TestFeatures = TestFeatures.astype(float)
DevLabels = DevLabels.astype(float)
TestLabels = TestLabels.astype(float)

DevFeatures = PCAAndScaler(DevFeatures,10)
TestFeatures = PCAAndScaler(TestFeatures,10)

np.save("dev_features_pca.csv", DevFeatures)
np.save("dev_labels_pca.csv", DevLabels)
np.save("dev_named_indices_pca.csv", DevNamedIndices)
np.save("test_features_pca.csv", TestFeatures)
np.save("test_labels_pca.csv", TestLabels)
np.save("test_named_indices_pca.csv", TestNamedIndices)



