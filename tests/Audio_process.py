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

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import time

def loadFiles(directory):
	current = 0
	for file in os.listdir(directory):
		print("Processing file : " + file)
		data = pd.read_csv(os.path.join(directory, file), sep=",")
		dataArray = np.array(data.values)
		features = dataArray[:,1:-1]
		labels = dataArray[:,-1]
		if(current != 0):
			X = np.concatenate((X,features))
			Y = np.concatenate((Y,labels))
			#if(current == 6):
			#	break
		else:
			first = False
			X = features
			Y = labels
		current+=1
		print("Added : " , dataArray.shape , ", Total : (" , X.shape , "," , Y.shape , ")")
	return X,Y

def PCAAndScaler(X, n):
	scaler = StandardScaler()
	XReduit = scaler.fit_transform(X) # On calcul la version centrée réduite de X
	pca = PCA(n_components=n)
	X_pca = pca.fit_transform(XReduit) # On transforme par l'ACP
	ratio = pca.explained_variance_ratio_
	print(ratio)
	return X_pca

def run_classifiers(clfs,X,Y, precision):
    kf10 = KFold(n_splits=10, shuffle=True, random_state=0)  # Calcul du score avec 10-folds
    kf5 = KFold(n_splits=5, shuffle=True, random_state=0)   # Calcul du score avec 5-folds
    for i in clfs:
        time_init = time.time()  # On démarre le timer
        clf = clfs[i] #clf correspond au ième algorithme du dictionnaire clfs.
        cv_acc = cross_val_score(clf, X, Y, cv=kf10, scoring='accuracy') #pour le calcul de l’accuracy
        cv_auc = cross_val_score(clf, X, Y, cv=kf10, scoring='roc_auc') #pour le calcul de l’AUC de la courbe ROC
        
        if precision:  # On choisis en argument rappel ou precision
            cv_pre = cross_val_score(clf, X, Y, cv=kf5, scoring='precision') #pour le calcul de la précision   
        else:
            cv_rap = cross_val_score(clf, X, Y, cv=kf5, scoring='recall') #pour le calcul de le rappel
        
        
        time_algo = time.time()-time_init  # On arrete le timer
    
        print("Accuracy for {0} is: {1:.3f} +/- {2:.3f}".format(i, np.mean(cv_acc), np.std(cv_acc)))
        print("ROC_AUC for {0} is: {1:.3f} +/- {2:.3f}".format(i, np.mean(cv_auc), np.std(cv_auc)))
        
        if precision:
            print("Precision for {0} is: {1:.3f} +/- {2:.3f}".format(i, np.mean(cv_pre), np.std(cv_pre)))
        else:
            print("Recall for {0} is: {1:.3f} +/- {2:.3f}".format(i, np.mean(cv_rap), np.std(cv_rap)))
        print("Time for {0} is: {1:.3f}s \n".format(i, time_algo))
		
DevFeatures, DevLabels = loadFiles("./DevSetComplete/complete/")
TestFeatures, TestLabels = loadFiles("./TestSetComplete/complete/")

DevFeatures = np.nan_to_num(DevFeatures)
TestFeatures = np.nan_to_num(TestFeatures)

DevFeatures = DevFeatures.astype(float)
TestFeatures = TestFeatures.astype(float)
DevLabels = DevLabels.astype(float)
TestLabels = TestLabels.astype(float)

DevFeatures = PCAAndScaler(DevFeatures,5)
TestFeatures = PCAAndScaler(TestFeatures,5)

clfs = {   # Définition des classifieurs testés
'RF': RandomForestClassifier(n_estimators=50),
'KNN': KNeighborsClassifier(n_neighbors=5),
'GNB': GaussianNB(),
'AB': AdaBoostClassifier(n_estimators=50),
'BAG': BaggingClassifier(n_estimators=50),
'MLP': MLPClassifier(hidden_layer_sizes=(20,10)),
'CART': DecisionTreeClassifier(criterion="gini"),
'ID3': DecisionTreeClassifier(criterion="entropy")
}

run_classifiers(clfs,DevFeatures,DevLabels, False)









