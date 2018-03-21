#!/usr/bin/python3.6

import os
import json
import sys
import numpy as np
np.set_printoptions(threshold=np.nan)
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import time

def PCAAndScaler(X, n):
	scaler = StandardScaler()
	XReduit = scaler.fit_transform(X) # On calcul la version centrée réduite de X
	pca = PCA(n_components=n)
	X_pca = pca.fit_transform(XReduit) # On transforme par l'ACP
	ratio = pca.explained_variance_ratio_
	print(ratio)
	return X_pca, pca
	
def ScaleAndApplyPCA(X, pca):
	scaler = StandardScaler()
	XReduit = scaler.fit_transform(X) # On calcul la version centrée réduite de X
	X_pca = pca.transform(XReduit) # On transforme par l'ACP
	return X_pca

def run_classifiers_app_test(clfs,Xapp,Xtest,Yapp, Ytest, precision):
    for i in clfs:
        time_init = time.time()  # On démarre le timer
        clf = clfs[i] #clf correspond au ième algorithme du dictionnaire clfs.
        
        clf.fit(Xapp, Yapp)
        cv_acc = clf.score(Xtest, Ytest)
        
        Ypred = clf.predict(Xtest)
        cv_rec = metrics.recall_score(Ytest, Ypred, average=None)
        cv_pre = metrics.precision_score(Ytest, Ypred, average=None)
        cv_rocauc = metrics.roc_auc_score(Ytest, Ypred)
        matrix = metrics.confusion_matrix(Ytest, Ypred)
        tn, fp, fn, tp = matrix.ravel()
        
        time_algo = time.time()-time_init  # On arrete le timer
        print("Classifier : {0}".format(i))
        print("    Accuracy is: {0:.3f}".format(cv_acc))
        print("    Recall is: {0:.3f}, {1:.3f}".format(cv_rec[0],cv_rec[1]))
        print("    Precision is: {0:.3f}, {1:.3f}".format(cv_pre[0],cv_pre[1]))
        print("    ROC AUC is: {0:.3f}".format(cv_rocauc))
        print("    Confusion matrix is:")
        print(matrix)
        print("    tn:{0}, fp:{1}, fn:{2}, tp:{3}".format(tn, fp, fn, tp ))
        print("    Time: {0:.3f}s \n".format(time_algo))

DevFeatures = np.load("dev_features_pca.csv.npy")
DevLabels = np.load("dev_labels_pca.csv.npy")
DevNamedIndices = np.load("dev_named_indices_pca.csv.npy")
TestFeatures = np.load("test_features_pca.csv.npy")
TestLabels = np.load("test_labels_pca.csv.npy")
TestNamedIndices = np.load("test_named_indices_pca.csv.npy")

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

run_classifiers_app_test(clfs,DevFeatures,TestFeatures,DevLabels,TestLabels, False)








