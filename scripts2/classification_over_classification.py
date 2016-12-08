# -*- coding: utf-8 -*-
"""
Created on Wed Dec 07 21:15:55 2016

@author: Juan
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors



melcoeff=20
melbands=40

probabilties_ulla="../prediction/" + 'ulla' + "_mfcc_" + str(melcoeff) + str(melbands) + "_proba.npy"
proba_ulla=np.load(probabilties_ulla)

probabilties_pablo="../prediction/" + 'pablo' + "_mfcc_" + str(melcoeff) + str(melbands) + "_proba.npy"
proba_pablo=np.load(probabilties_pablo)

probabilties_emma="../prediction/" + 'emma' + "_mfcc_" + str(melcoeff) + str(melbands) + "_proba.npy"
proba_emma=np.load(probabilties_emma)

probabilties_juan="../prediction/" + 'juan' + "_mfcc_" + str(melcoeff) + str(melbands) + "_proba.npy"
proba_juan=np.load(probabilties_juan)

probabilties_claire="../prediction/" + 'claire' + "_mfcc_" + str(melcoeff) + str(melbands) + "_proba.npy"
proba_claire=np.load(probabilties_claire)

#%%
artist = 'ulla'

#CLASSIFIER PARAMETER
n_estimators=10
k_neighbors=10

X = np.concatenate((proba_emma[:,0:3],proba_pablo[:,0:3],proba_juan[:,0:3],proba_claire[:,0:3]), axis=0)
y = np.concatenate((proba_emma[:,4],proba_pablo[:,4],proba_juan[:,4],proba_claire[:,4]), axis=0)
X_test = proba_ulla[:,0:3]
y_test = proba_ulla[:,4]

rf = RandomForestClassifier(n_estimators)
rf = rf.fit(X,y)
rf_score = rf.score(X_test,y_test)
print 'RandomForest: ' + str(rf_score) 

knn = neighbors.KNeighborsClassifier(n_neighbors)
knn.fit(X,y)
knn_score = knn.score(X_test,y_test)
print 'KNN: ' + str(knn_score)

#CLASSIFIER PARAMETER
n_estimators=10
#%%
X = np.concatenate((proba_ulla[:,0:3],proba_pablo[:,0:3],proba_juan[:,0:3],proba_claire[:,0:3]), axis=0)
y = np.concatenate((proba_ulla[:,4],proba_pablo[:,4],proba_juan[:,4],proba_claire[:,4]), axis=0)
X_test = proba_emma[:,0:3]
y_test = proba_emma[:,4]

rf = RandomForestClassifier(n_estimators)
rf = rf.fit(X,y)
rf_score = rf.score(X_test,y_test)
print 'RandomForest: ' + str(rf_score) 

knn = neighbors.KNeighborsClassifier(n_neighbors)
knn.fit(X,y)
knn_score = knn.score(X_test,y_test)
print 'KNN: ' + str(knn_score)

#%%
X = np.concatenate((proba_ulla[:,0:3],proba_emma[:,0:3],proba_juan[:,0:3],proba_claire[:,0:3]), axis=0)
y = np.concatenate((proba_ulla[:,4],proba_emma[:,4],proba_juan[:,4],proba_claire[:,4]), axis=0)
X_test = proba_pablo[:,0:3]
y_test = proba_pablo[:,4]

rf = RandomForestClassifier(n_estimators)
rf = rf.fit(X,y)
rf_score = rf.score(X_test,y_test)
print 'RandomForest: ' + str(rf_score) 

knn = neighbors.KNeighborsClassifier(n_neighbors)
knn.fit(X,y)
knn_score = knn.score(X_test,y_test)
print 'KNN: ' + str(knn_score)