# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 11:26:29 2016

@author: Juan
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import preprocessing

import sys
#import matlab.engine
import extract_features
import mat2npy_lpc_features
import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, int(100*cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
#    plt.show()

if __name__=='__main__':
    
    winlen = 1024   
    hop = 512    
    
    artist = sys.argv[1]
    feature_name = sys.argv[2]

#    artist = 'pablo'
#    feature_name = 'MFCC'
    
    if feature_name == 'LPC':

        p = int(sys.argv[3])

        if len(sys.argv)>4:
#            print 'Running Matlab script:'            
            print 'Computing LPC Features with ' + str(p) + ' poles model...'
#            eng = matlab.engine.start_matlab()
#            result = eng.extract_lpc_features(p, winlen, hop)  
#            if result == 1:
            print 'Converting from .mat to .npy...' 
            mat2npy_lpc_features.convert('../features/ulla_lpc_' + str(p), 'ulla')
            mat2npy_lpc_features.convert('../features/pablo_lpc_' + str(p), 'pablo')
            mat2npy_lpc_features.convert('../features/emma_lpc_' + str(p), 'emma')
            mat2npy_lpc_features.convert('../features/juan_lpc_' + str(p), 'juan') 
            mat2npy_lpc_features.convert('../features/claire_lpc_' + str(p), 'claire')               
                
        
        features_train="../features/" + artist + "_lpc_" + str(p) + "_train.npy"
        features_test="../features/" + artist + "_lpc_" + str(p) + "_test.npy"
        
        test=np.load(features_test)
        train=np.load(features_train)
        
        X=train[0:p,:].T
        y=train[p,:]
        
        X_test=test[0:p,:].T
        y_test=test[p,:]
          
    if feature_name == 'MFCCVOICING':
        
        melcoeff = int(sys.argv[3])
        melbands = int(sys.argv[4])
        
#        melcoeff = 20
#        melbands = 40
            
        if len(sys.argv)>5:
#        if True:
            print 'Computing Voicing + MFCC Features with: ' + str(melcoeff) + ' Mel-Coefficients...'
            extract_features.extract_mfcc(melcoeff, melbands, winlen=1024, hop=512, emb_number = '3', voicing = True)
        
        features_train="../features/" + artist + "_mfccvoicing_" + str(melcoeff) + str(melbands) + "_train.npy"
        features_test="../features/" + artist + "_mfccvoicing_" + str(melcoeff) + str(melbands) + "_test.npy"
        
        test=np.load(features_test)
        train=np.load(features_train)

        #con voicing
        melcoeff=melcoeff+1        
        
        X=train[0:melcoeff,:].T
        y=train[melcoeff,:]
        time_train=train[melcoeff+1,:]
        
        X_test=test[0:melcoeff,:].T
        y_test=test[melcoeff,:]
        time_test=test[melcoeff+1,:]
        
        melcoeff = melcoeff-1
        
        prediction_file="../prediction/" + artist + "_mfccvoicing_" + str(melcoeff) + str(melbands) + "_prediction.npy"    
        prediction_csv="../prediction/" + artist + "_mfccvoicing_" + str(melcoeff) + str(melbands) + "_prediction.csv" 
        proba_csv="../prediction/" + artist + "_mfccvoicing_" + str(melcoeff) + str(melbands) + "_proba.csv" 
        
    if feature_name == 'MFCC':
        
        melcoeff = int(sys.argv[3])
        melbands = int(sys.argv[4])
        
#        melcoeff = 20
#        melbands = 40
            
        if len(sys.argv)>5:
#        if True:
            print 'Computing MFCC Features with: ' + str(melcoeff) + ' Mel-Coefficients...'
            extract_features.extract_mfcc(melcoeff, melbands, winlen, hop, emb_number='3')
        
        features_train="../features/" + artist + "_mfcc_" + str(melcoeff) + str(melbands) + "_train.npy"
        features_test="../features/" + artist + "_mfcc_" + str(melcoeff) + str(melbands) + "_test.npy"
        
        test=np.load(features_test)
        train=np.load(features_train)
        
        X=train[0:melcoeff,:].T
        y=train[melcoeff,:]
        time_train=train[melcoeff+1,:]
        
        X_test=test[0:melcoeff,:].T
        y_test=test[melcoeff,:]
        time_test=test[melcoeff+1,:]
        
        prediction_file="../prediction/" + artist + "_mfcc_" + str(melcoeff) + str(melbands) + "_prediction.npy"    
        prediction_csv="../prediction/" + artist + "_mfcc_" + str(melcoeff) + str(melbands) + "_prediction.csv" 
        proba_csv="../prediction/" + artist + "_mfcc_" + str(melcoeff) + str(melbands) + "_proba.csv"
        proba_file="../prediction/" + artist + "_mfcc_" + str(melcoeff) + str(melbands) + "_proba.npy"
       
    if feature_name == 'SPECTRALCONTRAST':
        
        nbands = int(sys.argv[3])
        quantile = 0.02
            
        if len(sys.argv)>4:  
            print 'Computing Spectral Contrast Features with: ' + str(nbands) + ' analysis bands...'
            extract_features.extract_spectral_contrast(nbands, quantile, emb_number='3')
        
        features_train="../features/" + artist + "_spectral_contrast_" + str(nbands) + "_train.npy"
        features_test="../features/" + artist + "_spectral_contrast_" + str(nbands) + "_test.npy"
        
        test=np.load(features_test)
        train=np.load(features_train)
        
        X=train[0:nbands+1,:].T
        y=train[nbands+1,:]
        time_train=train[nbands+2,:]
        
        X_test=test[0:nbands+1,:].T
        y_test=test[nbands+1,:]
        time_test=test[nbands+2,:]
        
        prediction_file="../prediction/" + artist + "_spectral_contrast_" + str(nbands) + "_prediction.npy"    
        prediction_csv="../prediction/" + artist + "_spectral_contrast_" + str(nbands) + "_prediction.csv" 
        proba_csv="../prediction/" + artist + "_spectral_contrast_" + str(nbands) + "_proba.csv"
        proba_file="../prediction/" + artist + "_spectral_contrast_" + str(nbands) + "_proba.npy"

    if feature_name == 'SPECTRAL':
        
        winlen = int(sys.argv[3])
        hop = int(sys.argv[4])
            
        if len(sys.argv)>5:  
            print 'Computing some classical Spectral Features...'
            extract_features.extract_spectral(winlen, hop)
        
        features_train="../features/" + artist + "_spectral_" + str(winlen) + str(hop) + "_train.npy"
        features_test="../features/" + artist + "_spectral_" + str(winlen) + str(hop) + "_test.npy"
        
        test=np.load(features_test)
        train=np.load(features_train)
        
        X=train[0:5,:].T
        y=train[5,:]
        time_train=train[6,:]
        
        X_test=test[0:5,:].T
        y_test=test[5,:]
        time_test=test[6,:]
        
        prediction_file="../prediction/" + artist + "_spectral_" + str(winlen) + str(hop) + "_prediction.npy"    
        prediction_csv="../prediction/" + artist + "_spectral_" + str(winlen) + str(hop) + "_prediction.csv" 
        proba_csv="../prediction/" + artist + "_spectral_" + str(winlen) + str(hop) + "_proba.csv"

    
    #%% CLASSIFY
    
    print 'Classification performance:'   
    
#    print X.shape
#    print y.shape
    
    #CLASSIFIER PARAMETER
    n_neighbors=10
    n_estimators=10
    
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    
    scaler2 = preprocessing.StandardScaler().fit(X_test)
    X_test = scaler2.transform(X_test)
    
    print X.shape
    
    knn = neighbors.KNeighborsClassifier(n_neighbors)
    knn.fit(X,y)
    knn_score = knn.score(X_test,y_test)
    print 'KNN: ' + str(knn_score)
    
    rf = RandomForestClassifier(n_estimators)
    rf = rf.fit(X,y)
    rf_score = rf.score(X_test,y_test)
    print 'RandomForest: ' + str(rf_score)    
     
    lin_svm = svm.LinearSVC()
    lin_svm.fit(X, y)
    svm_score = lin_svm.score(X_test, y_test) 
    print 'SVM (Linear Kernel): ' + str(svm_score)

    X_prediction = knn.predict(X_test)
    X_prediction_proba = knn.predict_proba(X_test)
    #%%
    
    time_test=time_test.reshape((len(time_test),1))
    y_test_aux=y_test.reshape((len(time_test),1))
    X_prediction_proba = np.concatenate((X_prediction_proba, time_test, y_test_aux),axis=1)

    if feature_name=='MFCC':    
        print 'Saving prediction vector with KNN...'        
        aux_vec = np.c_[(X_prediction, y_test, time_test)]
        np.save(prediction_file, aux_vec)
        np.savetxt(prediction_csv, aux_vec)
        np.save(proba_file, X_prediction_proba)
        np.savetxt(proba_csv, X_prediction_proba)        
        
    
    print 'Generating confusion matrix ...'
    from sklearn.metrics import confusion_matrix
    cnf_matrix = confusion_matrix(y_test,X_prediction)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['BHC','Breathy','Normal'],
                          title='Confusion matrix, without normalization')