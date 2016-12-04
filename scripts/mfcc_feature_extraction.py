# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 23:47:43 2016

@author: Juan
"""

import csv
import numpy as np
import scipy.io.wavfile as wav
import scipy.io as io
import librosa
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_gt(gt_file, t, fs = 44100):
    
    cr = csv.reader(open(gt_file,"rb"))
    onset=[]
    labels=[]
    for row in cr:
        onset.append(row[0]) 
        labels.append(row[1])
    onset = np.array(onset, 'float32')
    
    aux_gt = np.empty([0,], 'int8')
    for label in labels:
#        print label
        if label=='1':
            aux_gt = np.r_[aux_gt,1]
        elif label=='2':
            aux_gt = np.r_[aux_gt,2]
        elif label=='3':
            aux_gt = np.r_[aux_gt,3]  
        elif label=='4':
            aux_gt = np.r_[aux_gt,4]
        elif label=='0':
            aux_gt = np.r_[aux_gt,0] 
        elif label=='-1':
            aux_gt = np.r_[aux_gt,-1]   
        elif label=='-2':
            aux_gt = np.r_[aux_gt,-2] 
    j=0
    gt = np.empty([len(t),], 'float64')
    for i in range(1,len(onset)):
        while (j<len(t) and t[j]>=onset[i-1] and t[j]<=onset[i]):
            gt[j]=float(aux_gt[i-1])
            j=j+1
    
    return gt, aux_gt


if __name__=='__main__':
    
    winlen = 1024;
    hop = 512;
    overlap = winlen - hop;

    audio_file = '../audio/UllaSuokko_mono.wav'
    fs, audio = wav.read(audio_file)
    t = np.arange(len(audio)) * (1/fs)   

    mfcc = librosa.feature.mfcc(audio, sr=fs, n_mfcc=40, n_fft=winlen, hop_length=hop)
    t_mfcc = np.arange(mfcc.shape[1])*(float(hop)/fs)+float(winlen)/(2*fs);

    gt_file = '../audio/ulla_embrochure.csv'
    gt, aux_gt = load_gt(gt_file, t_mfcc);
    gt[len(gt)-1] = 0
    
    mfcc = mfcc[:, gt==1 or gt==2 or gt==3]
    gt = gt[gt==1 or gt==2 or gt==3]    
    
#%%
    features = np.vstack([mfcc, gt, t_mfcc])
    io.savemat('ulla_mfcc_features.mat', mdict={'mfcc':mfcc, 'gt':gt, 't_mfcc':t_mfcc})
    