# -*- coding: utf-8 -*-
"""
Created on Tue Dec 06 21:45:09 2016

@author: Juan
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy import signal



if __name__=='__main__':
        
    melcoeff = 20
    melbands = 40   

    artist = 'claire'    
    audio_file = "../audio/" + artist + "_mono.wav"
    proba_csv = "../prediction/" + artist + "_mfcc_" + str(melcoeff) + str(melbands) + "_proba.csv"
    prediction_csv= "../prediction/" + artist + "_mfcc_" + str(melcoeff) + str(melbands) + "_prediction.csv"
#    audio_file='../audio/LP-mem-6-a.wav'

    fs, x = wav.read(audio_file)
    t = np.arange(len(x)) * (1/fs) 
    
    proba = np.loadtxt(proba_csv)
    prediction = np.loadtxt(prediction_csv)
    
    from PIL import Image
    import numpy as np
    
    aux=prediction[np.argsort(prediction[:,2]),1]
    r_gt=np.zeros(aux.shape,dtype='uint8')  
    g_gt=np.zeros(aux.shape,dtype='uint8')
    b_gt=np.zeros(aux.shape,dtype='uint8')
    r_gt[aux==1]=1
    g_gt[aux==2]=1
    b_gt[aux==3]=1
    
    for i in range(0,9):
        r_gt=np.c_[r_gt,r_gt]
        g_gt=np.c_[g_gt,g_gt]
        b_gt=np.c_[b_gt,b_gt]
        
    rgbArray_gt = np.zeros((r_gt.shape[1],r_gt.shape[0],3), 'uint8') 
    rgbArray_gt[..., 0] = r_gt.T*255
    rgbArray_gt[..., 1] = g_gt.T*255
    rgbArray_gt[..., 2] = b_gt.T*255
    
    
    r=np.c_[proba[np.argsort(proba[:,3]),0],proba[np.argsort(proba[:,3]),0]]
    g=np.c_[proba[np.argsort(proba[:,3]),1],proba[np.argsort(proba[:,3]),1]]
    b=np.c_[proba[np.argsort(proba[:,3]),2],proba[np.argsort(proba[:,3]),2]]
    for i in range(0,8):
        r=np.c_[r,r]
        g=np.c_[g,g]
        b=np.c_[b,b]

    rgbArray = np.zeros((r.shape[1],r.shape[0],3), 'uint8')  
  
    rgbArray[..., 0] = r.T*255
    rgbArray[..., 1] = g.T*255
    rgbArray[..., 2] = b.T*255
        
    rgbArray_mix=np.zeros((2*r.shape[1],r.shape[0],3), 'uint8')  
    
    rgbArray_mix[0:512,:,:]=rgbArray
    rgbArray_mix[512:1024,:,:]=rgbArray_gt
    img_mix = Image.fromarray(rgbArray_mix)
    img_mix.save(artist+'_mix.jpeg')