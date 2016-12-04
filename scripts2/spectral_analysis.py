# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 19:37:17 2016

@author: Juan
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy import signal

melcoeff = 20
melbands = 40
winlen=1024
nfft=2048
overlap=512

artist = 'pablo'

prediction_file="../prediction/" + artist + "_mfcc_" + str(melcoeff) + str(melbands) + "_prediction.npy"  
test=np.load(prediction_file)

t_extraction = np.sort(test[:,2])
emb_extraction = test[np.argsort(test[:,2]),0]

#%%

audio_file = "../audio/" + artist + "_mono.wav"

fs, x = wav.read(audio_file)
t = np.arange(len(x)) * (1/fs) 
    

f, t_S, Sxx = signal.spectrogram(x, fs, window='hamming', nperseg=winlen, 
                                 noverlap=overlap, nfft=nfft, detrend='constant',
                                 return_onesided=True, scaling='spectrum', axis=-1)
                                 

fragment = [15000, 17000]
emb_extraction_medianfiltered = signal.medfilt(emb_extraction,5)


plt.figure(figsize=(18,6))
plt.pcolormesh(t_S[fragment[0]:fragment[1]], f, 20*np.log(Sxx[:,fragment[0]:fragment[1]]))
plt.plot(t_extraction[(t_extraction >= t_S[fragment[0]]) & (t_extraction <= t_S[fragment[1]])],6000*emb_extraction[(t_extraction >= t_S[fragment[0]]) & (t_extraction <= t_S[fragment[1]])])
plt.plot(t_extraction[(t_extraction >= t_S[fragment[0]]) & (t_extraction <= t_S[fragment[1]])],6000*emb_extraction_medianfiltered[(t_extraction >= t_S[fragment[0]]) & (t_extraction <= t_S[fragment[1]])])
plt.xlabel('Time (s)')
plt.ylabel('Notes')
#plt.xlim(xmax = test[fragment[1],2], xmin=test[fragment[0],2])
#plt.ylim(ymax = 4 , ymin = 0)
plt.axis('tight')
plt.show()

#%%

plt.figure(), plt.plot(f, 20*np.log10(Sxx[:,3364]), 'g'),
plt.plot(f, 20*np.log10(Sxx[:,19304]), 'y'), 
plt.plot(f, 20*np.log10(Sxx[:,12517]), 'r'),
plt.grid(),plt.axis('tight') 