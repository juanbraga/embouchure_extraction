# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 19:59:26 2016

@author: Juan
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy import signal


def voicing(x, winlen=1024, hop=512, fs=44100, num_lags=250, kappa=0.15):

#    voicing_len=np.floor((len(x)-(winlen+num_lags))/hop)
    
    voicing_len=int(np.floor((len(x)-(winlen+num_lags))/hop))
    
    t_voicing=(np.arange(0,voicing_len)*(float(hop)/fs))+((winlen+num_lags)/float(2*fs))
    
    slices=np.zeros((winlen+num_lags,voicing_len))
    d=np.zeros((num_lags,voicing_len))
    d_prime=np.zeros((num_lags,voicing_len))
    
    for i in range(0,voicing_len):
#        print i
        slices[:,i]=x[i*hop:i*hop+(winlen+num_lags)]
        for j in range(0,num_lags):
            if (j==0):
                d_prime[0,i]=1;
            else:
                d[j,i] = np.sum(np.square(slices[0:winlen,i]-slices[j:winlen+j,i]));
                d_prime[j,i] = d[j,i]*float(j)/sum(d[1:j+1,i]);
                
    
    d_prime_threshold = np.zeros(d_prime.shape)
    np.copyto(d_prime_threshold,d_prime)
    d_prime_threshold[d_prime>kappa]=0
    
#    plt.figure(), plt.pcolormesh(t_voicing, np.arange(0,num_lags), d_prime)
    
    voicing_feature=np.ones((1,voicing_len))
    locs=np.zeros((1,voicing_len))
    
    for i in range(0,voicing_len):
        locs_aux = signal.find_peaks_cwt(d_prime_threshold[:,i],np.arange(1,10))                
        if (not locs_aux):
            locs_aux = np.argmin(d_prime[:,i])
            locs[0,i]=locs_aux
#            print locs_aux
#            print voicing_feature[0,i]
            voicing_feature[0,i]=d_prime[locs_aux,i]
        else:
            locs[0,i]=locs_aux[0]
            voicing_feature[0,i]=d_prime[locs_aux[0],i]

#    plt.figure(), plt.pcolormesh(t_voicing, np.arange(0,num_lags), d_prime_threshold),
#    plt.plot(t_voicing,locs[0,:],'w')
           
    voicing_feature=1-voicing_feature
    voicing_feature=np.nan_to_num(voicing_feature)
    
#    plt.figure(), plt.plot(t_voicing,locs[0,:]/max(locs[0,:]),'r'), plt.plot(t_voicing,voicing_feature[0,:],'b')
    
    return voicing_feature, t_voicing
    
    
if __name__=='__main__':

    artist = 'ulla'    
    audio_file = "../audio/" + artist + "_mono.wav"
    voicing_csv = "../audio/" + artist + "_voicing.csv"
#    audio_file='../audio/LP-mem-6-a.wav'

    fs, x = wav.read(audio_file)
    t = np.arange(len(x)) * (1/fs) 
    winlen = 1024
    hop = 512
    overlap = winlen - hop
    nfft = 16*winlen
    
    voicing_feature, t_voicing = voicing(x, winlen, hop, fs, num_lags=250)
    

#    f, t_S, Sxx = signal.spectrogram(x, fs, window='hamming', nperseg=winlen, 
#                                 noverlap=overlap, nfft=nfft, detrend='constant',
#                                 return_onesided=True, scaling='spectrum', axis=-1)
    
    #%%
#    plt.figure(), plt.pcolormesh(t_S, f, 20*np.log10(abs(Sxx))),
#    plt.plot(t_voicing,fs/2*voicing_feature[0,:],'r')
    
    t_voicing = t_voicing.reshape(voicing_feature.shape)
    aux_4saving = np.concatenate((voicing_feature, t_voicing))
    np.savetxt(voicing_csv, aux_4saving.T)