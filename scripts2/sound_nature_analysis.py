# -*- coding: utf-8 -*-
"""
Created on Sun Dec 04 20:38:59 2016

@author: Juan
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy import signal
import librosa
import voicing_extraction


if __name__=='__main__':
        
    winlen = 1024
    hop = 512   
    overlap = winlen - hop
    nfft = 16*winlen

    artist = 'ulla'    
    audio_file = "../audio/" + artist + "_mono.wav"
    features_csv = "../audio/" + artist + "_spectral_" + str(winlen) + str(hop) + "_features.csv"
#    audio_file='../audio/LP-mem-6-a.wav'

    fs, x = wav.read(audio_file)
    t = np.arange(len(x)) * (1/fs) 

    x_rolloff = librosa.feature.spectral_rolloff(x, sr=fs, n_fft=winlen, hop_length=hop)

    x_centroid = librosa.feature.spectral_centroid(x, sr=fs, n_fft=winlen, hop_length=hop)      

    x_bandwidth = librosa.feature.spectral_bandwidth(x, sr=fs, n_fft=winlen, hop_length=hop)        

    x_zcr = librosa.feature.zero_crossing_rate(x, frame_length=winlen, hop_length=hop)        
    
    x_voicing, dummy = voicing_extraction.voicing(x, winlen, hop, num_lags=250)
    x_voicing = np.append(x_voicing,np.zeros((x_zcr.shape[1]-x_voicing.shape[1],)))
    x_voicing = x_voicing.reshape((1,len(x_voicing)))    
    #%%
    t_feature = np.arange(x_rolloff.shape[1]) * (float(hop)/fs)
    t_feature = t_feature.reshape((1,len(t_feature)))   

#    f, t_S, Sxx = signal.spectrogram(x, fs, window='hamming', nperseg=winlen, 
#                                 noverlap=overlap, nfft=nfft, detrend='constant',
#                                 return_onesided=True, scaling='spectrum', axis=-1)
#    plt.figure(), plt.pcolormesh(t_S, f, 20*np.log10(abs(Sxx))),
#    plt.plot(t_voicing,fs/2*voicing_feature[0,:],'r')
    
    aux_4saving = np.concatenate((x_rolloff, x_centroid, x_bandwidth, x_zcr, x_voicing, t_feature))
    np.savetxt(features_csv, aux_4saving.T)