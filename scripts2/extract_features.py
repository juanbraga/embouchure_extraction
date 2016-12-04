    # -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 10:39:17 2016

@author: Juan
"""

import csv
import numpy as np
import scipy.io.wavfile as wav
import scipy.io as io
import librosa
import voicing_extraction 

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
    
    
def extract_spectral(winlen=1024, hop=512, emb_number = '3'):

#if __name__=='__main__':

    claire_audio_file='../audio/claire_mono.wav';    
    claire_gt_file='../audio/claire_embrochure.csv';   
    juan_audio_file='../audio/juan_mono.wav';    
    juan_gt_file='../audio/juan_embrochure.csv';    
    emma_audio_file = '../audio/emma_mono.wav'; 
    emma_gt_file = '../audio/emma_embrochure.csv';
    pablo_audio_file = '../audio/pablo_mono.wav'; 
    pablo_gt_file = '../audio/pablo_embrochure.csv';
    ulla_audio_file = '../audio/ulla_mono.wav'; 
    ulla_gt_file = '../audio/ulla_embrochure.csv';

    #PARAMETROS
#    winlen = 1024;
#    hop = 512;
#    overlap = winlen - hop
#    emb_number = '3'

    fs, claire = wav.read(claire_audio_file)
#    t_claire = np.arange(len(claire)) * (1/fs) 
    
    fs, juan = wav.read(juan_audio_file)
#    t_juan = np.arange(len(juan)) * (1/fs)  

    fs, emma = wav.read(emma_audio_file)
#    t_emma = np.arange(len(emma)) * (1/fs)  
    
    fs, pablo = wav.read(pablo_audio_file)
#    t_pablo = np.arange(len(pablo)) * (1/fs)
    
    fs, ulla = wav.read(ulla_audio_file)
#    t_ulla = np.arange(len(ulla)) * (1/fs)


#%% MFCC EXTRACTION

    if emb_number == '3':

        claire_rolloff = librosa.feature.spectral_rolloff(claire, sr=fs, n_fft=winlen, hop_length=hop)
        claire_centroid = librosa.feature.spectral_centroid(claire, sr=fs, n_fft=winlen, hop_length=hop)      
        claire_bandwidth = librosa.feature.spectral_bandwidth(claire, sr=fs, n_fft=winlen, hop_length=hop)        
        claire_zcr = librosa.feature.zero_crossing_rate(claire, frame_length=winlen, hop_length=hop)        
        claire_voicing, dummy = voicing_extraction.voicing(claire, winlen, hop, num_lags=250)
        claire_voicing=np.append(claire_voicing,np.zeros((claire_zcr.shape[1]-claire_voicing.shape[1],)))
        claire_voicing=claire_voicing.reshape((1,len(claire_voicing)))        
        claire_spectral = np.concatenate((claire_rolloff, claire_centroid, claire_bandwidth, claire_zcr, claire_voicing),axis=0)        
        claire_t_spectral = np.arange(claire_spectral.shape[1])*(float(hop)/fs)+float(winlen)/(2*fs);
        claire_gt, aux_gt = load_gt(claire_gt_file, claire_t_spectral);
        claire_gt[len(claire_gt)-1] = 0
        claire_spectral_aux = np.concatenate((claire_spectral[:, claire_gt==1], claire_spectral[:,claire_gt==2], claire_spectral[:,claire_gt==3]),axis=1)
        claire_gt_aux = np.concatenate((claire_gt[claire_gt==1], claire_gt[claire_gt==2], claire_gt[claire_gt==3]),axis=0)  
        claire_gt_aux=claire_gt_aux.reshape((1,len(claire_gt_aux)))
        claire_time_aux = np.concatenate((claire_t_spectral[claire_gt==1], claire_t_spectral[claire_gt==2], claire_t_spectral[claire_gt==3]),axis=0)  
        claire_time_aux=claire_time_aux.reshape((1,len(claire_time_aux)))
        
        #%%

        juan_rolloff = librosa.feature.spectral_rolloff(juan, sr=fs, n_fft=winlen, hop_length=hop)
        juan_centroid = librosa.feature.spectral_centroid(juan, sr=fs, n_fft=winlen, hop_length=hop)      
        juan_bandwidth = librosa.feature.spectral_bandwidth(juan, sr=fs, n_fft=winlen, hop_length=hop)        
        juan_zcr = librosa.feature.zero_crossing_rate(juan, frame_length=winlen, hop_length=hop)
        juan_voicing, dummy = voicing_extraction.voicing(juan, winlen, hop, num_lags=250)
        juan_voicing=np.append(juan_voicing,np.zeros((juan_zcr.shape[1]-juan_voicing.shape[1],)))
        juan_voicing=juan_voicing.reshape((1,len(juan_voicing)))          
        juan_spectral = np.concatenate((juan_rolloff, juan_centroid, juan_bandwidth, juan_zcr, juan_voicing),axis=0)        
        juan_t_spectral = np.arange(juan_spectral.shape[1])*(float(hop)/fs)+float(winlen)/(2*fs);
        juan_gt, aux_gt = load_gt(juan_gt_file, juan_t_spectral);
        juan_gt[len(juan_gt)-1] = 0
        juan_spectral_aux = np.concatenate((juan_spectral[:, juan_gt==1], juan_spectral[:,juan_gt==2], juan_spectral[:,juan_gt==3]),axis=1)
        juan_gt_aux = np.concatenate((juan_gt[juan_gt==1], juan_gt[juan_gt==2], juan_gt[juan_gt==3]),axis=0)  
        juan_gt_aux=juan_gt_aux.reshape((1,len(juan_gt_aux)))
        juan_time_aux = np.concatenate((juan_t_spectral[juan_gt==1], juan_t_spectral[juan_gt==2], juan_t_spectral[juan_gt==3]),axis=0)  
        juan_time_aux=juan_time_aux.reshape((1,len(juan_time_aux)))        
        
        emma_rolloff = librosa.feature.spectral_rolloff(emma, sr=fs, n_fft=winlen, hop_length=hop)
        emma_centroid = librosa.feature.spectral_centroid(emma, sr=fs, n_fft=winlen, hop_length=hop)      
        emma_bandwidth = librosa.feature.spectral_bandwidth(emma, sr=fs, n_fft=winlen, hop_length=hop)        
        emma_zcr = librosa.feature.zero_crossing_rate(emma, frame_length=winlen, hop_length=hop)        
        emma_voicing, dummy = voicing_extraction.voicing(emma, winlen, hop, num_lags=250)               
        emma_voicing=np.append(emma_voicing,np.zeros((emma_zcr.shape[1]-emma_voicing.shape[1],)))
        emma_voicing=emma_voicing.reshape((1,len(emma_voicing)))          
        emma_spectral = np.concatenate((emma_rolloff, emma_centroid, emma_bandwidth, emma_zcr, emma_voicing),axis=0)        
        emma_t_spectral = np.arange(emma_spectral.shape[1])*(float(hop)/fs)+float(winlen)/(2*fs);
        emma_gt, aux_gt = load_gt(emma_gt_file, emma_t_spectral);
        emma_gt[len(emma_gt)-1] = 0
        emma_spectral_aux = np.concatenate((emma_spectral[:, emma_gt==1], emma_spectral[:,emma_gt==2], emma_spectral[:,emma_gt==3]),axis=1)
        emma_gt_aux = np.concatenate((emma_gt[emma_gt==1], emma_gt[emma_gt==2], emma_gt[emma_gt==3]),axis=0)  
        emma_gt_aux=emma_gt_aux.reshape((1,len(emma_gt_aux)))
        emma_time_aux = np.concatenate((emma_t_spectral[emma_gt==1], emma_t_spectral[emma_gt==2], emma_t_spectral[emma_gt==3]),axis=0)  
        emma_time_aux=emma_time_aux.reshape((1,len(emma_time_aux)))        
        
        pablo_rolloff = librosa.feature.spectral_rolloff(pablo, sr=fs, n_fft=winlen, hop_length=hop)
        pablo_centroid = librosa.feature.spectral_centroid(pablo, sr=fs, n_fft=winlen, hop_length=hop)      
        pablo_bandwidth = librosa.feature.spectral_bandwidth(pablo, sr=fs, n_fft=winlen, hop_length=hop)        
        pablo_zcr = librosa.feature.zero_crossing_rate(pablo, frame_length=winlen, hop_length=hop)        
        pablo_voicing, dummy = voicing_extraction.voicing(pablo, winlen, hop, num_lags=250)       
        pablo_voicing=np.append(pablo_voicing,np.zeros((pablo_zcr.shape[1]-pablo_voicing.shape[1],)))
        pablo_voicing=pablo_voicing.reshape((1,len(pablo_voicing)))          
        pablo_spectral = np.concatenate((pablo_rolloff, pablo_centroid, pablo_bandwidth, pablo_zcr, pablo_voicing),axis=0)        
        pablo_t_spectral = np.arange(pablo_spectral.shape[1])*(float(hop)/fs)+float(winlen)/(2*fs);
        pablo_gt, aux_gt = load_gt(pablo_gt_file, pablo_t_spectral);
        pablo_gt[len(pablo_gt)-1] = 0
        pablo_spectral_aux = np.concatenate((pablo_spectral[:, pablo_gt==1], pablo_spectral[:,pablo_gt==2], pablo_spectral[:,pablo_gt==3]),axis=1)
        pablo_gt_aux = np.concatenate((pablo_gt[pablo_gt==1], pablo_gt[pablo_gt==2], pablo_gt[pablo_gt==3]),axis=0)  
        pablo_gt_aux=pablo_gt_aux.reshape((1,len(pablo_gt_aux)))
        pablo_time_aux = np.concatenate((pablo_t_spectral[pablo_gt==1], pablo_t_spectral[pablo_gt==2], pablo_t_spectral[pablo_gt==3]),axis=0)  
        pablo_time_aux=pablo_time_aux.reshape((1,len(pablo_time_aux)))

        ulla_rolloff = librosa.feature.spectral_rolloff(ulla, sr=fs, n_fft=winlen, hop_length=hop)
        ulla_centroid = librosa.feature.spectral_centroid(ulla, sr=fs, n_fft=winlen, hop_length=hop)      
        ulla_bandwidth = librosa.feature.spectral_bandwidth(ulla, sr=fs, n_fft=winlen, hop_length=hop)        
        ulla_zcr = librosa.feature.zero_crossing_rate(ulla, frame_length=winlen, hop_length=hop)        
        ulla_voicing, dummy = voicing_extraction.voicing(ulla, winlen, hop, num_lags=250)               
        ulla_voicing=np.append(ulla_voicing,np.zeros((ulla_zcr.shape[1]-ulla_voicing.shape[1],)))
        ulla_voicing=ulla_voicing.reshape((1,len(ulla_voicing)))          
        ulla_spectral = np.concatenate((ulla_rolloff, ulla_centroid, ulla_bandwidth, ulla_zcr, ulla_voicing),axis=0)        
        ulla_t_spectral = np.arange(ulla_spectral.shape[1])*(float(hop)/fs)+float(winlen)/(2*fs);
        ulla_gt, aux_gt = load_gt(ulla_gt_file, ulla_t_spectral);
        ulla_gt[len(ulla_gt)-1] = 0
        ulla_spectral_aux = np.concatenate((ulla_spectral[:, ulla_gt==1], ulla_spectral[:,ulla_gt==2], ulla_spectral[:,ulla_gt==3]),axis=1)
        ulla_gt_aux = np.concatenate((ulla_gt[ulla_gt==1], ulla_gt[ulla_gt==2], ulla_gt[ulla_gt==3]),axis=0)  
        ulla_gt_aux=ulla_gt_aux.reshape((1,len(ulla_gt_aux)))
        ulla_time_aux = np.concatenate((ulla_t_spectral[ulla_gt==1], ulla_t_spectral[ulla_gt==2], ulla_t_spectral[ulla_gt==3]),axis=0)  
        ulla_time_aux=ulla_time_aux.reshape((1,len(ulla_time_aux)))
    
    else:

        claire_mfcc = librosa.feature.mfcc(claire, sr=fs, n_mfcc=melcoeff, n_fft=winlen, hop_length=hop, n_mels=melbands)
        claire_t_mfcc = np.arange(claire_mfcc.shape[1])*(float(hop)/fs)+float(winlen)/(2*fs);
        claire_gt, aux_gt = load_gt(claire_gt_file, claire_t_mfcc);
        claire_gt[len(claire_gt)-1] = 0
        claire_mfcc_aux = np.concatenate((claire_mfcc[:, claire_gt==1], claire_mfcc[:,claire_gt==2]),axis=1)
        claire_gt_aux = np.concatenate((claire_gt[claire_gt==1], claire_gt[claire_gt==2]),axis=0)  
        claire_gt_aux=claire_gt_aux.reshape((1,len(claire_gt_aux)))
        claire_time_aux = np.concatenate((claire_t_mfcc[claire_gt==1], claire_t_mfcc[claire_gt==2]),axis=0)  
        claire_time_aux=claire_time_aux.reshape((1,len(claire_time_aux)))
        
        juan_mfcc = librosa.feature.mfcc(juan, sr=fs, n_mfcc=melcoeff, n_fft=winlen, hop_length=hop, n_mels=melbands)
        juan_t_mfcc = np.arange(juan_mfcc.shape[1])*(float(hop)/fs)+float(winlen)/(2*fs);
        juan_gt, aux_gt = load_gt(juan_gt_file, juan_t_mfcc);
        juan_gt[len(juan_gt)-1] = 0
        juan_mfcc_aux = np.concatenate((juan_mfcc[:, juan_gt==1], juan_mfcc[:,juan_gt==2]),axis=1)
        juan_gt_aux = np.concatenate((juan_gt[juan_gt==1], juan_gt[juan_gt==2]),axis=0)  
        juan_gt_aux=juan_gt_aux.reshape((1,len(juan_gt_aux)))
        juan_time_aux = np.concatenate((juan_t_mfcc[juan_gt==1], juan_t_mfcc[juan_gt==2]),axis=0)  
        juan_time_aux=juan_time_aux.reshape((1,len(juan_time_aux)))
        
        emma_mfcc = librosa.feature.mfcc(emma, sr=fs, n_mfcc=melcoeff, n_fft=winlen, hop_length=hop, n_mels=melbands)
        emma_t_mfcc = np.arange(emma_mfcc.shape[1])*(float(hop)/fs)+float(winlen)/(2*fs);
        emma_gt, aux_gt = load_gt(emma_gt_file, emma_t_mfcc);
        emma_gt[len(emma_gt)-1] = 0 
        emma_mfcc_aux = np.concatenate((emma_mfcc[:, emma_gt==1], emma_mfcc[:,emma_gt==2]),axis=1)
        emma_gt_aux = np.concatenate((emma_gt[emma_gt==1], emma_gt[emma_gt==2]),axis=0)  
        emma_gt_aux=emma_gt_aux.reshape((1,len(emma_gt_aux))) 
        emma_time_aux = np.concatenate((emma_t_mfcc[emma_gt==1], emma_t_mfcc[emma_gt==2]),axis=0)  
        emma_time_aux=emma_time_aux.reshape((1,len(emma_time_aux)))
        
        pablo_mfcc = librosa.feature.mfcc(pablo, sr=fs, n_mfcc=melcoeff, n_fft=winlen, hop_length=hop, n_mels=melbands)
        pablo_t_mfcc = np.arange(pablo_mfcc.shape[1])*(float(hop)/fs)+float(winlen)/(2*fs);
        pablo_gt, aux_gt = load_gt(pablo_gt_file, pablo_t_mfcc);
        pablo_gt[len(pablo_gt)-1] = 0
        pablo_mfcc_aux = np.concatenate((pablo_mfcc[:, pablo_gt==1], pablo_mfcc[:,pablo_gt==2]),axis=1)
        pablo_gt_aux = np.concatenate((pablo_gt[pablo_gt==1], pablo_gt[pablo_gt==2]),axis=0)  
        pablo_gt_aux=pablo_gt_aux.reshape((1,len(pablo_gt_aux)))
        pablo_time_aux = np.concatenate((pablo_t_mfcc[pablo_gt==1], pablo_t_mfcc[pablo_gt==2]),axis=0)  
        pablo_time_aux=pablo_time_aux.reshape((1,len(pablo_time_aux)))    
        
        ulla_mfcc = librosa.feature.mfcc(ulla, sr=fs, n_mfcc=melcoeff, n_fft=winlen, hop_length=hop, n_mels=melbands)
        ulla_t_mfcc = np.arange(ulla_mfcc.shape[1])*(float(hop)/fs)+float(winlen)/(2*fs);
        ulla_gt, aux_gt = load_gt(ulla_gt_file, ulla_t_mfcc);
        ulla_gt[len(ulla_gt)-1] = 0
        ulla_mfcc_aux = np.concatenate((ulla_mfcc[:, ulla_gt==1], ulla_mfcc[:,ulla_gt==2]),axis=1)
        ulla_gt_aux = np.concatenate((ulla_gt[ulla_gt==1], ulla_gt[ulla_gt==2]),axis=0)  
        ulla_gt_aux=ulla_gt_aux.reshape((1,len(ulla_gt_aux))) 
        ulla_time_aux = np.concatenate((ulla_t_mfcc[ulla_gt==1], ulla_t_mfcc[ulla_gt==2]),axis=0)  
        ulla_time_aux=ulla_time_aux.reshape((1,len(ulla_time_aux)))
    
    
    
#%% EXPORT

    str_aux="../features/claire_spectral_" + str(winlen) + str(hop) + "_test"
    np.save(str_aux, np.concatenate((claire_spectral_aux, claire_gt_aux, claire_time_aux),axis=0))
    str_aux="../features/claire_spectral_" + str(winlen) + str(hop) + "_train"
    train_spectral_aux=np.concatenate((emma_spectral_aux, pablo_spectral_aux, ulla_spectral_aux),axis=1)
    train_gt_aux=np.concatenate((emma_gt_aux, pablo_gt_aux, ulla_gt_aux),axis=1)
    time_aux=np.concatenate((emma_time_aux, pablo_time_aux, ulla_time_aux),axis=1)
    np.save(str_aux, np.concatenate((train_spectral_aux, train_gt_aux, time_aux),axis=0))
    
    str_aux="../features/juan_spectral_" + str(winlen) + str(hop) + "_test"
    np.save(str_aux, np.concatenate((juan_spectral_aux, juan_gt_aux, juan_time_aux),axis=0))
    str_aux="../features/juan_spectral_" + str(winlen) + str(hop) + "_train"
    train_spectral_aux=np.concatenate((emma_spectral_aux, pablo_spectral_aux, ulla_spectral_aux),axis=1)
    train_gt_aux=np.concatenate((emma_gt_aux, pablo_gt_aux, ulla_gt_aux),axis=1)
    time_aux=np.concatenate((emma_time_aux, pablo_time_aux, ulla_time_aux),axis=1)    
    np.save(str_aux, np.concatenate((train_spectral_aux, train_gt_aux, time_aux),axis=0))
    
    str_aux="../features/emma_spectral_" + str(winlen) + str(hop) + "_test"
    np.save(str_aux, np.concatenate((emma_spectral_aux, emma_gt_aux, emma_time_aux),axis=0))
    str_aux="../features/emma_spectral_" + str(winlen) + str(hop) + "_train"
    train_spectral_aux=np.concatenate((pablo_spectral_aux, ulla_spectral_aux),axis=1)
    train_gt_aux=np.concatenate((pablo_gt_aux, ulla_gt_aux),axis=1)
    time_aux=np.concatenate((pablo_time_aux, ulla_time_aux),axis=1)    
    np.save(str_aux, np.concatenate((train_spectral_aux, train_gt_aux, time_aux),axis=0))

    str_aux="../features/pablo_spectral_" + str(winlen) + str(hop) + "_test"
    np.save(str_aux, np.concatenate((pablo_spectral_aux, pablo_gt_aux, pablo_time_aux),axis=0))
    str_aux="../features/pablo_spectral_" + str(winlen) + str(hop) + "_train"
    train_spectral_aux=np.concatenate((emma_spectral_aux, ulla_spectral_aux),axis=1)
    train_gt_aux=np.concatenate((emma_gt_aux, ulla_gt_aux),axis=1)
    time_aux=np.concatenate((emma_time_aux, ulla_time_aux),axis=1)
    np.save(str_aux, np.concatenate((train_spectral_aux, train_gt_aux, time_aux),axis=0))

    str_aux="../features/ulla_spectral_" + str(winlen) + str(hop) + "_test"
    np.save(str_aux, np.concatenate((ulla_spectral_aux, ulla_gt_aux, ulla_time_aux),axis=0))
    str_aux="../features/ulla_spectral_" + str(winlen) + str(hop) + "_train"
    train_spectral_aux=np.concatenate((emma_spectral_aux, pablo_spectral_aux),axis=1)
    train_gt_aux=np.concatenate((emma_gt_aux, pablo_gt_aux),axis=1)
    time_aux=np.concatenate((emma_time_aux, pablo_time_aux),axis=1)
    np.save(str_aux, np.concatenate((train_spectral_aux, train_gt_aux, time_aux),axis=0))
    

#%%
#if __name__=='__main__':

def extract_mfcc(melcoeff = 20, melbands = 40, winlen=1024, hop=512, emb_number = '3', voicing = False):

    claire_audio_file='../audio/claire_mono.wav';    
    claire_gt_file='../audio/claire_embrochure.csv';   
    juan_audio_file='../audio/juan_mono.wav';    
    juan_gt_file='../audio/juan_embrochure.csv';    
    emma_audio_file = '../audio/emma_mono.wav'; 
    emma_gt_file = '../audio/emma_embrochure.csv';
    pablo_audio_file = '../audio/pablo_mono.wav'; 
    pablo_gt_file = '../audio/pablo_embrochure.csv';
    ulla_audio_file = '../audio/ulla_mono.wav'; 
    ulla_gt_file = '../audio/ulla_embrochure.csv';

    #PARAMETROS
#    melbands=40;
#    melcoeff=30;
#    winlen = 1024;
#    hop = 512;
#    overlap = winlen - hop
#    emb_number = '3'

    fs, claire = wav.read(claire_audio_file)
#    t_claire = np.arange(len(claire)) * (1/fs) 
    
    fs, juan = wav.read(juan_audio_file)
#    t_juan = np.arange(len(juan)) * (1/fs)  

    fs, emma = wav.read(emma_audio_file)
#    t_emma = np.arange(len(emma)) * (1/fs)  
    
    fs, pablo = wav.read(pablo_audio_file)
#    t_pablo = np.arange(len(pablo)) * (1/fs)
    
    fs, ulla = wav.read(ulla_audio_file)
#    t_ulla = np.arange(len(ulla)) * (1/fs)  

#%% MFCC EXTRACTION

    if emb_number == '3':

        claire_mfcc = librosa.feature.mfcc(claire, sr=fs, n_mfcc=melcoeff, n_fft=winlen, hop_length=hop, n_mels=melbands)
        if (voicing):        
            claire_voicing, dummy = voicing_extraction.voicing(claire, winlen, hop, num_lags=250)
            claire_voicing=np.append(claire_voicing,np.zeros((claire_mfcc.shape[1]-claire_voicing.shape[1],)))
            claire_voicing=claire_voicing.reshape((1,len(claire_voicing)))         
        claire_t_mfcc = np.arange(claire_mfcc.shape[1])*(float(hop)/fs)+float(winlen)/(2*fs);
        claire_gt, aux_gt = load_gt(claire_gt_file, claire_t_mfcc);
        claire_gt[len(claire_gt)-1] = 0
        if (voicing):
            claire_mfcc=np.concatenate((claire_mfcc, claire_voicing),axis=0)        
        claire_mfcc_aux = np.concatenate((claire_mfcc[:, claire_gt==1], claire_mfcc[:,claire_gt==2], claire_mfcc[:,claire_gt==3]),axis=1)
        claire_gt_aux = np.concatenate((claire_gt[claire_gt==1], claire_gt[claire_gt==2], claire_gt[claire_gt==3]),axis=0)  
        claire_gt_aux=claire_gt_aux.reshape((1,len(claire_gt_aux)))
        claire_time_aux = np.concatenate((claire_t_mfcc[claire_gt==1], claire_t_mfcc[claire_gt==2], claire_t_mfcc[claire_gt==3]),axis=0)  
        claire_time_aux=claire_time_aux.reshape((1,len(claire_time_aux)))

#%%
        
        juan_mfcc = librosa.feature.mfcc(juan, sr=fs, n_mfcc=melcoeff, n_fft=winlen, hop_length=hop, n_mels=melbands)
        if (voicing):        
            juan_voicing, dummy = voicing_extraction.voicing(juan, winlen, hop, num_lags=250)
            juan_voicing=np.append(juan_voicing,np.zeros((juan_mfcc.shape[1]-juan_voicing.shape[1],)))
            juan_voicing=juan_voicing.reshape((1,len(juan_voicing)))        
        juan_t_mfcc = np.arange(juan_mfcc.shape[1])*(float(hop)/fs)+float(winlen)/(2*fs);
        juan_gt, aux_gt = load_gt(juan_gt_file, juan_t_mfcc);
        juan_gt[len(juan_gt)-1] = 0
        if (voicing):        
            juan_mfcc=np.concatenate((juan_mfcc, juan_voicing),axis=0)         
        juan_mfcc_aux = np.concatenate((juan_mfcc[:, juan_gt==1], juan_mfcc[:,juan_gt==2], juan_mfcc[:,juan_gt==3]),axis=1)
        juan_gt_aux = np.concatenate((juan_gt[juan_gt==1], juan_gt[juan_gt==2], juan_gt[juan_gt==3]),axis=0)  
        juan_gt_aux=juan_gt_aux.reshape((1,len(juan_gt_aux)))
        juan_time_aux = np.concatenate((juan_t_mfcc[juan_gt==1], juan_t_mfcc[juan_gt==2], juan_t_mfcc[juan_gt==3]),axis=0)  
        juan_time_aux=juan_time_aux.reshape((1,len(juan_time_aux)))
        
        emma_mfcc = librosa.feature.mfcc(emma, sr=fs, n_mfcc=melcoeff, n_fft=winlen, hop_length=hop, n_mels=melbands)
        if (voicing):        
            emma_voicing, dummy = voicing_extraction.voicing(emma, winlen, hop, num_lags=250)
            emma_voicing=np.append(emma_voicing,np.zeros((emma_mfcc.shape[1]-emma_voicing.shape[1],)))
            emma_voicing=emma_voicing.reshape((1,len(emma_voicing)))        
        emma_t_mfcc = np.arange(emma_mfcc.shape[1])*(float(hop)/fs)+float(winlen)/(2*fs);
        emma_gt, aux_gt = load_gt(emma_gt_file, emma_t_mfcc);
        emma_gt[len(emma_gt)-1] = 0 
        if (voicing):        
            emma_mfcc=np.concatenate((emma_mfcc, emma_voicing),axis=0)         
        emma_mfcc_aux = np.concatenate((emma_mfcc[:, emma_gt==1], emma_mfcc[:,emma_gt==2], emma_mfcc[:,emma_gt==3]),axis=1)
        emma_gt_aux = np.concatenate((emma_gt[emma_gt==1], emma_gt[emma_gt==2], emma_gt[emma_gt==3]),axis=0)  
        emma_gt_aux=emma_gt_aux.reshape((1,len(emma_gt_aux))) 
        emma_time_aux = np.concatenate((emma_t_mfcc[emma_gt==1], emma_t_mfcc[emma_gt==2], emma_t_mfcc[emma_gt==3]),axis=0)  
        emma_time_aux=emma_time_aux.reshape((1,len(emma_time_aux)))
        
        pablo_mfcc = librosa.feature.mfcc(pablo, sr=fs, n_mfcc=melcoeff, n_fft=winlen, hop_length=hop, n_mels=melbands)
        if (voicing):            
            pablo_voicing, dummy = voicing_extraction.voicing(pablo, winlen, hop, num_lags=250)
            pablo_voicing=np.append(pablo_voicing,np.zeros((pablo_mfcc.shape[1]-pablo_voicing.shape[1],)))
            pablo_voicing=pablo_voicing.reshape((1,len(pablo_voicing)))        
        pablo_t_mfcc = np.arange(pablo_mfcc.shape[1])*(float(hop)/fs)+float(winlen)/(2*fs);
        pablo_gt, aux_gt = load_gt(pablo_gt_file, pablo_t_mfcc);
        pablo_gt[len(pablo_gt)-1] = 0
        if (voicing):
            pablo_mfcc=np.concatenate((pablo_mfcc, pablo_voicing),axis=0)         
        pablo_mfcc_aux = np.concatenate((pablo_mfcc[:, pablo_gt==1], pablo_mfcc[:,pablo_gt==2], pablo_mfcc[:,pablo_gt==3]),axis=1)
        pablo_gt_aux = np.concatenate((pablo_gt[pablo_gt==1], pablo_gt[pablo_gt==2], pablo_gt[pablo_gt==3]),axis=0)  
        pablo_gt_aux=pablo_gt_aux.reshape((1,len(pablo_gt_aux)))
        pablo_time_aux = np.concatenate((pablo_t_mfcc[pablo_gt==1], pablo_t_mfcc[pablo_gt==2], pablo_t_mfcc[pablo_gt==3]),axis=0)  
        pablo_time_aux=pablo_time_aux.reshape((1,len(pablo_time_aux)))    
        
        ulla_mfcc = librosa.feature.mfcc(ulla, sr=fs, n_mfcc=melcoeff, n_fft=winlen, hop_length=hop, n_mels=melbands)
        if (voicing):        
            ulla_voicing, dummy = voicing_extraction.voicing(ulla, winlen, hop, num_lags=250)
            ulla_voicing=np.append(ulla_voicing,np.zeros((ulla_mfcc.shape[1]-ulla_voicing.shape[1],)))
            ulla_voicing=ulla_voicing.reshape((1,len(ulla_voicing)))        
        ulla_t_mfcc = np.arange(ulla_mfcc.shape[1])*(float(hop)/fs)+float(winlen)/(2*fs);
        ulla_gt, aux_gt = load_gt(ulla_gt_file, ulla_t_mfcc);
        ulla_gt[len(ulla_gt)-1] = 0
        if (voicing):        
            ulla_mfcc=np.concatenate((ulla_mfcc, ulla_voicing),axis=0)         
        ulla_mfcc_aux = np.concatenate((ulla_mfcc[:, ulla_gt==1], ulla_mfcc[:,ulla_gt==2], ulla_mfcc[:,ulla_gt==3]),axis=1)
        ulla_gt_aux = np.concatenate((ulla_gt[ulla_gt==1], ulla_gt[ulla_gt==2], ulla_gt[ulla_gt==3]),axis=0)  
        ulla_gt_aux=ulla_gt_aux.reshape((1,len(ulla_gt_aux))) 
        ulla_time_aux = np.concatenate((ulla_t_mfcc[ulla_gt==1], ulla_t_mfcc[ulla_gt==2], ulla_t_mfcc[ulla_gt==3]),axis=0)  
        ulla_time_aux=ulla_time_aux.reshape((1,len(ulla_time_aux)))

    else:

        claire_mfcc = librosa.feature.mfcc(claire, sr=fs, n_mfcc=melcoeff, n_fft=winlen, hop_length=hop, n_mels=melbands)
        claire_t_mfcc = np.arange(claire_mfcc.shape[1])*(float(hop)/fs)+float(winlen)/(2*fs);
        claire_gt, aux_gt = load_gt(claire_gt_file, claire_t_mfcc);
        claire_gt[len(claire_gt)-1] = 0
        claire_mfcc_aux = np.concatenate((claire_mfcc[:, claire_gt==1], claire_mfcc[:,claire_gt==2]),axis=1)
        claire_gt_aux = np.concatenate((claire_gt[claire_gt==1], claire_gt[claire_gt==2]),axis=0)  
        claire_gt_aux=claire_gt_aux.reshape((1,len(claire_gt_aux)))
        claire_time_aux = np.concatenate((claire_t_mfcc[claire_gt==1], claire_t_mfcc[claire_gt==2]),axis=0)  
        claire_time_aux=claire_time_aux.reshape((1,len(claire_time_aux)))
        
        juan_mfcc = librosa.feature.mfcc(juan, sr=fs, n_mfcc=melcoeff, n_fft=winlen, hop_length=hop, n_mels=melbands)
        juan_t_mfcc = np.arange(juan_mfcc.shape[1])*(float(hop)/fs)+float(winlen)/(2*fs);
        juan_gt, aux_gt = load_gt(juan_gt_file, juan_t_mfcc);
        juan_gt[len(juan_gt)-1] = 0
        juan_mfcc_aux = np.concatenate((juan_mfcc[:, juan_gt==1], juan_mfcc[:,juan_gt==2]),axis=1)
        juan_gt_aux = np.concatenate((juan_gt[juan_gt==1], juan_gt[juan_gt==2]),axis=0)  
        juan_gt_aux=juan_gt_aux.reshape((1,len(juan_gt_aux)))
        juan_time_aux = np.concatenate((juan_t_mfcc[juan_gt==1], juan_t_mfcc[juan_gt==2]),axis=0)  
        juan_time_aux=juan_time_aux.reshape((1,len(juan_time_aux)))
        
        emma_mfcc = librosa.feature.mfcc(emma, sr=fs, n_mfcc=melcoeff, n_fft=winlen, hop_length=hop, n_mels=melbands)
        emma_t_mfcc = np.arange(emma_mfcc.shape[1])*(float(hop)/fs)+float(winlen)/(2*fs);
        emma_gt, aux_gt = load_gt(emma_gt_file, emma_t_mfcc);
        emma_gt[len(emma_gt)-1] = 0 
        emma_mfcc_aux = np.concatenate((emma_mfcc[:, emma_gt==1], emma_mfcc[:,emma_gt==2]),axis=1)
        emma_gt_aux = np.concatenate((emma_gt[emma_gt==1], emma_gt[emma_gt==2]),axis=0)  
        emma_gt_aux=emma_gt_aux.reshape((1,len(emma_gt_aux))) 
        emma_time_aux = np.concatenate((emma_t_mfcc[emma_gt==1], emma_t_mfcc[emma_gt==2]),axis=0)  
        emma_time_aux=emma_time_aux.reshape((1,len(emma_time_aux)))
        
        pablo_mfcc = librosa.feature.mfcc(pablo, sr=fs, n_mfcc=melcoeff, n_fft=winlen, hop_length=hop, n_mels=melbands)
        pablo_t_mfcc = np.arange(pablo_mfcc.shape[1])*(float(hop)/fs)+float(winlen)/(2*fs);
        pablo_gt, aux_gt = load_gt(pablo_gt_file, pablo_t_mfcc);
        pablo_gt[len(pablo_gt)-1] = 0
        pablo_mfcc_aux = np.concatenate((pablo_mfcc[:, pablo_gt==1], pablo_mfcc[:,pablo_gt==2]),axis=1)
        pablo_gt_aux = np.concatenate((pablo_gt[pablo_gt==1], pablo_gt[pablo_gt==2]),axis=0)  
        pablo_gt_aux=pablo_gt_aux.reshape((1,len(pablo_gt_aux)))
        pablo_time_aux = np.concatenate((pablo_t_mfcc[pablo_gt==1], pablo_t_mfcc[pablo_gt==2]),axis=0)  
        pablo_time_aux=pablo_time_aux.reshape((1,len(pablo_time_aux)))    
        
        ulla_mfcc = librosa.feature.mfcc(ulla, sr=fs, n_mfcc=melcoeff, n_fft=winlen, hop_length=hop, n_mels=melbands)
        ulla_t_mfcc = np.arange(ulla_mfcc.shape[1])*(float(hop)/fs)+float(winlen)/(2*fs);
        ulla_gt, aux_gt = load_gt(ulla_gt_file, ulla_t_mfcc);
        ulla_gt[len(ulla_gt)-1] = 0
        ulla_mfcc_aux = np.concatenate((ulla_mfcc[:, ulla_gt==1], ulla_mfcc[:,ulla_gt==2]),axis=1)
        ulla_gt_aux = np.concatenate((ulla_gt[ulla_gt==1], ulla_gt[ulla_gt==2]),axis=0)  
        ulla_gt_aux=ulla_gt_aux.reshape((1,len(ulla_gt_aux))) 
        ulla_time_aux = np.concatenate((ulla_t_mfcc[ulla_gt==1], ulla_t_mfcc[ulla_gt==2]),axis=0)  
        ulla_time_aux=ulla_time_aux.reshape((1,len(ulla_time_aux)))
    
    
    
#%% EXPORT

    if (voicing):
        str_aux="../features/claire_mfccvoicing_" + str(melcoeff) + str(melbands) + "_test"
    else:
        str_aux="../features/claire_mfcc_" + str(melcoeff) + str(melbands) + "_test"
    np.save(str_aux, np.concatenate((claire_mfcc_aux, claire_gt_aux, claire_time_aux),axis=0))
    if (voicing):
        str_aux="../features/claire_mfccvoicing_" + str(melcoeff) + str(melbands) + "_train"
    else:
        str_aux="../features/claire_mfcc_" + str(melcoeff) + str(melbands) + "_train"
    train_mfcc_aux=np.concatenate((emma_mfcc_aux, pablo_mfcc_aux, ulla_mfcc_aux),axis=1)
    train_gt_aux=np.concatenate((emma_gt_aux, pablo_gt_aux, ulla_gt_aux),axis=1)
    time_aux=np.concatenate((emma_time_aux, pablo_time_aux, ulla_time_aux),axis=1)
    np.save(str_aux, np.concatenate((train_mfcc_aux, train_gt_aux, time_aux),axis=0))
    
    if (voicing):    
        str_aux="../features/juan_mfccvoicing_" + str(melcoeff) + str(melbands) + "_test"
    else:
        str_aux="../features/juan_mfcc_" + str(melcoeff) + str(melbands) + "_test"
    np.save(str_aux, np.concatenate((juan_mfcc_aux, juan_gt_aux, juan_time_aux),axis=0))
    if (voicing):
        str_aux="../features/juan_mfccvoicing_" + str(melcoeff) + str(melbands) + "_train"
    else:
        str_aux="../features/juan_mfcc_" + str(melcoeff) + str(melbands) + "_train"
    train_mfcc_aux=np.concatenate((emma_mfcc_aux, pablo_mfcc_aux, ulla_mfcc_aux),axis=1)
    train_gt_aux=np.concatenate((emma_gt_aux, pablo_gt_aux, ulla_gt_aux),axis=1)
    time_aux=np.concatenate((emma_time_aux, pablo_time_aux, ulla_time_aux),axis=1)    
    np.save(str_aux, np.concatenate((train_mfcc_aux, train_gt_aux, time_aux),axis=0))
    
    if (voicing):    
        str_aux="../features/emma_mfccvoicing_" + str(melcoeff) + str(melbands) + "_test"
    else:
        str_aux="../features/emma_mfcc_" + str(melcoeff) + str(melbands) + "_test"
    np.save(str_aux, np.concatenate((emma_mfcc_aux, emma_gt_aux, emma_time_aux),axis=0))
    str_aux="../features/emma_mfcc_" + str(melcoeff) + str(melbands) + "_train"
    train_mfcc_aux=np.concatenate((pablo_mfcc_aux, ulla_mfcc_aux),axis=1)
    train_gt_aux=np.concatenate((pablo_gt_aux, ulla_gt_aux),axis=1)
    time_aux=np.concatenate((pablo_time_aux, ulla_time_aux),axis=1)    
    np.save(str_aux, np.concatenate((train_mfcc_aux, train_gt_aux, time_aux),axis=0))

    if (voicing):
        str_aux="../features/pablo_mfccvoicing_" + str(melcoeff) + str(melbands) + "_test"
    else:
        str_aux="../features/pablo_mfcc_" + str(melcoeff) + str(melbands) + "_test"
    np.save(str_aux, np.concatenate((pablo_mfcc_aux, pablo_gt_aux, pablo_time_aux),axis=0))
    if (voicing):
        str_aux="../features/pablo_mfccvoicing_" + str(melcoeff) + str(melbands) + "_train"
    else:
        str_aux="../features/pablo_mfcc_" + str(melcoeff) + str(melbands) + "_train"
    train_mfcc_aux=np.concatenate((emma_mfcc_aux, ulla_mfcc_aux),axis=1)
    train_gt_aux=np.concatenate((emma_gt_aux, ulla_gt_aux),axis=1)
    time_aux=np.concatenate((emma_time_aux, ulla_time_aux),axis=1)
    np.save(str_aux, np.concatenate((train_mfcc_aux, train_gt_aux, time_aux),axis=0))

    if (voicing):
        str_aux="../features/ulla_mfccvoicing_" + str(melcoeff) + str(melbands) + "_test"
    else:
        str_aux="../features/ulla_mfcc_" + str(melcoeff) + str(melbands) + "_test"        
    np.save(str_aux, np.concatenate((ulla_mfcc_aux, ulla_gt_aux, ulla_time_aux),axis=0))
    if (voicing):    
        str_aux="../features/ulla_mfccvoicing_" + str(melcoeff) + str(melbands) + "_train"
    else:
        str_aux="../features/ulla_mfcc_" + str(melcoeff) + str(melbands) + "_train"
    train_mfcc_aux=np.concatenate((emma_mfcc_aux, pablo_mfcc_aux),axis=1)
    train_gt_aux=np.concatenate((emma_gt_aux, pablo_gt_aux),axis=1)
    time_aux=np.concatenate((emma_time_aux, pablo_time_aux),axis=1)
    np.save(str_aux, np.concatenate((train_mfcc_aux, train_gt_aux, time_aux),axis=0))
    

def extract_spectral_contrast(nbands = 6, quantile = 0.02, winlen=1024, hop=512, emb_number = '3'):
    
#if __name__=='__main__':
#    
#    nbands=6
#    quantile=0.02
#    winlen=1024
#    hop=512
#    emb_number='3'
        

    claire_audio_file='../audio/claire_mono.wav';    
    claire_gt_file='../audio/claire_embrochure.csv';   
    juan_audio_file='../audio/juan_mono.wav';    
    juan_gt_file='../audio/juan_embrochure.csv';    
    emma_audio_file = '../audio/emma_mono.wav'; 
    emma_gt_file = '../audio/emma_embrochure.csv';
    pablo_audio_file = '../audio/pablo_mono.wav'; 
    pablo_gt_file = '../audio/pablo_embrochure.csv';
    ulla_audio_file = '../audio/ulla_mono.wav'; 
    ulla_gt_file = '../audio/ulla_embrochure.csv';

    fs, claire = wav.read(claire_audio_file)
#    t_claire = np.arange(len(claire)) * (1/fs) 
    
    fs, juan = wav.read(juan_audio_file)
#    t_juan = np.arange(len(juan)) * (1/fs)  

    fs, emma = wav.read(emma_audio_file)
#    t_emma = np.arange(len(emma)) * (1/fs)  
    
    fs, pablo = wav.read(pablo_audio_file)
#    t_pablo = np.arange(len(pablo)) * (1/fs)
    
    fs, ulla = wav.read(ulla_audio_file)
#    t_ulla = np.arange(len(ulla)) * (1/fs)  

#%% SPECTRAL CONTRAST EXTRACTION

    if emb_number == '3':
    
        claire_spectral_contrast = librosa.feature.spectral_contrast(claire, sr=fs, n_fft=winlen, hop_length=hop, n_bands=nbands, quantile=quantile)
        claire_t_spectral_contrast = np.arange(claire_spectral_contrast.shape[1])*(float(hop)/fs)+float(winlen)/(2*fs);
        claire_gt, aux_gt = load_gt(claire_gt_file, claire_t_spectral_contrast);
        claire_gt[len(claire_gt)-1] = 0
        claire_spectral_contrast_aux = np.concatenate((claire_spectral_contrast[:, claire_gt==1], claire_spectral_contrast[:,claire_gt==2], claire_spectral_contrast[:,claire_gt==3]),axis=1)
        claire_gt_aux = np.concatenate((claire_gt[claire_gt==1], claire_gt[claire_gt==2], claire_gt[claire_gt==3]),axis=0)  
        claire_gt_aux=claire_gt_aux.reshape((1,len(claire_gt_aux)))
        claire_time_aux = np.concatenate((claire_t_spectral_contrast[claire_gt==1], claire_t_spectral_contrast[claire_gt==2], claire_t_spectral_contrast[claire_gt==3]),axis=0)  
        claire_time_aux=claire_time_aux.reshape((1,len(claire_time_aux)))
        
        juan_spectral_contrast = librosa.feature.spectral_contrast(juan, sr=fs, n_fft=winlen, hop_length=hop, n_bands=nbands, quantile=quantile)
        juan_t_spectral_contrast = np.arange(juan_spectral_contrast.shape[1])*(float(hop)/fs)+float(winlen)/(2*fs);
        juan_gt, aux_gt = load_gt(juan_gt_file, juan_t_spectral_contrast);
        juan_gt[len(juan_gt)-1] = 0
        juan_spectral_contrast_aux = np.concatenate((juan_spectral_contrast[:, juan_gt==1], juan_spectral_contrast[:,juan_gt==2], juan_spectral_contrast[:,juan_gt==3]),axis=1)
        juan_gt_aux = np.concatenate((juan_gt[juan_gt==1], juan_gt[juan_gt==2], juan_gt[juan_gt==3]),axis=0)  
        juan_gt_aux=juan_gt_aux.reshape((1,len(juan_gt_aux)))
        juan_time_aux = np.concatenate((juan_t_spectral_contrast[juan_gt==1], juan_t_spectral_contrast[juan_gt==2], juan_t_spectral_contrast[juan_gt==3]),axis=0)  
        juan_time_aux=juan_time_aux.reshape((1,len(juan_time_aux)))
        
        emma_spectral_contrast = librosa.feature.spectral_contrast(emma, sr=fs, n_fft=winlen, hop_length=hop, n_bands=nbands, quantile=quantile)
        emma_t_spectral_contrast = np.arange(emma_spectral_contrast.shape[1])*(float(hop)/fs)+float(winlen)/(2*fs);
        emma_gt, aux_gt = load_gt(emma_gt_file, emma_t_spectral_contrast);
        emma_gt[len(emma_gt)-1] = 0 
        emma_spectral_contrast_aux = np.concatenate((emma_spectral_contrast[:, emma_gt==1], emma_spectral_contrast[:,emma_gt==2], emma_spectral_contrast[:,emma_gt==3]),axis=1)
        emma_gt_aux = np.concatenate((emma_gt[emma_gt==1], emma_gt[emma_gt==2], emma_gt[emma_gt==3]),axis=0)  
        emma_gt_aux=emma_gt_aux.reshape((1,len(emma_gt_aux))) 
        emma_time_aux = np.concatenate((emma_t_spectral_contrast[emma_gt==1], emma_t_spectral_contrast[emma_gt==2], emma_t_spectral_contrast[emma_gt==3]),axis=0)  
        emma_time_aux=emma_time_aux.reshape((1,len(emma_time_aux)))
        
        pablo_spectral_contrast = librosa.feature.spectral_contrast(pablo, sr=fs, n_fft=winlen, hop_length=hop, n_bands=nbands, quantile=quantile)
        pablo_t_spectral_contrast = np.arange(pablo_spectral_contrast.shape[1])*(float(hop)/fs)+float(winlen)/(2*fs);
        pablo_gt, aux_gt = load_gt(pablo_gt_file, pablo_t_spectral_contrast);
        pablo_gt[len(pablo_gt)-1] = 0
        pablo_spectral_contrast_aux = np.concatenate((pablo_spectral_contrast[:, pablo_gt==1], pablo_spectral_contrast[:,pablo_gt==2], pablo_spectral_contrast[:,pablo_gt==3]),axis=1)
        pablo_gt_aux = np.concatenate((pablo_gt[pablo_gt==1], pablo_gt[pablo_gt==2], pablo_gt[pablo_gt==3]),axis=0)  
        pablo_gt_aux=pablo_gt_aux.reshape((1,len(pablo_gt_aux)))
        pablo_time_aux = np.concatenate((pablo_t_spectral_contrast[pablo_gt==1], pablo_t_spectral_contrast[pablo_gt==2], pablo_t_spectral_contrast[pablo_gt==3]),axis=0)  
        pablo_time_aux=pablo_time_aux.reshape((1,len(pablo_time_aux)))    
        
        ulla_spectral_contrast = librosa.feature.spectral_contrast(ulla, sr=fs, n_fft=winlen, hop_length=hop, n_bands=nbands, quantile=quantile)
        ulla_t_spectral_contrast = np.arange(ulla_spectral_contrast.shape[1])*(float(hop)/fs)+float(winlen)/(2*fs);
        ulla_gt, aux_gt = load_gt(ulla_gt_file, ulla_t_spectral_contrast);
        ulla_gt[len(ulla_gt)-1] = 0
        ulla_spectral_contrast_aux = np.concatenate((ulla_spectral_contrast[:, ulla_gt==1], ulla_spectral_contrast[:,ulla_gt==2], ulla_spectral_contrast[:,ulla_gt==3]),axis=1)
        ulla_gt_aux = np.concatenate((ulla_gt[ulla_gt==1], ulla_gt[ulla_gt==2], ulla_gt[ulla_gt==3]),axis=0)  
        ulla_gt_aux=ulla_gt_aux.reshape((1,len(ulla_gt_aux))) 
        ulla_time_aux = np.concatenate((ulla_t_spectral_contrast[ulla_gt==1], ulla_t_spectral_contrast[ulla_gt==2], ulla_t_spectral_contrast[ulla_gt==3]),axis=0)  
        ulla_time_aux=ulla_time_aux.reshape((1,len(ulla_time_aux)))
        
    else: 
    
        claire_spectral_contrast = librosa.feature.spectral_contrast(claire, sr=fs, n_fft=winlen, hop_length=hop, n_bands=nbands, quantile=quantile)
        claire_t_spectral_contrast = np.arange(claire_spectral_contrast.shape[1])*(float(hop)/fs)+float(winlen)/(2*fs);
        claire_gt, aux_gt = load_gt(claire_gt_file, claire_t_spectral_contrast);
        claire_gt[len(claire_gt)-1] = 0
        claire_spectral_contrast_aux = np.concatenate((claire_spectral_contrast[:, claire_gt==1], claire_spectral_contrast[:,claire_gt==2]),axis=1)
        claire_gt_aux = np.concatenate((claire_gt[claire_gt==1], claire_gt[claire_gt==2]),axis=0)  
        claire_gt_aux=claire_gt_aux.reshape((1,len(claire_gt_aux)))
        claire_time_aux = np.concatenate((claire_t_spectral_contrast[claire_gt==1], claire_t_spectral_contrast[claire_gt==2]),axis=0)  
        claire_time_aux=claire_time_aux.reshape((1,len(claire_time_aux)))
        
        juan_spectral_contrast = librosa.feature.spectral_contrast(juan, sr=fs, n_fft=winlen, hop_length=hop, n_bands=nbands, quantile=quantile)
        juan_t_spectral_contrast = np.arange(juan_spectral_contrast.shape[1])*(float(hop)/fs)+float(winlen)/(2*fs);
        juan_gt, aux_gt = load_gt(juan_gt_file, juan_t_spectral_contrast);
        juan_gt[len(juan_gt)-1] = 0
        juan_spectral_contrast_aux = np.concatenate((juan_spectral_contrast[:, juan_gt==1], juan_spectral_contrast[:,juan_gt==2]),axis=1)
        juan_gt_aux = np.concatenate((juan_gt[juan_gt==1], juan_gt[juan_gt==2]),axis=0)  
        juan_gt_aux=juan_gt_aux.reshape((1,len(juan_gt_aux)))
        juan_time_aux = np.concatenate((juan_t_spectral_contrast[juan_gt==1], juan_t_spectral_contrast[juan_gt==2]),axis=0)  
        juan_time_aux=juan_time_aux.reshape((1,len(juan_time_aux)))
        
        emma_spectral_contrast = librosa.feature.spectral_contrast(emma, sr=fs, n_fft=winlen, hop_length=hop, n_bands=nbands, quantile=quantile)
        emma_t_spectral_contrast = np.arange(emma_spectral_contrast.shape[1])*(float(hop)/fs)+float(winlen)/(2*fs);
        emma_gt, aux_gt = load_gt(emma_gt_file, emma_t_spectral_contrast);
        emma_gt[len(emma_gt)-1] = 0 
        emma_spectral_contrast_aux = np.concatenate((emma_spectral_contrast[:, emma_gt==1], emma_spectral_contrast[:,emma_gt==2]),axis=1)
        emma_gt_aux = np.concatenate((emma_gt[emma_gt==1], emma_gt[emma_gt==2]),axis=0)  
        emma_gt_aux=emma_gt_aux.reshape((1,len(emma_gt_aux))) 
        emma_time_aux = np.concatenate((emma_t_spectral_contrast[emma_gt==1], emma_t_spectral_contrast[emma_gt==2]),axis=0)  
        emma_time_aux=emma_time_aux.reshape((1,len(emma_time_aux)))
        
        pablo_spectral_contrast = librosa.feature.spectral_contrast(pablo, sr=fs, n_fft=winlen, hop_length=hop, n_bands=nbands, quantile=quantile)
        pablo_t_spectral_contrast = np.arange(pablo_spectral_contrast.shape[1])*(float(hop)/fs)+float(winlen)/(2*fs);
        pablo_gt, aux_gt = load_gt(pablo_gt_file, pablo_t_spectral_contrast);
        pablo_gt[len(pablo_gt)-1] = 0
        pablo_spectral_contrast_aux = np.concatenate((pablo_spectral_contrast[:, pablo_gt==1], pablo_spectral_contrast[:,pablo_gt==2]),axis=1)
        pablo_gt_aux = np.concatenate((pablo_gt[pablo_gt==1], pablo_gt[pablo_gt==2]),axis=0)  
        pablo_gt_aux=pablo_gt_aux.reshape((1,len(pablo_gt_aux)))
        pablo_time_aux = np.concatenate((pablo_t_spectral_contrast[pablo_gt==1], pablo_t_spectral_contrast[pablo_gt==2]),axis=0)  
        pablo_time_aux=pablo_time_aux.reshape((1,len(pablo_time_aux)))    
        
        ulla_spectral_contrast = librosa.feature.spectral_contrast(ulla, sr=fs, n_fft=winlen, hop_length=hop, n_bands=nbands, quantile=quantile)
        ulla_t_spectral_contrast = np.arange(ulla_spectral_contrast.shape[1])*(float(hop)/fs)+float(winlen)/(2*fs);
        ulla_gt, aux_gt = load_gt(ulla_gt_file, ulla_t_spectral_contrast);
        ulla_gt[len(ulla_gt)-1] = 0
        ulla_spectral_contrast_aux = np.concatenate((ulla_spectral_contrast[:, ulla_gt==1], ulla_spectral_contrast[:,ulla_gt==2]),axis=1)
        ulla_gt_aux = np.concatenate((ulla_gt[ulla_gt==1], ulla_gt[ulla_gt==2]),axis=0)  
        ulla_gt_aux=ulla_gt_aux.reshape((1,len(ulla_gt_aux))) 
        ulla_time_aux = np.concatenate((ulla_t_spectral_contrast[ulla_gt==1], ulla_t_spectral_contrast[ulla_gt==2]),axis=0)  
        ulla_time_aux=ulla_time_aux.reshape((1,len(ulla_time_aux)))
        
    
    
#%% EXPORT

    str_aux="../features/claire_spectral_contrast_" + str(nbands) + "_test"
    np.save(str_aux, np.concatenate((claire_spectral_contrast_aux, claire_gt_aux, claire_time_aux),axis=0))
    str_aux="../features/claire_spectral_contrast_" + str(nbands) + "_train"
    train_spectral_contrast_aux=np.concatenate((emma_spectral_contrast_aux, pablo_spectral_contrast_aux, ulla_spectral_contrast_aux),axis=1)
    train_gt_aux=np.concatenate((emma_gt_aux, pablo_gt_aux, ulla_gt_aux),axis=1)
    time_aux=np.concatenate((emma_time_aux, pablo_time_aux, ulla_time_aux),axis=1)
    np.save(str_aux, np.concatenate((train_spectral_contrast_aux, train_gt_aux, time_aux),axis=0))
    
    str_aux="../features/juan_spectral_contrast_" + str(nbands) + "_test"
    np.save(str_aux, np.concatenate((juan_spectral_contrast_aux, juan_gt_aux, juan_time_aux),axis=0))
    str_aux="../features/juan_spectral_contrast_" + str(nbands) + "_train"
    train_spectral_contrast_aux=np.concatenate((emma_spectral_contrast_aux, pablo_spectral_contrast_aux, ulla_spectral_contrast_aux),axis=1)
    train_gt_aux=np.concatenate((emma_gt_aux, pablo_gt_aux, ulla_gt_aux),axis=1)
    time_aux=np.concatenate((emma_time_aux, pablo_time_aux, ulla_time_aux),axis=1)    
    np.save(str_aux, np.concatenate((train_spectral_contrast_aux, train_gt_aux, time_aux),axis=0))
    
    str_aux="../features/emma_spectral_contrast_" + str(nbands) + "_test"
    np.save(str_aux, np.concatenate((emma_spectral_contrast_aux, emma_gt_aux, emma_time_aux),axis=0))
    str_aux="../features/emma_spectral_contrast_" + str(nbands) + "_train"
    train_spectral_contrast_aux=np.concatenate((pablo_spectral_contrast_aux, ulla_spectral_contrast_aux),axis=1)
    train_gt_aux=np.concatenate((pablo_gt_aux, ulla_gt_aux),axis=1)
    time_aux=np.concatenate((pablo_time_aux, ulla_time_aux),axis=1)    
    np.save(str_aux, np.concatenate((train_spectral_contrast_aux, train_gt_aux, time_aux),axis=0))

    str_aux="../features/pablo_spectral_contrast_" + str(nbands) + "_test"
    np.save(str_aux, np.concatenate((pablo_spectral_contrast_aux, pablo_gt_aux, pablo_time_aux),axis=0))
    str_aux="../features/pablo_spectral_contrast_" + str(nbands) + "_train"
    train_spectral_contrast_aux=np.concatenate((emma_spectral_contrast_aux, ulla_spectral_contrast_aux),axis=1)
    train_gt_aux=np.concatenate((emma_gt_aux, ulla_gt_aux),axis=1)
    time_aux=np.concatenate((emma_time_aux, ulla_time_aux),axis=1)
    np.save(str_aux, np.concatenate((train_spectral_contrast_aux, train_gt_aux, time_aux),axis=0))

    str_aux="../features/ulla_spectral_contrast_" + str(nbands) + "_test"
    np.save(str_aux, np.concatenate((ulla_spectral_contrast_aux, ulla_gt_aux, ulla_time_aux),axis=0))
    str_aux="../features/ulla_spectral_contrast_" + str(nbands) + "_train"
    train_spectral_contrast_aux=np.concatenate((emma_spectral_contrast_aux, pablo_spectral_contrast_aux),axis=1)
    train_gt_aux=np.concatenate((emma_gt_aux, pablo_gt_aux),axis=1)
    time_aux=np.concatenate((emma_time_aux, pablo_time_aux),axis=1)
    np.save(str_aux, np.concatenate((train_spectral_contrast_aux, train_gt_aux, time_aux),axis=0))