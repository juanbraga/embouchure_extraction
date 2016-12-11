# -*- coding: utf-8 -*-
"""
Created on Mon Dec 05 20:22:02 2016

@author: Juan
"""

import numpy as np
import seaborn as sns
sns.set(style="whitegrid")
import matplotlib.pyplot as plt
import pandas as pd

plt.close('all')

#%% CRAZY PLOTS

##%% pairwaise comparison classical features
#features_spectral="../features/" + 'juan' + "_spectral_" + str(256) + str(128) + "_train.npy"
#data_spectral=np.load(features_spectral)
#d_spectral = {'rolloff': data_spectral[0,:], 'centroid': data_spectral[1,:], 'bandwith': data_spectral[2,:], 'zcr': data_spectral[3,:], 'voicing': data_spectral[4,:], 'gt': data_spectral[5,:]}
#df_spectral = pd.DataFrame(data=d_spectral)
#sns.pairplot(df_spectral, hue='gt', vars=['voicing', 'zcr','centroid'])
#
##%% 2x2 probabilites out of RandomForest
#melcoeff=20
#melbands=40
#probabilties="../prediction/" + 'pablo' + "_mfcc_" + str(melcoeff) + str(melbands) + "_proba.npy"
#proba=np.load(probabilties)
#proba=proba.T
#d_proba = {'bhc': proba[0,:], 'breathy': proba[1,:], 'normal': proba[2,:], 'time': proba[3,:], 'gt': proba[4,:]}
#df_proba = pd.DataFrame(data=d_proba)
#sns.pairplot(df_proba, hue='gt', vars=['bhc', 'breathy','normal'])

#%% LPC

lpc_10_knn = [47,52,55,42,55]
lpc_10_rf = [43,56,57,40,54]
lpc_10_svm = [50,70,70,56,72]
lpc_20_knn = [47,52,51,46,52]
lpc_20_rf = [45,52,51,44,52]
lpc_20_svm = [52,72,70,45,64]
lpc_40_knn = [49,52,56,53,56]
lpc_40_rf = [45,52,54,45,51]
lpc_40_svm = [55,69,69,48,67]

d_lpc = {'accuracy' : pd.Series(np.concatenate((lpc_10_knn,lpc_10_rf,lpc_10_svm,lpc_20_knn,lpc_20_rf,lpc_20_svm,lpc_40_knn,lpc_40_rf,lpc_40_svm))),
     'artist': pd.Series(['ulla','pablo','emma','juan','claire','ulla','pablo','emma','juan','claire','ulla','pablo','emma','juan','claire', \
     'ulla','pablo','emma','juan','claire','ulla','pablo','emma','juan','claire','ulla','pablo','emma','juan','claire', \
     'ulla','pablo','emma','juan','claire','ulla','pablo','emma','juan','claire','ulla','pablo','emma','juan','claire']),
     'clf': pd.Series(['knn','knn','knn','knn','knn','rf','rf','rf','rf','rf','svm','svm','svm','svm','svm', \
     'knn','knn','knn','knn','knn','rf','rf','rf','rf','rf','svm','svm','svm','svm','svm', \
     'knn','knn','knn','knn','knn','rf','rf','rf','rf','rf','svm','svm','svm','svm','svm']),
     'poles': pd.Series(['10poles','10poles','10poles','10poles','10poles','10poles','10poles','10poles','10poles','10poles','10poles','10poles', \
     '10poles','10poles','10poles','20poles','20poles','20poles','20poles','20poles','20poles','20poles','20poles','20poles','20poles','20poles', \
     '20poles','20poles','20poles','20poles','40poles','40poles','40poles','40poles','40poles','40poles','40poles','40poles','40poles','40poles', \
     '40poles','40poles','40poles','40poles','40poles'])}



df_lpc = pd.DataFrame(data=d_lpc)
     
# Draw a pointplot to show pulse as a function of three categorical factors
g = sns.factorplot(x="clf", y="accuracy", hue="poles", data=df_lpc,
                   capsize=.2, palette="hls", size=6, aspect=.75)
g.despine(left=True)

#%% MFCC

mfcc_2040_knn = [80,77,78,78,82]
mfcc_2040_rf = [79,75,79,76,81]
mfcc_2040_svm = [79,72,79,74,82]
mfcc_3040_knn = [82,81,80,80,84]
mfcc_3040_rf = [80,79,82,75,84]
mfcc_3040_svm = [81,75,80,76,82]
mfcc_4040_knn = [82,82,81,82,85]
mfcc_4040_rf = [81,79,82,75,83]
mfcc_4040_svm = [81,77,80,77,83]

d_mfcc = {'accuracy' : pd.Series(np.concatenate((mfcc_2040_knn,mfcc_2040_rf,mfcc_2040_svm,mfcc_3040_knn,mfcc_3040_rf,mfcc_3040_svm,mfcc_4040_knn,mfcc_4040_rf,mfcc_4040_svm))),
     'artist': pd.Series(['ulla','pablo','emma','juan','claire','ulla','pablo','emma','juan','claire','ulla','pablo','emma','juan','claire', \
     'ulla','pablo','emma','juan','claire','ulla','pablo','emma','juan','claire','ulla','pablo','emma','juan','claire', \
     'ulla','pablo','emma','juan','claire','ulla','pablo','emma','juan','claire','ulla','pablo','emma','juan','claire']),
     'clf': pd.Series(['knn','knn','knn','knn','knn','rf','rf','rf','rf','rf','svm','svm','svm','svm','svm', \
     'knn','knn','knn','knn','knn','rf','rf','rf','rf','rf','svm','svm','svm','svm','svm', \
     'knn','knn','knn','knn','knn','rf','rf','rf','rf','rf','svm','svm','svm','svm','svm']),
     'params': pd.Series(['20coeff-40bands','20coeff-40bands','20coeff-40bands','20coeff-40bands','20coeff-40bands','20coeff-40bands', \
     '20coeff-40bands','20coeff-40bands','20coeff-40bands','20coeff-40bands','20coeff-40bands','20coeff-40bands','20coeff-40bands', \
     '20coeff-40bands','20coeff-40bands','30coeff-40bands','30coeff-40bands','30coeff-40bands','30coeff-40bands','30coeff-40bands', \
     '30coeff-40bands','30coeff-40bands','30coeff-40bands','30coeff-40bands','30coeff-40bands','30coeff-40bands','30coeff-40bands',\
     '30coeff-40bands','30coeff-40bands','30coeff-40bands','40coeff-40bands','40coeff-40bands','40coeff-40bands','40coeff-40bands', \
     '40coeff-40bands','40coeff-40bands','40coeff-40bands','40coeff-40bands','40coeff-40bands','40coeff-40bands','40coeff-40bands',\
     '40coeff-40bands','40coeff-40bands','40coeff-40bands','40coeff-40bands'])}



df_mfcc = pd.DataFrame(data=d_mfcc)
     
# Draw a pointplot to show pulse as a function of three categorical factors
g = sns.factorplot(x="clf", y="accuracy", hue="params", data=df_mfcc,
                   capsize=.2, palette="hls", size=6, aspect=.75)
g.despine(left=True)

# Draw a pointplot to show pulse as a function of three categorical factors
g = sns.factorplot(x="artist", y="accuracy", hue="params", data=df_mfcc,
                   capsize=.2, palette="hls", size=6, aspect=.75)
g.despine(left=True)


#%% SPECTRAL

spectral_256128_knn = [70,76,81,71,78]
spectral_256128_rf = [68,76,80,71,77]
spectral_256128_svm = [74,70,80,73,75]
spectral_1024512_knn = [69,76,80,72,78]
spectral_1024512_rf = [68,75,80,71,77]
spectral_1024512_svm = [74,70,79,73,75]
spectral_20481024_knn = [70,74,78,71,77]
spectral_20481024_rf = [69,74,77,70,77]
spectral_20481024_svm = [72,69,77,70,74]

d_spectral = {'accuracy' : pd.Series(np.concatenate((spectral_256128_knn,spectral_256128_rf,spectral_256128_svm,spectral_1024512_knn,spectral_1024512_rf,spectral_1024512_svm,spectral_20481024_knn,spectral_20481024_rf,spectral_20481024_svm))),
     'artist': pd.Series(['ulla','pablo','emma','juan','claire','ulla','pablo','emma','juan','claire','ulla','pablo','emma','juan','claire', \
     'ulla','pablo','emma','juan','claire','ulla','pablo','emma','juan','claire','ulla','pablo','emma','juan','claire', \
     'ulla','pablo','emma','juan','claire','ulla','pablo','emma','juan','claire','ulla','pablo','emma','juan','claire']),
     'clf': pd.Series(['knn','knn','knn','knn','knn','rf','rf','rf','rf','rf','svm','svm','svm','svm','svm', \
     'knn','knn','knn','knn','knn','rf','rf','rf','rf','rf','svm','svm','svm','svm','svm', \
     'knn','knn','knn','knn','knn','rf','rf','rf','rf','rf','svm','svm','svm','svm','svm']),
     'params': pd.Series(['256win-128hop','256win-128hop','256win-128hop','256win-128hop','256win-128hop','256win-128hop', \
     '256win-128hop','256win-128hop','256win-128hop','256win-128hop','256win-128hop','256win-128hop','256win-128hop', \
     '256win-128hop','256win-128hop','1024win-512hop','1024win-512hop','1024win-512hop','1024win-512hop','1024win-512hop', \
     '1024win-512hop','1024win-512hop','1024win-512hop','1024win-512hop','1024win-512hop','1024win-512hop','1024win-512hop',\
     '1024win-512hop','1024win-512hop','1024win-512hop','2048win-256hop','2048win-256hop','2048win-256hop','2048win-256hop', \
     '2048win-256hop','2048win-256hop','2048win-256hop','2048win-256hop','2048win-256hop','2048win-256hop','2048win-256hop',\
     '2048win-256hop','2048win-256hop','2048win-256hop','2048win-256hop'])}


df_spectral = pd.DataFrame(data=d_spectral)
     
# Draw a pointplot to show pulse as a function of three categorical factors
g = sns.factorplot(x="clf", y="accuracy", hue="params", data=df_spectral,
                   capsize=.2, palette="hls", size=6, aspect=.75)
g.despine(left=True)

#%% SPECTRAL CONTRAST

spectralcontrast_6_knn = [66,77,78,70,78]
spectralcontrast_6_rf = [71,77,77,68,77]
spectralcontrast_6_svm = [67,75,77,70,80]
spectralcontrast_3_knn = [64,68,73,60,70]
spectralcontrast_3_rf = [63,65,71,60,69]
spectralcontrast_3_svm = [65,73,75,62,71]

d_spectralcontrast = {'accuracy' : pd.Series(np.concatenate((spectralcontrast_3_knn,spectralcontrast_3_rf,spectralcontrast_3_svm,spectralcontrast_6_knn,spectralcontrast_6_rf,spectralcontrast_6_svm))),
     'artist': pd.Series(['ulla','pablo','emma','juan','claire','ulla','pablo','emma','juan','claire','ulla','pablo','emma','juan','claire', \
     'ulla','pablo','emma','juan','claire','ulla','pablo','emma','juan','claire','ulla','pablo','emma','juan','claire']),
     'clf': pd.Series(['knn','knn','knn','knn','knn','rf','rf','rf','rf','rf','svm','svm','svm','svm','svm', \
     'knn','knn','knn','knn','knn','rf','rf','rf','rf','rf','svm','svm','svm','svm','svm']),
     'bands': pd.Series(['3bands','3bands','3bands','3bands','3bands','3bands', \
     '3bands','3bands','3bands','3bands','3bands','3bands','3bands', \
     '3bands','3bands','6bands','6bands','6bands','6bands','6bands', \
     '6bands','6bands','6bands','6bands','6bands','6bands','6bands',\
     '6bands','6bands','6bands'])}


df_spectralcontrast = pd.DataFrame(data=d_spectralcontrast)
     
# Draw a pointplot to show pulse as a function of three categorical factors
g = sns.factorplot(x="clf", y="accuracy", hue="bands", data=df_spectralcontrast,
                   capsize=.2, palette="hls", size=6, aspect=.75)
g.despine(left=True)

# Draw a pointplot to show pulse as a function of three categorical factors
g = sns.factorplot(x="artist", y="accuracy", hue="bands", data=df_spectralcontrast,
                   capsize=.2, palette="hls", size=6, aspect=.75)
g.despine(left=True)

#%% BHC Vs. Breathy

mfcc_bhcbr = np.array([83,84,81,81,77,74,76,78,79,79,76,73,82,82,84])

d_bhcbr = {'accuracy' : pd.Series(mfcc_bhcbr),
     'artist': pd.Series(['ulla','ulla','ulla','pablo','pablo','pablo','emma','emma','emma','juan','juan','juan','claire','claire','claire']),
     'clf': pd.Series(['knn','rf','svm','knn','rf','svm','knn','rf','svm','knn','rf','svm','knn','rf','svm']),
     'params': pd.Series(['20coeff-40bands','20coeff-40bands','20coeff-40bands','20coeff-40bands','20coeff-40bands','20coeff-40bands', \
     '20coeff-40bands','20coeff-40bands','20coeff-40bands','20coeff-40bands','20coeff-40bands','20coeff-40bands','20coeff-40bands', \
     '20coeff-40bands','20coeff-40bands'])}


df_bhcbr = pd.DataFrame(data=d_bhcbr)
     
# Draw a pointplot to show pulse as a function of three categorical factors
g = sns.factorplot(x="clf", y="accuracy", hue="params", data=df_bhcbr,
                   capsize=.2, palette="hls", size=6, aspect=.75)
g.despine(left=True)

# Draw a pointplot to show pulse as a function of three categorical factors
g = sns.factorplot(x="artist", y="accuracy", hue="params", data=df_bhcbr,
                   capsize=.2, palette="hls", size=6, aspect=.75)
g.despine(left=True)

#%% MFCC + Voicing

mfcc_2040 = np.array([80,79,79,77,75,72,78,79,79,78,76,74,82,81,82])
mfccvoicing = np.array([80,74,74,77,76,76,78,81,80,77,74,71,83,82,82])

d_mfccvoicing = {'accuracy' : pd.Series(np.concatenate((mfcc_2040,mfccvoicing))),
     'artist': pd.Series(['ulla','ulla','ulla','pablo','pablo','pablo','emma','emma','emma','juan','juan','juan','claire','claire','claire', \
     'ulla','ulla','ulla','pablo','pablo','pablo','emma','emma','emma','juan','juan','juan','claire','claire','claire']),
     'clf': pd.Series(['knn','rf','svm','knn','rf','svm','knn','rf','svm','knn','rf','svm','knn','rf','svm', \
     'knn','rf','svm','knn','rf','svm','knn','rf','svm','knn','rf','svm','knn','rf','svm',]),
     'params': pd.Series(['20coeff-40bands','20coeff-40bands','20coeff-40bands','20coeff-40bands','20coeff-40bands','20coeff-40bands', \
     '20coeff-40bands','20coeff-40bands','20coeff-40bands','20coeff-40bands','20coeff-40bands','20coeff-40bands','20coeff-40bands', \
     '20coeff-40bands','20coeff-40bands','20coeff-40bands + Voicing','20coeff-40bands + Voicing','20coeff-40bands + Voicing','20coeff-40bands + Voicing', \
     '20coeff-40bands + Voicing','20coeff-40bands + Voicing','20coeff-40bands + Voicing','20coeff-40bands + Voicing','20coeff-40bands + Voicing', \
     '20coeff-40bands + Voicing','20coeff-40bands + Voicing','20coeff-40bands + Voicing','20coeff-40bands + Voicing','20coeff-40bands + Voicing','20coeff-40bands + Voicing'])}


df_mfccvoicing = pd.DataFrame(data=d_mfccvoicing)
     
# Draw a pointplot to show pulse as a function of three categorical factors
g = sns.factorplot(x="clf", y="accuracy", hue="params", data=df_mfccvoicing,
                   capsize=.2, palette="hls", size=6, aspect=.75)
g.despine(left=True)
plt.title('MFCC Vs. MFCC + Voicing')


# Draw a pointplot to show pulse as a function of three categorical factors
g = sns.factorplot(x="artist", y="accuracy", hue="params", data=df_mfccvoicing,
                   capsize=.2, palette="hls", size=6, aspect=.75)
g.despine(left=True)
plt.title('MFCC Vs. MFCC + Voicing')


#%% MFCC REFINATION

mfcc_512256_knn = [80,76,77,77,79]
mfcc_512256_rf = [78,75,78,75,79]
mfcc_512256_svm = [79,73,79,76,80]
mfcc_1024512_knn = [80,77,78,78,82]
mfcc_1024512_rf = [79,75,79,76,81]
mfcc_1024512_svm = [79,72,79,74,82]
mfcc_20481024_knn = [80,78,79,78,82]
mfcc_20481024_rf = [78,76,78,75,81]
mfcc_20481024_svm = [78,72,79,73,82]

d_mfcc_refi = {'accuracy' : pd.Series(np.concatenate((mfcc_512256_knn,mfcc_512256_rf,mfcc_512256_svm,mfcc_1024512_knn,mfcc_1024512_rf,mfcc_1024512_svm,mfcc_20481024_knn,mfcc_20481024_rf,mfcc_20481024_svm))),
     'artist': pd.Series(['ulla','pablo','emma','juan','claire','ulla','pablo','emma','juan','claire','ulla','pablo','emma','juan','claire', \
     'ulla','pablo','emma','juan','claire','ulla','pablo','emma','juan','claire','ulla','pablo','emma','juan','claire', \
     'ulla','pablo','emma','juan','claire','ulla','pablo','emma','juan','claire','ulla','pablo','emma','juan','claire']),
     'clf': pd.Series(['knn','knn','knn','knn','knn','rf','rf','rf','rf','rf','svm','svm','svm','svm','svm', \
     'knn','knn','knn','knn','knn','rf','rf','rf','rf','rf','svm','svm','svm','svm','svm', \
     'knn','knn','knn','knn','knn','rf','rf','rf','rf','rf','svm','svm','svm','svm','svm']),
     'params': pd.Series(['512win-256hop','512win-256hop','512win-256hop','512win-256hop','512win-256hop','512win-256hop', \
     '512win-256hop','512win-256hop','512win-256hop','512win-256hop','512win-256hop','512win-256hop','512win-256hop', \
     '512win-256hop','512win-256hop','1024win-512hop','1024win-512hop','1024win-512hop','1024win-512hop','1024win-512hop', \
     '1024win-512hop','1024win-512hop','1024win-512hop','1024win-512hop','1024win-512hop','1024win-512hop','1024win-512hop',\
     '1024win-512hop','1024win-512hop','1024win-512hop','2048win-1024hop','2048win-1024hop','2048win-1024hop','2048win-1024hop', \
     '2048win-1024hop','2048win-1024hop','2048win-1024hop','2048win-1024hop','2048win-1024hop','2048win-1024hop','2048win-1024hop',\
     '2048win-1024hop','2048win-1024hop','2048win-1024hop','2048win-1024hop'])}

df_mfcc_refi = pd.DataFrame(data=d_mfcc_refi)
     
# Draw a pointplot to show pulse as a function of three categorical factors
g = sns.factorplot(x="clf", y="accuracy", hue="params", data=df_mfcc_refi,
                   capsize=.2, palette="hls", size=6, aspect=.75)
g.despine(left=True)

# Draw a pointplot to show pulse as a function of three categorical factors
g = sns.factorplot(x="artist", y="accuracy", hue="params", data=df_mfcc_refi,
                   capsize=.2, palette="hls", size=6, aspect=.75)
g.despine(left=True)