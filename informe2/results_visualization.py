# -*- coding: utf-8 -*-
"""
Created on Mon Dec 05 20:22:02 2016

@author: Juan
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas

plt.close('all')

mfcc_20_40 = np.array([79,71,72,76,74,73,77,80,79,77,72,70,82,81,82])
mfcc_30_40 = np.array([82,72,77,79,78,75,80,82,80,80,75,76,84,84,82])
mfcc_40_40 = np.array([82,74,77,81,79,76,81,82,80,82,75,77,85,83,83])

d_mfcc = {'mfcc_20_40': mfcc_20_40, 'mfcc_30_40': mfcc_30_40, 'mfcc_40_40': mfcc_40_40}
df_mfcc = pandas.DataFrame(data=d_mfcc)

speccont_3 = np.array([64,63,65,68,65,73,73,71,75,60,60,62,70,69,71])
speccont_6 = np.array([66,71,67,77,77,75,78,77,77,70,68,70,78,77,80])

d_speccon = {'speccont_3': speccont_3, 'speccont_6': speccont_6}
df_speccon = pandas.DataFrame(data=d_speccon)

spectral_256128 = np.array([70,69,72,74,74,69,78,77,77,71,70,70,77,77,74])
spectral_1024512 = np.array([69,68,74,76,75,70,80,80,79,72,71,73,78,77,75])
spectral_20481024 = np.array([70,68,74,76,76,70,81,80,80,71,71,73,78,77,75])

d_spectral = {'spectral_256128': spectral_256128, 'spectral_1024512': spectral_1024512, 'spectral_20481024': spectral_20481024}
df_spectral = pandas.DataFrame(data=d_spectral)

lpc_10 = np.array([47,43,50,52,56,70,55,57,70,42,40,56,55,54,72])
lpc_20 = np.array([47,45,52,52,52,72,51,51,70,46,44,45,52,52,64])
lpc_40 = np.array([49,45,55,52,52,69,56,54,69,53,45,48,56,51,67])

d_lpc = {'lpc_10': lpc_10, 'lpc_20': lpc_20, 'lpc_40': lpc_40}
df_lpc = pandas.DataFrame(data=d_lpc)

plt.figure()
plt.subplot(1,4,1)
sns.boxplot(data=df_mfcc)
plt.ylim( (35, 90) )
plt.subplot(1,4,2)
sns.boxplot(data=df_spectral)
plt.ylim( (35, 90) )
plt.subplot(1,4,3)
sns.boxplot(data=df_speccon)
plt.ylim( (35, 90) )
plt.subplot(1,4,4)
sns.boxplot(data=df_lpc)
plt.ylim( (35, 90) )

mfcc_bhcbr = np.array([80,72,74,78,74,75,76,76,77,78,72,64,82,82,82])
speccont_bhcbr = np.array([67,70,57,78,76,78,77,77,78,69,68,64,77,77,79])

d_bhcbr = {'mfcc_bhcbr': mfcc_bhcbr, 'speccont_bhcbr': speccont_bhcbr}
df_bhcbr = pandas.DataFrame(data=d_bhcbr)

plt.figure()
sns.boxplot(data=df_bhcbr)
plt.ylim( (50, 90) )

mfccvoicing = np.array([80,72,74,78,74,75,76,76,77,78,72,64,82,82,82])

d_plus_voicing = {'mfcc_20_40': mfcc_20_40, 'mfccvoicing': mfccvoicing}
df_plus_voicing = pandas.DataFrame(data=d_plus_voicing)

plt.figure()
sns.boxplot(data=df_plus_voicing)
plt.ylim( (50, 90) )

#%%
features_spectral="../features/" + 'juan' + "_spectral_" + str(256) + str(128) + "_train.npy"
data_spectral=np.load(features_spectral)
d_spectral = {'rolloff': data_spectral[0,:], 'centroid': data_spectral[1,:], 'bandwith': data_spectral[2,:], 'zcr': data_spectral[3,:], 'voicing': data_spectral[4,:], 'gt': data_spectral[5,:]}
df_spectral = pandas.DataFrame(data=d_spectral)
sns.pairplot(df_spectral, hue='gt', vars=['voicing', 'zcr','centroid'])

#%%
melcoeff=20
melbands=40
probabilties="../prediction/" + 'pablo' + "_mfcc_" + str(melcoeff) + str(melbands) + "_proba.npy"
proba=np.load(probabilties)
proba=proba.T
d_proba = {'bhc': proba[0,:], 'breathy': proba[1,:], 'normal': proba[2,:], 'time': proba[3,:], 'gt': proba[4,:]}
df_proba = pandas.DataFrame(data=d_proba)
sns.pairplot(df_proba, hue='gt', vars=['bhc', 'breathy','normal'])



