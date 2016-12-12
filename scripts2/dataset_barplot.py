# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 11:30:30 2016

@author: Juan
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy import signal
import pandas as pd
import seaborn as sns

melcoeff = 20
melbands = 40

prediction_file_pablo="../prediction/pablo_mfcc_" + str(melcoeff) + str(melbands) + "_prediction.npy"  
test_pablo=np.load(prediction_file_pablo)

prediction_file_ulla="../prediction/ulla_mfcc_" + str(melcoeff) + str(melbands) + "_prediction.npy"  
test_ulla=np.load(prediction_file_ulla)

prediction_file_emma="../prediction/emma_mfcc_" + str(melcoeff) + str(melbands) + "_prediction.npy"  
test_emma=np.load(prediction_file_emma)

prediction_file_juan="../prediction/juan_mfcc_" + str(melcoeff) + str(melbands) + "_prediction.npy"  
test_juan=np.load(prediction_file_juan)

prediction_file_claire="../prediction/claire_mfcc_" + str(melcoeff) + str(melbands) + "_prediction.npy"  
test_claire=np.load(prediction_file_claire)

gt_pablo = test_pablo[:,1]
gt_ulla = test_ulla[:,1]
gt_emma = test_emma[:,1]
gt_juan = test_juan[:,1]
gt_claire = test_claire[:,1]

#%%
N = 5
bhc = (len(np.argwhere(gt_ulla==1)), len(np.argwhere(gt_pablo==1)), \
len(np.argwhere(gt_emma==1)), len(np.argwhere(gt_juan==1)), len(np.argwhere(gt_claire==1)))
breathy = (len(np.argwhere(gt_ulla==2)), len(np.argwhere(gt_pablo==2)), \
len(np.argwhere(gt_emma==2)), len(np.argwhere(gt_juan==2)), len(np.argwhere(gt_claire==2)))
normal = (len(np.argwhere(gt_ulla==3)), len(np.argwhere(gt_pablo==3)), \
len(np.argwhere(gt_emma==3)), len(np.argwhere(gt_juan==3)), len(np.argwhere(gt_claire==3)))
ind = np.arange(N)    # the x locations for the groups
width = 0.50           # the width of the bars: can also be len(x) sequence

p1 = plt.barh(ind, bhc, width, color='r')
p2 = plt.barh(ind, breathy, width, color='y',
             left=bhc)
bottomp3=np.array(breathy)+np.array(bhc)
p3 = plt.barh(ind, normal, width, color='g', left=bottomp3)

plt.xlabel('Cantidad de frames')
plt.title('Bag of Frames de Aliento/Arrugas')
plt.yticks(ind + width/2., ('ulla', 'pablo', 'emma', 'juan', 'claire'))
plt.xticks(np.arange(0, 25000, 1000))
plt.legend((p1[0], p2[0], p3[0]), ('Blow Hole Covert', 'Breathy', 'Normal'))

plt.show()