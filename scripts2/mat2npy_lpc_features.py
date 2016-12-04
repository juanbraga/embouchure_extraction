# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 17:26:12 2016

@author: Juan
"""

import scipy.io as io
import numpy as np
#import sys

#if __name__=='__main__':
    
def convert(filename, autor):
    
#    filename=sys.argv[1]
    test_aux = filename + '_test.mat'
    train_aux = filename + '_train.mat'    
    test=io.loadmat(test_aux)
    train=io.loadmat(train_aux)
    
#    autor = sys.argv[2]
    dict_aux_test = autor + '_test'
    dict_aux_train = autor + '_train'

    filename_save = filename + '_test'
    np.save(filename_save, test[dict_aux_test])
    filename_save = filename + '_train'    
    np.save(filename_save, train[dict_aux_train])
    
    
