# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 11:04:49 2021

@author: Ashwini Ojha
"""

import os
import sys
from sklearn.utils import shuffle

# orgy,forged = BHSig260('Datasets/BHSig260/Hindi/')
# orgy,forged = Cedar('Datasets/cedar_dataset')
# k = generate_batch(orgy,forged)


def BHSig260(path):
    dir_list = next(os.walk(path))[1]
    dir_list.sort()    
    original = []
    forged = []
    for directory in dir_list:
        images = os.listdir(path+directory)
        images.sort()
        images = [path+directory+'/'+x for x in images]
        forged.append(images[:30]) # First 30 signatures in each folder are forged
        original.append(images[30:]) # Next 24 signatures are genuine
    
    return original, forged


def Cedar(path):
    dir_original_list = os.listdir(path + '/full_org')
    dir_original_list = [path+'/full_org'+'/'+x for x in dir_original_list]
    dir_forged_list = os.listdir(path + '/full_forg')
    dir_forged_list = [path+'/full_forg'+'/'+x for x in dir_forged_list]
    
    original = []
    forged = [] 
    i=0
    while i < (len(dir_original_list)):
        original.append(dir_original_list[i:i+24])
        forged.append(dir_forged_list[i:i+24])
        i = i+24
        
    return original, forged

    
