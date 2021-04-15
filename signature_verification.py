# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 22:37:12 2021

@author: Ashwini Ojha
"""

import sys
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import cv2
import time
import itertools
import random
from Preprocessing.image_processing import * 
from Preprocessing.data_reading import * 

import tensorflow as tf
from keras.optimizers import Adam, RMSprop
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate, Dropout
from keras.models import Model, Sequential

from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform

from keras.engine.topology import Layer
from keras.regularizers import l2
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def generate_batch(orig_groups, forg_groups,img_h,img_w,batch_size = 32):
    '''Function to generate a batch of data with batch_size number of data points
    Half of the data points will be Genuine-Genuine pairs and half will be Genuine-Forged pairs'''
    while True:
        orig_pairs = []
        forg_pairs = []
        gen_gen_labels = []
        gen_for_labels = []
        all_pairs = []
        all_labels = []
        
        # Here we create pairs of Genuine-Genuine image names and Genuine-Forged image names
        # For every person we have 24 genuine signatures, hence we have 
        # 24 choose 2 = 276 Genuine-Genuine image pairs for one person.
        # To make Genuine-Forged pairs, we pair every Genuine signature of a person
        # with 12 randomly sampled Forged signatures of the same person.
        # Thus we make 24 * 12 = 300 Genuine-Forged image pairs for one person.
        # In all we have 120 person's data in the training data.
        # Total no. of Genuine-Genuine pairs = 120 * 276 = 33120
        # Total number of Genuine-Forged pairs = 120 * 300 = 36000
        # Total no. of data points = 33120 + 36000 = 69120
        for orig, forg in zip(orig_groups, forg_groups):
            orig_pairs.extend(list(itertools.combinations(orig, 2)))
            for i in range(len(forg)):
                forg_pairs.extend(list(itertools.product(orig[i:i+1], random.sample(forg, 12))))
        
        # Label for Genuine-Genuine pairs is 1
        # Label for Genuine-Forged pairs is 0
        gen_gen_labels = [1]*len(orig_pairs)
        gen_for_labels = [0]*len(forg_pairs)
        
        # Concatenate all the pairs together along with their labels and shuffle them
        all_pairs = orig_pairs + forg_pairs
        all_labels = gen_gen_labels + gen_for_labels
        del orig_pairs, forg_pairs, gen_gen_labels, gen_for_labels
        all_pairs, all_labels = shuffle(all_pairs, all_labels)
        
        # Note the lists above contain only the image names and
        # actual images are loaded and yielded below in batches
        # Below we prepare a batch of data points and yield the batch
        # In each batch we load "batch_size" number of image pairs
        # These images are then removed from the original set so that
        # they are not added again in the next batch.
            
        k = 0
        pairs=[np.zeros((batch_size, img_h, img_w, 1)) for i in range(2)]
        targets=np.zeros((batch_size,))
        for ix, pair in enumerate(all_pairs):
            img1 = cv2.imread(pair[0], 0)
            img2 = cv2.imread(pair[1], 0)
            # Image processing
            # img1 = preprocess(img1)
            # img2 = preprocess(img2)
            img1 = cv2.resize(img1, (img_w, img_h)) # Check for different interpolation method
            img2 = cv2.resize(img2, (img_w, img_h))
            img1 = np.array(img1, dtype = np.float64)
            img2 = np.array(img2, dtype = np.float64)
            img1 /= 255
            img2 /= 255
            img1 = img1[..., np.newaxis]
            img2 = img2[..., np.newaxis]
            pairs[0][k, :, :, :] = img1
            pairs[1][k, :, :, :] = img2
            targets[k] = all_labels[ix]
            k += 1
            if k == batch_size:
                yield pairs, targets
                k = 0
                pairs=[np.zeros((batch_size, img_h, img_w, 1)) for i in range(2)]
                targets=np.zeros((batch_size,))

def network_architecture(input_shape):
    model = Sequential()
    model.add(Conv2D(96, kernel_size=(11, 11), activation='relu', name='conv_layer_1', strides=(1,1),  kernel_initializer="glorot_uniform", input_shape= input_shape))
    model.add(BatchNormalization(epsilon=1e-06, axis=1, momentum=0.9))
    model.add(MaxPooling2D((3,3), strides=(2, 2)))    
    model.add(ZeroPadding2D((2, 2)))
    
    model.add(Conv2D(256, kernel_size=(5, 5), activation='relu', name='conv_layer_2', strides=(1,1) ,kernel_initializer="glorot_uniform"))
    model.add(BatchNormalization(epsilon=1e-06,axis=1, momentum=0.9))
    model.add(MaxPooling2D((3,3), strides=(2, 2)))
    model.add(Dropout(0.3))# added extra
    model.add(ZeroPadding2D((1, 1)))
    
    model.add(Conv2D(384, kernel_size=(3, 3), activation='relu', name='conv_layer_3', strides=(1,1),kernel_initializer="glorot_uniform"))
    model.add(ZeroPadding2D((1, 1)))
    
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', name='conv_layer_4', strides=(1,1), kernel_initializer="glorot_uniform"))    
    model.add(MaxPooling2D((3,3), strides=(2, 2)))
    model.add(Dropout(0.3))# added extra
    
    model.add(Flatten(name='flatten'))
    model.add(Dense(1024, kernel_regularizer=l2(0.0005), activation='relu',kernel_initializer="glorot_uniform"))
    model.add(Dropout(0.5))
    
    model.add(Dense(128, kernel_regularizer=l2(0.0005), activation='relu',kernel_initializer="glorot_uniform"))
    
    return model

def euclidean_distance(vects):
    '''Compute Euclidean Distance between two vectors'''
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


img_h, img_w = 155, 220
input_shape=(img_h, img_w, 1)

original_list, forged_list = Cedar('Datasets/cedar_dataset')

base_network = network_architecture(input_shape)
input_a = Input(shape=(input_shape))
input_b = Input(shape=(input_shape))

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
network_a = base_network(input_a)
network_b = base_network(input_b)





# Compute the Euclidean distance between the two vectors in the latent space
distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([network_a, network_b])
model = Model(inputs=[input_a, input_b], outputs=distance)

# compile model using RMSProp Optimizer and Contrastive loss function defined above
rms = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-08)
model.compile(loss=contrastive_loss, optimizer=rms)

# Using Keras Callbacks, save the model after every epoch
# Reduce the learning rate by a factor of 0.1 if the validation loss does not improve for 5 epochs
# Stop the training using early stopping if the validation loss does not improve for 12 epochs
callbacks = [
    EarlyStopping(patience=12, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.000001, verbose=1),
    ModelCheckpoint('./Weights/signet-bhsig260-{epoch:03d}.h5', verbose=1, save_weights_only=True)
]

orig_train, orig_val, orig_test = original_list[:120], original_list[120:140], original_list[140:]
forg_train, forg_val, forg_test = forged_list[:120], forged_list[120:140], forged_list[140:]


batch_sz = 3
num_train_samples = 276*120 + 300*120
num_val_samples = num_test_samples = 276*20 + 300*20
num_train_samples, num_val_samples, num_test_samples

results = model.fit(generate_batch(orig_train, forg_train,img_h,img_w, batch_sz),
                              steps_per_epoch = num_train_samples//batch_sz,
                              epochs = 100,
                              validation_data = generate_batch(orig_val, forg_val,img_h,img_w, batch_sz),
                              validation_steps = num_val_samples//batch_sz,
                              callbacks = callbacks)
