import os
import time

import cPickle as pickle

import numpy as np
import theano
import theano.tensor as T

import lasagne

filename = '/home/habtegebrial/Desktop/Academic/Sem_III/Lasagne/Lasagne-master/examples/mnist.pkl'
fid = open(filename, 'rb')
train_set, val_set, test_set = pickle.load(fid)
fid.close()

img1 = train_set[0][0][:]

input_var = T.tensor4('inputs')
target_var = T.ivector('targets')
network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=input_var)

