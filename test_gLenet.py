from __future__ import print_function

import sys
import os
import time

import cPickle as pickle
import numpy as np
import theano
import theano.tensor as T

import lasagne

fid = open('/home/habtegebrial/Desktop/Academic/Sem_III/Lasagne/pre_trained_models/blvc_googlenet.pkl','r')
network = pickle.load(fid)
fid.close()

print(network)
