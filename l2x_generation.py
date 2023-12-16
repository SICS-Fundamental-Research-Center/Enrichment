from __future__ import print_function
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle as pkl
from collections import defaultdict
import re 
from bs4 import BeautifulSoup 
import sys
import os
import time
from keras.callbacks import ModelCheckpoint    
from keras.layers import Dense, Input, Flatten, Add, Multiply, Lambda
# from keras.layers.normalization import BatchNormalization
from keras.layers.normalization.batch_normalization_v1 import BatchNormalization
from keras.models import Model, Sequential
from keras import regularizers
from keras import backend as K
# from keras.engine.topology import Layer 
from tensorflow.python.keras.layers import Layer, InputSpec
import json
import random
from keras import optimizers
import tensorflow.keras.backend as K
import numpy as np
import random
BATCH_SIZE = 5000
np.random.seed(0)
# tf.random.seed(0)
random.seed(0)
# The number of key features for each data set.
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str,
                    choices=["persons", "imdb", "amazon", "dblp","imdb-1hop","imdb-3hop"], default="persons")
parser.add_argument("--max_path", type=int, default=5)
main_args = parser.parse_args()
datatype = main_args.data_name
m = main_args.max_path

ks = {'orange_skin': 4, 'XOR': 2, 'nonlinear_additive': 4, 'switch': 5, 'persons': 4,'imdb' : 2,'imdb-1hop' : 2,
      'amazon' : 5,'imdb-3hop' : 5,'dblp' : 5} ## Modify it, the data_name/orange_skin should be the same


import numpy as np  
from scipy.stats import chi2
def create_data_modify(datatype):
    if(datatype=='persons'):
        train = pd.read_csv('data/l2x-similarity/wiki_train.csv',index_col=0)
        valid = pd.read_csv('data/l2x-similarity/wiki_valid.csv',index_col=0)
        x_train = np.array(train.iloc[:,:-1]).astype(float)
        y_train = np.array(train.iloc[:,-1]).astype(int)
        y_train_minus = 1 - y_train
        
        x_valid = np.array(valid.iloc[:,:-1]).astype(float)
        y_valid = np.array(valid.iloc[:,-1]).astype(int)
        y_valid_minus = 1 - y_valid
        input_shape = x_train.shape[1]
        return x_train,np.column_stack((y_train, y_train_minus)),x_valid,np.column_stack((y_valid, y_valid_minus)),datatype,input_shape
    elif(datatype=='imdb'):
        train = pd.read_csv('data/l2x-similarity/imdb_train.csv',index_col=0)
        valid = pd.read_csv('data/l2x-similarity/imdb_valid.csv',index_col=0)
        x_train = np.array(train.iloc[:,:-1]).astype(float)
        y_train = np.array(train.iloc[:,-1]).astype(int)
        y_train_minus = 1 - y_train
        
        x_valid = np.array(valid.iloc[:,:-1]).astype(float)
        y_valid = np.array(valid.iloc[:,-1]).astype(int)
        y_valid_minus = 1 - y_valid
        input_shape = x_train.shape[1]
        return x_train,np.column_stack((y_train, y_train_minus)),x_valid,np.column_stack((y_valid, y_valid_minus)),datatype,input_shape
    elif(datatype=='imdb-3hop'):
        train = pd.read_csv('data/l2x-similarity/imdb_train_3hop.csv',index_col=0)
        valid = pd.read_csv('data/l2x-similarity/imdb_valid_3hop.csv',index_col=0)
        x_train = np.array(train.iloc[:,:-1]).astype(float)
        y_train = np.array(train.iloc[:,-1]).astype(int)
        y_train_minus = 1 - y_train
        
        x_valid = np.array(valid.iloc[:,:-1]).astype(float)
        y_valid = np.array(valid.iloc[:,-1]).astype(int)
        y_valid_minus = 1 - y_valid
        input_shape = x_train.shape[1]
        return x_train,np.column_stack((y_train, y_train_minus)),x_valid,np.column_stack((y_valid, y_valid_minus)),datatype,input_shape
    elif(datatype=='imdb-1hop'):
        train = pd.read_csv('data/l2x-similarity/imdb_train.csv',index_col=0)
        valid = pd.read_csv('data/l2x-similarity/imdb_valid.csv',index_col=0)
        x_train = np.array(train.iloc[:,:18]).astype(float)
        y_train = np.array(train.iloc[:,-1]).astype(int)
        y_train_minus = 1 - y_train
        
        x_valid = np.array(valid.iloc[:,:18]).astype(float)
        y_valid = np.array(valid.iloc[:,-1]).astype(int)
        y_valid_minus = 1 - y_valid
        input_shape = x_train.shape[1]
        return x_train,np.column_stack((y_train, y_train_minus)),x_valid,np.column_stack((y_valid, y_valid_minus)),datatype,input_shape
    elif(datatype=='amazon'):
        train = pd.read_csv('data/l2x-similarity/amazon_train.csv',index_col=0)
        valid = pd.read_csv('data/l2x-similarity/amazon_valid.csv',index_col=0)
        x_train = np.array(train.iloc[:,:-1]).astype(float)
        y_train = np.array(train.iloc[:,-1]).astype(int)
        y_train_minus = 1 - y_train
        
        x_valid = np.array(valid.iloc[:,:-1]).astype(float)
        y_valid = np.array(valid.iloc[:,-1]).astype(int)
        y_valid_minus = 1 - y_valid
        input_shape = x_train.shape[1]
        return x_train,np.column_stack((y_train, y_train_minus)),x_valid,np.column_stack((y_valid, y_valid_minus)),datatype,input_shape
    elif(datatype=='dblp'):
        train = pd.read_csv('data/l2x-similarity/dblp_train.csv',index_col=0)
        valid = pd.read_csv('data/l2x-similarity/dblp_valid.csv',index_col=0)
        x_train = np.array(train.iloc[:,:-1]).astype(float)
        y_train = np.array(train.iloc[:,-1]).astype(int)
        y_train_minus = 1 - y_train
        
        x_valid = np.array(valid.iloc[:,:-1]).astype(float)
        y_valid = np.array(valid.iloc[:,-1]).astype(int)
        y_valid_minus = 1 - y_valid
        input_shape = x_train.shape[1]
        return x_train,np.column_stack((y_train, y_train_minus)),x_valid,np.column_stack((y_valid, y_valid_minus)),datatype,input_shape
def generate_XOR_labels(X):
    y = np.exp(X[:,0]*X[:,1])

    prob_1 = np.expand_dims(1 / (1+y) ,1)
    prob_0 = np.expand_dims(y / (1+y) ,1)

    y = np.concatenate((prob_0,prob_1), axis = 1)

    return y

def generate_orange_labels(X):
    logit = np.exp(np.sum(X[:,:4]**2, axis = 1) - 4.0) 

    prob_1 = np.expand_dims(1 / (1+logit) ,1)
    prob_0 = np.expand_dims(logit / (1+logit) ,1)

    y = np.concatenate((prob_0,prob_1), axis = 1)

    return y

def generate_additive_labels(X):
    logit = np.exp(-100 * np.sin(0.2*X[:,0]) + abs(X[:,1]) + X[:,2] + np.exp(-X[:,3])  - 2.4) 

    prob_1 = np.expand_dims(1 / (1+logit) ,1)
    prob_0 = np.expand_dims(logit / (1+logit) ,1)

    y = np.concatenate((prob_0,prob_1), axis = 1)

    return y



def generate_data(n=100, datatype='', seed = 0, val = False):
    """
    Generate data (X,y)
    Args:
        n(int): number of samples 
        datatype(string): The type of data 
        choices: 'orange_skin', 'XOR', 'regression'.
        seed: random seed used
    Return: 
        X(float): [n,d].  
        y(float): n dimensional array. 
    """

    np.random.seed(seed)

    X = np.random.randn(n, 10)

    datatypes = None 

    if datatype == 'orange_skin': 
        y = generate_orange_labels(X) 

    elif datatype == 'XOR':
        y = generate_XOR_labels(X)    

    elif datatype == 'nonlinear_additive':  
        y = generate_additive_labels(X) 

    elif datatype == 'switch':

        # Construct X as a mixture of two Gaussians.
        X[:n//2,-1] += 3
        X[n//2:,-1] += -3
        X1 = X[:n//2]; X2 = X[n//2:]

        y1 = generate_orange_labels(X1)
        y2 = generate_additive_labels(X2)

        # Set the key features of X2 to be the 4-8th features.
        X2[:,4:8],X2[:,:4] = X2[:,:4],X2[:,4:8]

        X = np.concatenate([X1,X2], axis = 0)
        y = np.concatenate([y1,y2], axis = 0) 

        # Used for evaluation purposes.
        datatypes = np.array(['orange_skin'] * len(y1) + ['nonlinear_additive'] * len(y2)) 

        # Permute the instances randomly.
        perm_inds = np.random.permutation(n)
        X,y = X[perm_inds],y[perm_inds]
        datatypes = datatypes[perm_inds]


    return X, y, datatypes

def create_data(datatype, n = 1000): 
	"""
	Create train and validation datasets.

	"""
# 
	x_train, y_train, _ = generate_data(n = n, 
		datatype = datatype, seed = 0)  
	x_val, y_val, datatypes_val = generate_data(n = 10 ** 5, 
		datatype = datatype, seed = 1)  

	input_shape = x_train.shape[1]
	return x_train,y_train,x_val,y_val,datatypes_val, input_shape

def create_rank(scores, k): 
	"""
	Compute rank of each feature based on weight.
	
	"""
	scores = abs(scores)
	n, d = scores.shape
	ranks = []
	for i, score in enumerate(scores):
		# Random permutation to avoid bias due to equal weights.
		idx = np.random.permutation(d) 
		permutated_weights = score[idx]  
		permutated_rank=(-permutated_weights).argsort().argsort()+1
		rank = permutated_rank[np.argsort(idx)]

		ranks.append(rank)

	return np.array(ranks)

def compute_median_rank(scores, k, datatype_val = None):
	ranks = create_rank(scores, k)
	if datatype_val is None: 
		median_ranks = np.median(ranks[:,:k], axis = 1)
	else:
		datatype_val = datatype_val[:len(scores)]
		median_ranks1 = np.median(ranks[datatype_val == 'orange_skin',:][:,np.array([0,1,2,3,9])], 
			axis = 1)
		median_ranks2 = np.median(ranks[datatype_val == 'nonlinear_additive',:][:,np.array([4,5,6,7,9])], 
			axis = 1)
		median_ranks = np.concatenate((median_ranks1,median_ranks2), 0)
	return median_ranks 

class Sample_Concrete(Layer):
	"""
	Layer for sample Concrete / Gumbel-Softmax variables. 

	"""
	def __init__(self, tau0, k, **kwargs): 
		self.tau0 = tau0
		self.k = k
		super(Sample_Concrete, self).__init__(**kwargs)

	def call(self, logits):   
		# logits: [BATCH_SIZE, d]
		logits_ = K.expand_dims(logits, -2)# [BATCH_SIZE, 1, d]

		batch_size = tf.shape(logits_)[0]
		d = tf.shape(logits_)[2]
		uniform = tf.random.uniform(shape =(batch_size, self.k, d), 
			minval = np.finfo(tf.float32.as_numpy_dtype).tiny,
			maxval = 1.0)

		gumbel = - K.log(-K.log(uniform))
		noisy_logits = (gumbel + logits_)/self.tau0
		samples = K.softmax(noisy_logits)
		samples = K.max(samples, axis = 1) 

		# Explanation Stage output.
		threshold = tf.expand_dims(tf.nn.top_k(logits, self.k, sorted = True)[0][:,-1], -1)
		discrete_logits = tf.cast(tf.greater_equal(logits,threshold),tf.float32)
		
		return K.in_train_phase(samples, discrete_logits)

	def compute_output_shape(self, input_shape):
		return input_shape 



def L2X(datatype, train = True,epoch=1): 
	# x_train,y_train,x_val,y_val,datatype_val, input_shape = create_data(datatype, 
	# 	n = int(1e6))
	x_train,y_train,x_val,y_val, datatype_val,input_shape = create_data_modify(datatype)
	datatype_val = 'orange_skin'
	st1 = time.time()
	st2 = st1

	activation = 'relu' if datatype in ['orange_skin','XOR','wiki'] else 'selu'
	# P(S|X)
	model_input = Input(shape=(input_shape,), dtype='float32') 

	net = Dense(100, activation=activation, name = 's/dense1',
		kernel_regularizer=regularizers.l2(1e-3))(model_input)
	net = Dense(100, activation=activation, name = 's/dense2',
		kernel_regularizer=regularizers.l2(1e-3))(net) 

	# A tensor of shape, [batch_size, max_sents, 100]
	logits = Dense(input_shape)(net) 
	# [BATCH_SIZE, max_sents, 1]  
	k = ks[datatype]; tau = 0.1
	samples = Sample_Concrete(tau, k, name = 'sample')(logits)
 
	# q(X_S)
	new_model_input = Multiply()([model_input, samples]) 
	net = Dense(200, activation=activation, name = 'dense1',
		kernel_regularizer=regularizers.l2(1e-3))(new_model_input) 
	net = BatchNormalization()(net) # Add batchnorm for stability.
	net = Dense(200, activation=activation, name = 'dense2',
		kernel_regularizer=regularizers.l2(1e-3))(net)
	net = BatchNormalization()(net)

	preds = Dense(2, activation='softmax', name = 'dense4',
		kernel_regularizer=regularizers.l2(1e-3))(net) 
	model = Model(model_input, preds)

	if train: 
		adam = optimizers.Adam(lr = 1e-3)
		model.compile(loss='categorical_crossentropy',
					  optimizer=adam,
					  metrics=['acc']) 
		filepath="models/{}/L2X.hdf5".format(datatype)
		checkpoint = ModelCheckpoint(filepath, monitor='val_acc', 
			verbose=1, save_best_only=True, mode='max')
		callbacks_list = [checkpoint]
		model.fit(x_train, y_train, validation_data=(x_val, y_val),callbacks = callbacks_list, epochs=1, batch_size=BATCH_SIZE)
		st2 = time.time() 
	else:
		model.load_weights('models/{}/L2X.hdf5'.format(datatype), 
			by_name=True) 


	pred_model = Model(model_input, samples)
	pred_model.compile(loss=None,
				  optimizer='rmsprop',
				  metrics=['acc']) 

	scores = pred_model.predict(x_val, verbose = 1, batch_size = BATCH_SIZE) 
	print(scores.shape)
	median_ranks = compute_median_rank(scores, k = ks[datatype],
		datatype_val=datatype_val)

	return median_ranks, time.time() - st2, st2 - st1, scores

median_ranks, exp_time, train_time, samples = L2X(datatype, 
		train = True, epoch=1)


arr = samples
m = 4
# random.seed(42)
column_sums = np.sum(arr, axis=0)


top_indices = np.argpartition(column_sums, -m)[-m:]


if len(top_indices) > m:
    top_indices = random.sample(list(top_indices), m)

print(top_indices)