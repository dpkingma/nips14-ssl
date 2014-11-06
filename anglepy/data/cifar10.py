import numpy as np
import scipy.io
import os

# from http://cs.nyu.edu/~roweis/data.html
path = os.environ['ML_DATA_PATH']+'/cifar10/'

# Load the original images into numpy arrays
def load_numpy():
	n_train = 5
	train = [unpickle(path+'data_batch_'+str(i+1)) for i in range(n_train)]
	train_x = np.concatenate([train[i]['data'].astype('float16')/256. for i in range(n_train)])
	train_y = np.concatenate([train[i]['labels'] for i in range(n_train)])
	test = unpickle(path+'test_batch')
	test_x = test['data'].astype('float64')/256.
	test_y = np.asarray(test['labels'])
	return train_x, train_y, test_x, test_y

def unpickle(file):
	import cPickle
	fo = open(file, 'rb')
	dict = cPickle.load(fo)
	fo.close()
	return dict

# ====
# Code to create random patches
# ====


# Create 16x16 patches, load into numpy arrays
# Input: train_x or test_x returned by load_numpy() above
def random_patches(x, width, n_patches):
	x = x.reshape((-1,3,32,32))
	n = x.shape[0]
	y = np.zeros((n_patches, 3, width, width))
	
	idxs = np.random.randint(0, n, n_patches)
	xs = np.random.randint(0, 32-width, n_patches)
	ys = np.random.randint(0, 32-width, n_patches)
	for i in range(n_patches):
		y[i] = x[idxs[i],:,xs[i]:xs[i]+width,ys[i]:ys[i]+width]
	
	y = y.reshape((n_patches,3*width*width))
	y = y.T
	
	return y

# ZCA whitening
# x: DxN matrix where D=dimensionality of data, N=number of datapoints
def zca(X):
	n = X.shape[1]
	
	# substract mean
	mean = X.mean(axis=1, keepdims=True)
	X -= mean
	
	# Eigen decomposition of covariance of x
	D, P = np.linalg.eig(X.dot(X.T))
	
	D = D.reshape((-1,1))
	
	W = P.dot(D**(-0.5) * P.T) #* (n-1)**0.5
	
	Winv = P.dot(D**0.5 * P.T) #* (n-1)**-0.5

	# tmp
	#Y = W.dot(X)
	#print Y.dot(Y.T)[0:5,0:5]
	#raise Exception()

	return mean, W, Winv
