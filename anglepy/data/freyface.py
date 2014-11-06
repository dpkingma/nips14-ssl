import numpy as np
import scipy.io
import os

# from http://cs.nyu.edu/~roweis/data.html
path = os.environ['ML_DATA_PATH']+'/freyface/frey_rawface.mat'

def load_numpy():
	return scipy.io.loadmat(path)['ff'].T/256.
