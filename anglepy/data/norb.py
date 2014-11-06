import numpy as np
import scipy.io
import os
import gzip
import cPickle

path = os.environ['ML_DATA_PATH']+'/norb/'

#export PYLEARN2_DATA_PATH=~/ws/python/pylearn2data

def load_resized(size=24, binarize_y=False):
    # resized NORB dataset
    f = gzip.open(path+'norb_'+str(size)+'.pkl.gz', 'rb')
    train, valid = cPickle.load(f)
    f.close()
    train_x, train_y = train
    valid_x, valid_y = valid
    if binarize_y:
        train_y = binarize_labels(train_y, n_classes=5)
        valid_y = binarize_labels(valid_y, n_classes=5)
    return train_x, train_y, valid_x, valid_y

# Loads data where data is split into class labels
def load_resized_split(size=24, binarize_y=False):
    train_x, train_y, test_x, test_y = load_resized(size,False)
    
    def split_by_class(x, y, num_classes):
        result_x = [0]*num_classes
        result_y = [0]*num_classes
        for i in range(num_classes):
            idx_i = np.where(y == i)[0]
            result_x[i] = x[:,idx_i]
            result_y[i] = y[idx_i]
        return result_x, result_y
    
    train_x, train_y = split_by_class(train_x, train_y, 5)
    if binarize_y:
        test_y = binarize_labels(test_y)
        for i in range(10):
            train_y[i] = binarize_labels(train_y[i])
    return train_x, train_y, test_x, test_y

def load_numpy(toFloat=True, binarize_y=False):
    train = np.load(path+'norb_train.npz')
    train_x = train['arr_0'].T[:9216,:]
    train_y = train['arr_1']
    test = np.load(path+'norb_test.npz')
    test_x = test['arr_0'].T[:9216,:]
    test_y = test['arr_1']
    if toFloat:
        train_x = train_x.astype('float16')/256.
        test_x = test_x.astype('float16')/256.
    if binarize_y:
        train_y = binarize_labels(train_y)
        test_y = binarize_labels(test_y)
    raise Exception()
    
    return train_x, train_y, test_x, test_y 

# Converts integer labels to binarized labels (1-of-K coding)
def binarize_labels(y, n_classes=10):
    new_y = np.zeros((n_classes, y.shape[0]))
    for i in range(y.shape[0]):
        new_y[y[i], i] = 1
    return new_y

# Save rescaled images
def save_reshaped(shape):
    orig_shape = 96,96
    def reshape_digits(x, shape):
        x = x.T
        def rebin(_a, shape):
            sh = shape[0],_a.shape[0]//shape[0],shape[1],_a.shape[1]//shape[1]
            return _a.reshape(sh).mean(-1).mean(1)
        nrows = x.shape[0]
        ncols = shape[0]*shape[1]
        result = np.zeros((nrows, ncols))
        for i in range(nrows):
            result[i,:] = rebin(x[i,:].reshape(orig_shape), shape).reshape((1, ncols))
        return result.T

    # MNIST dataset
    train_x, train_y, test_x, test_y = load_numpy()
    train = reshape_digits(train_x, shape), train_y
    test = reshape_digits(test_x, shape), test_y
    
    f = gzip.open(path+'norb_'+str(shape[0])+'.pkl.gz','wb')
    import cPickle
    cPickle.dump((train, test), f)
    f.close()

    
    