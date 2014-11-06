import numpy as np
import scipy.io
import os, time, math

path = os.environ['ML_DATA_PATH']

def load_numpy(toFloat=True, binarize_y=False, dtype=np.float32):
    train = scipy.io.loadmat(path+'/svhn/train_32x32.mat')
    train_x = train['X'].swapaxes(0,1).T.reshape((train['X'].shape[3], -1)).T
    train_y = train['y'].reshape((-1)) - 1
    test = scipy.io.loadmat(path+'/svhn/test_32x32.mat')
    test_x = test['X'].swapaxes(0,1).T.reshape((test['X'].shape[3], -1)).T
    test_y = test['y'].reshape((-1)) - 1
    if toFloat:
        train_x = train_x.astype(dtype)/256.
        test_x = test_x.astype(dtype)/256.
    if binarize_y:
        train_y = binarize_labels(train_y)
        test_y = binarize_labels(test_y)

    return train_x, train_y, test_x, test_y 

def load_numpy_extra(toFloat=True, binarize_y=False, dtype=np.float32):
    extra = scipy.io.loadmat(path+'/svhn_extra/extra_32x32.mat')
    extra_x = extra['X'].swapaxes(0,1).T.reshape((extra['X'].shape[3], -1)).T
    extra_y = extra['y'].reshape((-1)) - 1
    if toFloat:
        extra_x = extra_x.astype(dtype)/256.
    if binarize_y:
        extra_y = binarize_labels(extra_y)
    return extra_x, extra_y

# Loads data where data is split into class labels
def load_numpy_split(toFloat=True, binarize_y=False, extra=False):
    
    train_x, train_y, test_x, test_y = load_numpy(toFloat,binarize_y=False)
    
    if extra:
        extra_x, extra_y = load_numpy_extra(toFloat, binarize_y=False)
        train_x = np.hstack((train_x, extra_x))[:,:604000] #chop off some in the end
        train_y = np.hstack((train_y, extra_y))[:604000]
    
    # Make trainingset divisible by 1000
    keep = int(math.floor(train_x.shape[1]/1000.)*1000)
    train_x = train_x[:,:keep]
    train_y = train_y[:keep]
    
    # Use last n_valid as validation set
    n_valid = 5000
    valid_x = train_x[:,-n_valid:]
    valid_y = train_y[-n_valid:]
    train_x = train_x[:,:-n_valid]
    train_y = train_y[:-n_valid]
    
    def split_by_class(x, y, num_classes):
        result_x = [0]*num_classes
        result_y = [0]*num_classes
        for i in range(num_classes):
            idx_i = np.where(y == i)[0]
            result_x[i] = x[:,idx_i]
            result_y[i] = y[idx_i]
        return result_x, result_y
    
    train_x, train_y = split_by_class(train_x, train_y, 10)
    if binarize_y:
        test_y = binarize_labels(test_y)
        valid_y = binarize_labels(valid_y)
        for i in range(10):
            train_y[i] = binarize_labels(train_y[i])
    
    return train_x, train_y, valid_x, valid_y, test_x, test_y

# Converts integer labels to binarized labels (1-of-K coding)
def binarize_labels(y, n_classes=10):
    new_y = np.zeros((n_classes, y.shape[0]))
    for i in range(y.shape[0]):
        new_y[y[i], i] = 1
    return new_y
