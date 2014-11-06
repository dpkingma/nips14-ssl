import numpy as np
import scipy.io
import os
import gzip
import cPickle

path = os.environ['ML_DATA_PATH']+'/norb_orig_np/'

def load_numpy_dat(size=48):
    with gzip.open(path+'train_dat_'+str(size)+'.pkl.gz', 'rb') as f:
        train_dat = cPickle.load(f)
    with gzip.open(path+'test_dat_'+str(size)+'.pkl.gz', 'rb') as f:
        test_dat = cPickle.load(f)
    return train_dat, test_dat

def load_numpy_cat():
    with gzip.open(path+'train_cat.pkl.gz', 'rb') as f:
        train_cat = cPickle.load(f)
    with gzip.open(path+'test_cat.pkl.gz', 'rb') as f:
        test_cat = cPickle.load(f)
    return train_cat, test_cat

def load_numpy_info():
    with gzip.open(path+'train_info.pkl.gz', 'rb') as f:
        train_info = cPickle.load(f)
    with gzip.open(path+'test_info.pkl.gz', 'rb') as f:
        test_info = cPickle.load(f)
    return train_info, test_info

# Load dataset with 50 subclasses, merged to single matrices
def load_numpy_subclasses(size=48, binarize_y=False):
    train_dat, test_dat = load_numpy_dat(size)
    train_info, test_info = load_numpy_info()
    train_cat, test_cat = load_numpy_cat()
    
    n = train_dat.shape[0]
    
    n_class = 5 #number of classes
    n_ipc = 10 #number of instances per class
    
    train_x_left = train_dat[:,0].reshape((n, -1)).T
    train_x_right = train_dat[:,1].reshape((n, -1)).T
    train_y = (train_cat[:]*n_ipc + train_info[:,0]).reshape(1, n) # computes which of the 50 subclasses
    
    test_x_left = test_dat[:,0].reshape((n, -1)).T
    test_x_right = test_dat[:,1].reshape((n, -1)).T
    test_y = (test_cat[:]*n_ipc + test_info[:,0]).reshape(1, n)
    
    x = np.hstack((train_x_left, train_x_right, test_x_left, test_x_right)) # computes which of the 50 subclasses
    y = np.hstack((train_y, train_y, test_y, test_y))
    
    if binarize_y:
        y = binarize_labels(y.reshape(-1,), n_classes=n_class*n_ipc)
    
    return x, y
    
# Original data to numpy-format data
def convert_orig_to_np():
    from pylearn2.datasets.filetensor import read
    import gzip
    import cPickle
    # Load data
    path_orig = os.environ['ML_DATA_PATH']+'/norb_orig/'
    prefix_train = path_orig+'smallnorb-5x46789x9x18x6x2x96x96-training-'
    train_cat = read(gzip.open(prefix_train+'cat.mat.gz'))
    train_dat = read(gzip.open(prefix_train+'dat.mat.gz'))
    train_info = read(gzip.open(prefix_train+'info.mat.gz'))
    prefix_test = path_orig+'smallnorb-5x01235x9x18x6x2x96x96-testing-'
    test_cat = read(gzip.open(prefix_test+'cat.mat.gz'))
    test_dat = read(gzip.open(prefix_test+'dat.mat.gz'))
    test_info = read(gzip.open(prefix_test+'info.mat.gz'))
    
    # Save originals matrices to file
    files = (('train_cat', train_cat), ('train_dat_96', train_dat), ('train_info', train_info), ('test_cat', test_cat), ('test_dat_96', test_dat), ('test_info', test_info))
    for fname, tensor in files:
        print 'Saving to ', fname, '...'
        with gzip.open(path+fname+'.pkl.gz','wb') as f:
            cPickle.dump(tensor, f)

    # Save downscaled version too
    w = 48
    files = (('test_dat', test_dat),)
    for fname, tensor in files:
        print 'Generating downscaled version ' + fname + '...'
        left = reshape_images(tensor[:,0,:,:], (w,w))
        right = reshape_images(tensor[:,1,:,:], (w,w)) 
        result = np.zeros((tensor.shape[0], 2, w, w), dtype=np.uint8)
        result[:,0,:,:] = left
        result[:,1,:,:] = right
        f = gzip.open(path+fname+'_'+str(w)+'.pkl.gz', 'wb')
        cPickle.dump(result, f)
        f.close()
        
# Reshape digits
def reshape_images(x, shape):
    def rebin(_a, shape):
        sh = shape[0],_a.shape[0]//shape[0],shape[1],_a.shape[1]//shape[1]
        result = _a.reshape(sh).mean(-1).mean(1)
        return np.floor(result).astype(np.uint8)
    nrows = x.shape[0]
    result = np.zeros((nrows, shape[0], shape[1]), dtype=np.uint8)
    for i in range(nrows):
        result[i,:,:] = rebin(x[i,:,:], shape)
    return result

# Converts integer labels to binarized labels (1-of-K coding)
def binarize_labels(y, n_classes=5):
    new_y = np.zeros((n_classes, y.shape[0]))
    for i in range(y.shape[0]):
        new_y[y[i], i] = 1
    return new_y
    