import os
import anglepy.data
import h5py

path = os.environ['ML_DATA_PATH']+'/mnist_binarized/'

# MNIST binarized dataset from Hugo Larochelle
def load_numpy(size=28):
    train_x = h5py.File(path+"binarized_mnist-train.h5")['data'][:].T
    valid_x = h5py.File(path+"binarized_mnist-valid.h5")['data'][:].T
    test_x = h5py.File(path+"binarized_mnist-test.h5")['data'][:].T
    return train_x, valid_x, test_x
