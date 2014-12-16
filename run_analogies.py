import sys
sys.path.append('..')
sys.path.append('../../data/')

import math, os, time, scipy.stats, numpy as np, pylab, Image, PIL.Image
import matplotlib.pyplot as plt, matplotlib.cm as cm
import numpy.random
import theano, theano.tensor as T
import anglepy as ap
import anglepy.models as apmodels
import anglepy.ndict as ndict
import anglepy.paramgraphics as paramgraphics

import preprocessing as pp

dataset = sys.argv[1]
draw_rows = 1 #bool(sys.argv[2])

if dataset == 'svhn':
    
    # SVHN dataset
    import anglepy.data.svhn as svhn
    size = 32
    train_x, train_y, test_x, test_y = svhn.load_numpy(True, binarize_y=True) #norb.load_resized(size, binarize_y=True)
    
    n_x = 3*32*32
    dim_input = (32,32)
    type_qz = 'gaussianmarg'
    type_pz = 'gaussianmarg'
    type_px = 'gaussian'
    nonlinear = 'softplus'
    
    n_y = 10
    n_batch_w = 10
    
    colorImg = True
    binarize = False
    
    if True:
        n_hidden = (500,500)
        n_z = 300
        dir = 'models/svhn_yz_x_300-500-500/'
        from anglepy.models import GPUVAE_YZ_X
        model = GPUVAE_YZ_X(None, n_x, n_y, n_hidden, n_z, n_hidden[::-1], nonlinear, nonlinear, type_px, type_qz=type_qz, type_pz=type_pz, prior_sd=100, init_sd=1e-2)
        w = ndict.loadz(dir+'w_best.ndict.tar.gz')
        v = ndict.loadz(dir+'v_best.ndict.tar.gz')
        ndict.set_value(model.w, w)
        ndict.set_value(model.v, v)

        # PCA
        f_enc, f_dec = pp.PCA_fromfile(dir+'pca_params.ndict.tar.gz')
        
if dataset == 'mnist':
    # MNIST
    import anglepy.data.mnist as mnist
    train_x, train_y, valid_x, valid_y, test_x, test_y = mnist.load_numpy(size=28, binarize_y=True)
    f_enc, f_dec = lambda x:x, lambda x:x
    
    n_x = 28*28
    dim_input = (28,28)
    type_qz = 'gaussianmarg'
    type_pz = 'gaussianmarg'
    type_px = 'bernoulli'
    nonlinear = 'softplus'
    
    n_y = 10
    n_batch_w = 10
    
    colorImg = False
    binarize = False
    
    if True:
        n_hidden = (500,500)
        n_z = 50
        dir = 'models/mnist_yz_x_50-500-500/'
        from anglepy.models import GPUVAE_YZ_X
        model = GPUVAE_YZ_X(None, n_x, n_y, n_hidden, n_z, n_hidden[::-1], nonlinear, nonlinear, type_px, type_qz=type_qz, type_pz=type_pz, prior_sd=100, init_sd=1e-2)
        w = ndict.loadz(dir+'w.ndict.tar.gz')
        v = ndict.loadz(dir+'v.ndict.tar.gz')
        ndict.set_value(model.w, w)
        ndict.set_value(model.v, v)
        f_dec = lambda x: x

if True:
    # Some interesting analogies
    if dataset == 'mnist':
        idxs = np.asarray([[7910, 8150, 3623, 2645, 4066, 9660, 5083, 948, 2595, 2]]).T
    elif dataset == 'svhn':
        idxs = np.asarray([[2439, 820, 6590, 24106, 23978, 18466, 191, 20638, 8496, 8779, 25783, 3926, 91, 6904, 2865, 9107, 23066, 14359, 24415, 1754]]).T
    n_batch_w = idxs.shape[0]
else:
    n_samples = 50
    idxs = np.arange(test_y.shape[1])
    np.random.shuffle(idxs)
    idxs = idxs[:n_batch_w*n_samples].reshape((n_batch_w,-1))
        
# Test model
print "Test model"
z = np.random.standard_normal((n_z, n_batch_w))
zsmooth = z.copy()
smoothingfactor = 0.1
noise_var = 0.06

import time
logdir = 'results/analogies_new_'+dataset+'_'+str(int(time.time()))
if not os.path.exists(logdir): os.makedirs(logdir)

if draw_rows:
    tile_shape1 = (n_batch_w, 1)
    tile_shape2 = (n_batch_w, n_y)
else:
    tile_shape1 = (1, n_batch_w)
    tile_shape2 = (n_y, n_batch_w)
    
for sample in range(idxs.shape[1]):
    # Get some random testset datapoints
    idx = idxs[:,sample]
    np.savetxt(logdir+'/'+str(sample)+'_idx.txt', idx, fmt='%5u')
    #with open(, "w") as text_file:
    #    text_file.write("Purchase Amount: %s" % TotalAmount)
    human_x = test_x[:,idx].astype(np.float32)
    human_y = test_y[:,idx].astype(np.float32)
    
    image = paramgraphics.mat_to_img(human_x, dim_input, colorImg=colorImg, tile_shape=tile_shape1)
    fname = logdir+'/'+str(sample)+'_human.png'
    print 'Saving to '+fname
    image.save(fname, 'PNG')
    
    # Infer corresponding 'z'
    A = np.ones((1, n_batch_w)).astype(np.float32)
    q_mean, q_logvar = model.dist_qz['z'](f_enc(human_x).astype(np.float32), human_y, A)
    z = q_mean
    
    # set 'y'
    y = np.zeros((n_y, n_y*n_batch_w))
    for i in range(n_y):
        if draw_rows:   y[i,i::n_y] = 1
        else:           y[i,(n_batch_w*i):(n_batch_w*(i+1))] = 1
    
    # Set interactive mode
    
    if draw_rows:
        _z = np.repeat(z,n_y,axis=1)
    else:
        _z = np.tile(z,n_y)
    _, _, _z_confab = model.gen_xz({'y':y}, {'z':_z}, n_batch=n_y*n_batch_w)
    x_samples = f_dec(_z_confab['x'])
    
    image = paramgraphics.mat_to_img(x_samples, dim_input, colorImg=colorImg, tile_shape=tile_shape2)
    fname = logdir+'/'+str(sample)+'_machine.png'
    print 'Saving to '+fname
    image.save(fname, 'PNG')
    
    if False:
        plt.ion()
        plt.clf()
        plt.imshow(image, cmap=pylab.gray(), origin='upper')
        plt.show()
        plt.draw()

        
