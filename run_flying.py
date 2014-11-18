import sys

import math, os, time, scipy.stats, numpy as np, pylab, Image, PIL.Image
import matplotlib.pyplot as plt, matplotlib.cm as cm
import numpy.random
import theano, theano.tensor as T
import anglepy as ap
import anglepy.models as apmodels
import anglepy.ndict as ndict
import anglepy.paramgraphics as paramgraphics

dataset = sys.argv[1] #e.g. mnist
draw_rows = bool(sys.argv[2]) # e.g. 1 (True)
target_fname = sys.argv[3]

if dataset == 'svhn':
    n_x = 3*32*32
    dim_input = (32,32)
    type_qz = 'gaussianmarg'
    type_pz = 'gaussianmarg'
    type_px = 'gaussian'
    nonlinear = 'softplus'
    
    n_y = 10
    n_batch_w = 7
    
    colorImg = True
    binarize = False
    
    if True:
        if False:
            n_hidden = (500,500)
            n_z = 300
            dir = 'models/svhn_yz_x_300-500-500/'
        else:
            n_hidden = (1000,1000)
            n_z = 300
            dir = 'models/svhn_yz_x_300-1000-1000/'
        
        from anglepy.models import GPUVAE_YZ_X
        model = GPUVAE_YZ_X(None, n_x, n_y, n_hidden, n_z, n_hidden[::-1], nonlinear, nonlinear, type_px, type_qz=type_qz, type_pz=type_pz, prior_sd=100, init_sd=1e-2)
        w = ndict.loadz(dir+'w_best.ndict.tar.gz')
        v = ndict.loadz(dir+'v_best.ndict.tar.gz')
        ndict.set_value(model.w, w)
        ndict.set_value(model.v, v)
        # PCA
        pca = ndict.loadz(dir+'pca_params.ndict.tar.gz')
        def f_dec(x):
            result = pca['eigvec'].dot(x * np.sqrt(pca['eigval'])) * pca['x_sd'] + pca['x_center']
            result = np.maximum(0, np.minimum(1, result))
            return result

if dataset == 'mnist':
    n_x = 28*28
    dim_input = (28,28)
    type_qz = 'gaussianmarg'
    type_pz = 'gaussianmarg'
    type_px = 'bernoulli'
    nonlinear = 'softplus'
    
    n_y = 10
    n_batch_w = 7
    
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


# Test model
print "Test model"
z = np.random.standard_normal((n_z, n_batch_w))
zsmooth = z.copy()
smoothingfactor = 0.1
noise_var = 0.06

import time
logdir = '/tmp/flying_'+dataset+'_'+str(int(time.time()))

if draw_rows:
    tile_shape = (n_batch_w, n_y)
else:
    tile_shape = (n_y, n_batch_w)

# set 'y'
y = np.zeros((n_y, n_y*n_batch_w))
for i in range(n_y):
    if draw_rows:   y[i,i::n_y] = 1
    else:           y[i,(n_batch_w*i):(n_batch_w*(i+1))] = 1

# Set interactive mode
plt.ion()

for i in range(2000):
    # Do step of Gaussian diffusion process
    z = np.sqrt(1-noise_var)*z + np.sqrt(noise_var)*np.random.standard_normal(z.shape)
    # Smooth the trajectory
    zsmooth += smoothingfactor*(z-zsmooth)
    if draw_rows: _z = np.repeat(zsmooth,n_y,axis=1)
    else: _z = np.tile(zsmooth,n_y)
    _, _, _z_confab = model.gen_xz({'y':y}, {'z':_z}, n_batch=n_y*n_batch_w)
    x_samples = f_dec(_z_confab['x'])
    
    if False:
        image = paramgraphics.mat_to_img(x_samples, dim_input, colorImg=colorImg, tile_shape=tile_shape)
        plt.clf()
        plt.imshow(image, cmap=pylab.gray(), origin='upper')
        plt.show()
        plt.draw()
    else:
        if not os.path.exists(logdir): os.makedirs(logdir)
        image = paramgraphics.mat_to_img(x_samples, dim_input, colorImg=colorImg, tile_shape=tile_shape)
        # Make sure the nr of rows and cols are even
        width, height = image.size
        if width%2==1: width += 1
        if height%2==1: height += 1
        image = image.resize((width, height))
        # Save it
        fname = logdir+'/'+str(i)+'.png'
        print 'Saving to '+fname
        image.save(fname, 'PNG')
        
import os
os.system("ffmpeg -start_number 0 -i "+logdir+"/%d.png -c:v libx264 -pix_fmt yuv420p -r 30 "+target_fname)
print "Saved to "+target_fname
print "Done."

