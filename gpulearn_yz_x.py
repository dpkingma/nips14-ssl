import sys

import os, numpy as np
import scipy.stats
import anglepy.paramgraphics as paramgraphics
import anglepy.ndict as ndict
import matplotlib.pyplot as plt
import Image
import math

import theano
import theano.tensor as T
from collections import OrderedDict

import preprocessing as pp

def main(n_z, n_hidden, dataset, seed, gfx=True, _size=None):
    '''Learn a variational auto-encoder with generative model p(x,y,z)=p(y)p(z)p(x|y,z).
    x and y are (always) observed.
    I.e. this cannot be used for semi-supervised learning
    '''
    assert (type(n_hidden) == tuple or type(n_hidden) == list)
    assert type(n_z) == int
    assert isinstance(dataset, basestring)
    
    print 'gpulearn_yz_x', n_z, n_hidden, dataset, seed
    
    import time
    logdir = 'results/gpulearn_yz_x_'+dataset+'_'+str(n_z)+'-'+str(n_hidden)+'-'+str(int(time.time()))+'/'
    if not os.path.exists(logdir): os.makedirs(logdir)
    print 'logdir:', logdir
    
    np.random.seed(seed)
    
    # Init data
    if dataset == 'mnist':
        '''
        What works well:
        100-2-100 (Generated digits stay bit shady)
        1000-2-1000 (Needs pretty long training)
        '''
        import anglepy.data.mnist as mnist
        
        # MNIST
        size = 28
        train_x, train_y, valid_x, valid_y, test_x, test_y = mnist.load_numpy(size, binarize_y=True)
        f_enc, f_dec = lambda x:x, lambda x:x
        x = {'x': train_x[:,:].astype(np.float32), 'y': train_y[:,:].astype(np.float32)}
        x_valid = {'x': valid_x.astype(np.float32), 'y': valid_y.astype(np.float32)}
        L_valid = 1
        dim_input = (size,size)
        n_x = size*size
        n_y = 10
        n_batch = 1000
        colorImg = False
        bernoulli_x = True
        byteToFloat = False
        mosaic_w = 5
        mosaic_h = 2
        type_px = 'bernoulli'

    elif dataset == 'norb':
        # resized NORB dataset, reshuffled
        import anglepy.data.norb as norb
        size = _size #48
        train_x, train_y, test_x, test_y = norb.load_resized(size, binarize_y=True)
        _x = {'x': train_x, 'y': train_y}
        ndict.shuffleCols(_x)
        train_x = _x['x']
        train_y = _x['y']
        
        # Do PCA
        f_enc, f_dec, pca_params = pp.PCA(_x['x'][:,:10000], cutoff=2000, toFloat=False)
        ndict.savez(pca_params, logdir+'pca_params')
        
        x = {'x': f_enc(train_x).astype(np.float32), 'y':train_y.astype(np.float32)}
        x_valid = {'x': f_enc(test_x).astype(np.float32), 'y':test_y.astype(np.float32)}
        
        L_valid = 1
        n_x = x['x'].shape[0]
        n_y = 5
        dim_input = (size,size)
        n_batch = 1000 #23400/900 = 27
        colorImg = False
        bernoulli_x = False
        byteToFloat = False
        mosaic_w = 5
        mosaic_h = 1
        type_px = 'gaussian'

    elif dataset == 'norb_instances': 
        # resized NORB dataset with the instances as classes
        import anglepy.data.norb2 as norb2
        size = _size #48
        x, y = norb2.load_numpy_subclasses(size, binarize_y=True)
        _x = {'x': x, 'y': y}
        ndict.shuffleCols(_x)
        
        # Do pre=processing
        if True:
            # Works
            f_enc, f_dec, pca_params = pp.PCA(_x['x'][:,:10000], cutoff=600, global_sd=True, toFloat=True)
            ndict.savez(pca_params, logdir+'pca_params')
        elif False:
            # Doesn't work
            f_enc, f_dec, pp_params = pp.normalize_noise(_x['x'][:,:50000], noise_sd=0.01, global_sd=True, toFloat=True)
        else:
            # Doesn't work
            f_enc, f_dec, params = pp.normalize_random(x=x[:,:10000], global_sd=True, toFloat=True)
            ndict.savez(params, logdir+'normalize_random_params')
        
        n_valid = 5000
        x = {'x': f_enc(_x['x'][:,:-n_valid]).astype(np.float32), 'y':_x['y'][:,:-n_valid].astype(np.float32)}
        x_valid = {'x': f_enc(_x['x'][:,:n_valid]).astype(np.float32), 'y':_x['y'][:,:n_valid].astype(np.float32)}
        
        L_valid = 1
        n_x = x['x'].shape[0]
        n_y = 50
        dim_input = (size,size)
        n_batch = 5000 #23400/900 = 27
        colorImg = False
        bernoulli_x = False
        byteToFloat = False
        mosaic_w = 5
        mosaic_h = 1
        type_px = 'gaussian'

    elif dataset == 'svhn':    
        # SVHN dataset
        import anglepy.data.svhn as svhn
        size = 32
        train_x, train_y, test_x, test_y = svhn.load_numpy(False, binarize_y=True) #norb.load_resized(size, binarize_y=True)
        extra_x, extra_y = svhn.load_numpy_extra(False, binarize_y=True)
        x = {'x': np.hstack((train_x, extra_x)), 'y':np.hstack((train_y, extra_y))}
        ndict.shuffleCols(x)
        
        #f_enc, f_dec, (x_sd, x_mean) = pp.preprocess_normalize01(train_x, True)
        f_enc, f_dec, pca_params = pp.PCA(x['x'][:,:10000], cutoff=1000, toFloat=True)
        ndict.savez(pca_params, logdir+'pca_params')
        
        n_y = 10
        x = {'x': f_enc(x['x']).astype(np.float32), 'y': x['y'].astype(np.float32)}
        x_valid = {'x': f_enc(test_x).astype(np.float32), 'y': test_y.astype(np.float32)}
        L_valid = 1
        n_x = x['x'].shape[0]
        dim_input = (size,size)
        n_batch = 5000
        colorImg = True
        bernoulli_x = False
        byteToFloat = False
        mosaic_w = 5
        mosaic_h = 2
        type_px = 'gaussian'
        
    # Init model
    n_hidden_q = n_hidden
    n_hidden_p = n_hidden
    from anglepy.models import GPUVAE_YZ_X
    updates = get_adam_optimizer(alpha=3e-4, beta1=0.9, beta2=0.999, weight_decay=0)
    model = GPUVAE_YZ_X(updates, n_x, n_y, n_hidden_q, n_z, n_hidden_p[::-1], 'softplus', 'softplus', type_px=type_px, type_qz='gaussianmarg', type_pz='gaussianmarg', prior_sd=1, uniform_y=True)
    
    if False:
        dir = '/home/ubuntu/results/gpulearn_yz_x_svhn_300-(500, 500)-1414094291/'
        dir = '/home/ubuntu/results/gpulearn_yz_x_svhn_300-(500, 500)-1414163488/'
        w = ndict.loadz(dir+'w_best.ndict.tar.gz')
        v = ndict.loadz(dir+'v_best.ndict.tar.gz')
        ndict.set_value(model.w, w)
        ndict.set_value(model.v, v)
    
    # Some statistics for optimization
    ll_valid_stats = [-1e99, 0]

    # Fixed sample for visualisation
    z_sample = {'z': np.repeat(np.random.standard_normal(size=(n_z, 12)), 12, axis=1).astype(np.float32)}
    y_sample = {'y': np.tile(np.random.multinomial(1, [1./n_y]*n_y, size=12).T, (1, 12))}
    
    # Progress hook
    def hook(epoch, t, ll):
        
        if epoch%10 != 0:
            return
        
        ll_valid, _ = model.est_loglik(x_valid, n_samples=L_valid, n_batch=n_batch, byteToFloat=byteToFloat)
            
        if math.isnan(ll_valid):
            print "NaN detected. Reverting to saved best parameters"
            ndict.set_value(model.v, ndict.loadz(logdir+'v.ndict.tar.gz'))
            ndict.set_value(model.w, ndict.loadz(logdir+'w.ndict.tar.gz'))
            return
            
        if ll_valid > ll_valid_stats[0]:
            ll_valid_stats[0] = ll_valid
            ll_valid_stats[1] = 0
            ndict.savez(ndict.get_value(model.v), logdir+'v_best')
            ndict.savez(ndict.get_value(model.w), logdir+'w_best')
        else:
            ll_valid_stats[1] += 1
            # Stop when not improving validation set performance in 100 iterations
            if False and ll_valid_stats[1] > 1000:
                print "Finished"
                with open(logdir+'hook.txt', 'a') as f:
                    print >>f, "Finished"
                exit()

        # Log
        ndict.savez(ndict.get_value(model.v), logdir+'v')
        ndict.savez(ndict.get_value(model.w), logdir+'w')
        print epoch, t, ll, ll_valid
        with open(logdir+'hook.txt', 'a') as f:
            print >>f, t, ll, ll_valid
        
        if gfx:   
            # Graphics
            
            v = {i: model.v[i].get_value() for i in model.v}
            w = {i: model.w[i].get_value() for i in model.w}
                
            tail = '-'+str(epoch)+'.png'
            
            image = paramgraphics.mat_to_img(f_dec(v['w0x'][:].T), dim_input, True, colorImg=colorImg)
            image.save(logdir+'q_w0x'+tail, 'PNG')
            
            image = paramgraphics.mat_to_img(f_dec(w['out_w'][:]), dim_input, True, colorImg=colorImg)
            image.save(logdir+'out_w'+tail, 'PNG')
            
            _x = {'y': np.random.multinomial(1, [1./n_y]*n_y, size=144).T}
            _, _, _z_confab = model.gen_xz(_x, {}, n_batch=144)
            image = paramgraphics.mat_to_img(f_dec(_z_confab['x']), dim_input, colorImg=colorImg)
            image.save(logdir+'samples'+tail, 'PNG')
            
            _, _, _z_confab = model.gen_xz(y_sample, z_sample, n_batch=144)
            image = paramgraphics.mat_to_img(f_dec(_z_confab['x']), dim_input, colorImg=colorImg)
            image.save(logdir+'samples_fixed'+tail, 'PNG')
            
            if n_z == 2:
                
                import ImageFont
                import ImageDraw
                
                n_width = 10
                submosaic_offset = 15
                submosaic_width = (dim_input[1]*n_width)
                submosaic_height = (dim_input[0]*n_width)
                mosaic = Image.new("RGB", (submosaic_width*mosaic_w, submosaic_offset+submosaic_height*mosaic_h))
                
                for digit in range(0,n_y):
                    if digit >= mosaic_h*mosaic_w: continue
                    
                    _x = {}
                    n_batch_plot = n_width*n_width
                    _x['y'] = np.zeros((n_y,n_batch_plot))
                    _x['y'][digit,:] = 1
                    _z = {'z':np.zeros((2,n_width**2))}
                    for i in range(0,n_width):
                        for j in range(0,n_width):
                            _z['z'][0,n_width*i+j] = scipy.stats.norm.ppf(float(i)/n_width+0.5/n_width)
                            _z['z'][1,n_width*i+j] = scipy.stats.norm.ppf(float(j)/n_width+0.5/n_width)
                    
                    _x, _, _z_confab = model.gen_xz(_x, _z, n_batch=n_batch_plot)
                    x_samples = _z_confab['x']
                    image = paramgraphics.mat_to_img(f_dec(x_samples), dim_input, colorImg=colorImg, tile_spacing=(0,0))
                    
                    #image.save(logdir+'samples_digit_'+str(digit)+'_'+tail, 'PNG')
                    mosaic_x = (digit%mosaic_w)*submosaic_width
                    mosaic_y = submosaic_offset+int(digit/mosaic_w)*submosaic_height
                    mosaic.paste(image, (mosaic_x, mosaic_y))
                
                draw = ImageDraw.Draw(mosaic)
                draw.text((1,1),"Epoch #"+str(epoch)+" Loss="+str(int(ll)))
                    
                #plt.savefig(logdir+'mosaic'+tail, format='PNG')
                mosaic.save(logdir+'mosaic'+tail, 'PNG')
                
                #x_samples = _x['x']
                #image = paramgraphics.mat_to_img(f_dec(x_samples), dim_input, colorImg=colorImg)
                #image.save(logdir+'samples2'+tail, 'PNG')
        
    # Optimize
    dostep = epoch_vae_adam(model, x, n_batch=n_batch, bernoulli_x=bernoulli_x, byteToFloat=byteToFloat)
    loop_va(dostep, hook)
    
    pass

# Training loop for variational autoencoder
def loop_va(doEpoch, hook, n_epochs=9999999):
    import time
    t0 = time.time()
    for t in xrange(1, n_epochs):
        L = doEpoch()
        hook(t, time.time() - t0, L)
        
    print 'Optimization loop finished'

# Learning step for variational auto-encoder
def epoch_vae_adam(model, x, n_batch=100, convertImgs=False, bernoulli_x=False, byteToFloat=False):
    print 'Variational Auto-Encoder', n_batch
    
    def doEpoch():
        
        from collections import OrderedDict

        n_tot = x.itervalues().next().shape[1]
        idx_from = 0
        L = 0
        while idx_from < n_tot:
            idx_to = min(n_tot, idx_from+n_batch)
            x_minibatch = ndict.getCols(x, idx_from, idx_to)
            idx_from += n_batch
            if byteToFloat: x_minibatch['x'] = x_minibatch['x'].astype(np.float32)/256.
            if bernoulli_x: x_minibatch['x'] = np.random.binomial(n=1, p=x_minibatch['x']).astype(np.float32)
            
            # Get gradient
            #raise Exception()
            L += model.evalAndUpdate(x_minibatch, {}).sum()
            #model.profmode.print_summary()
            
        L /= n_tot
        
        return L
        
    return doEpoch

def get_adam_optimizer(alpha=3e-4, beta1=0.9, beta2=0.999, weight_decay=0.0):
    print 'AdaM', alpha, beta1, beta2, weight_decay
    def shared32(x, name=None, borrow=False):
        return theano.shared(np.asarray(x, dtype='float32'), name=name, borrow=borrow)

    def get_optimizer(w, g):
        updates = OrderedDict()
        
        it = shared32(0.)
        updates[it] = it + 1.
        
        fix1 = 1.-beta1**(it+1.) # To make estimates unbiased
        fix2 = 1.-beta2**(it+1.) # To make estimates unbiased
        lr_t = alpha * T.sqrt(fix2) / fix1
        
        for i in w:
    
            gi = g[i]
            if weight_decay > 0:
                gi -= weight_decay * w[i] #T.tanh(w[i])

            # mean_squared_grad := E[g^2]_{t-1}
            mom1 = shared32(w[i].get_value() * 0.)
            mom2 = shared32(w[i].get_value() * 0.)
            
            # Update moments
            mom1_new = mom1 + (1.-beta1) * (gi - mom1)
            mom2_new = mom2 + (1.-beta2) * (T.sqr(gi) - mom2)
            
            # Compute the effective gradient
            effgrad = mom1_new / (T.sqrt(mom2_new) + 1e-8)
            
            # Do update
            w_new = w[i] + lr_t * effgrad
                
            # Apply update
            updates[w[i]] = w_new
            updates[mom1] = mom1_new
            updates[mom2] = mom2_new
            
        return updates
    
    return get_optimizer
