import sys

import os, numpy as np
import scipy.stats

import anglepy.paramgraphics as paramgraphics
import anglepy.ndict as ndict

from anglepy.sfo import SFO
from adam import AdaM

import theano
import theano.tensor as T

import preprocessing as pp
import time

def main(n_passes, n_labeled, n_z, n_hidden, dataset, seed, alpha, n_minibatches, comment):
    '''
    Learn a variational auto-encoder with generative model p(x,y,z)=p(y)p(z)p(x|y,z)
    And where 'x' is always observed and 'y' is _sometimes_ observed (hence semi-supervised).
    We're going to use q(y|x) as a classification model.
    '''

    import time
    logdir = 'results/learn_yz_x_ss_'+dataset+'_'+str(n_z)+'-'+str(n_hidden)+'_nlabeled'+str(n_labeled)+'_alpha'+str(alpha)+'_seed'+str(seed)+'_'+comment+'-'+str(int(time.time()))+'/'
    if not os.path.exists(logdir): os.makedirs(logdir)
    print 'logdir:', logdir
    
    print sys.argv[0], n_labeled, n_z, n_hidden, dataset, seed, comment
    
    np.random.seed(seed)
    
    # Init data
    if dataset == 'mnist_2layer':
        
        size = 28
        dim_input = (size,size)
        
        # Load model for feature extraction
        path = 'models/mnist_z_x_50-500-500_longrun/' #'models/mnist_z_x_50-600-600/'
        l1_v = ndict.loadz(path+'v.ndict.tar.gz')
        l1_w = ndict.loadz(path+'w.ndict.tar.gz')
        n_h = (500,500)
        from anglepy.models.VAE_Z_X import VAE_Z_X
        l1_model = VAE_Z_X(n_x=28*28, n_hidden_q=n_h, n_z=50, n_hidden_p=n_h, nonlinear_q='softplus', nonlinear_p='softplus', type_px='bernoulli', type_qz='gaussianmarg', type_pz='gaussianmarg', prior_sd=1)
        
        # Load dataset
        import anglepy.data.mnist as mnist
        # load train and test sets
        train_x, train_y, valid_x, valid_y, test_x, test_y = mnist.load_numpy_split(size, binarize_y=True)
       
        # create labeled/unlabeled split in training set
        x_l, y_l, x_u, y_u = mnist.create_semisupervised(train_x, train_y, n_labeled)
        
        # Extract features
        
        # 1. Determine which dimensions to keep
        def transform(v, _x):
            return l1_model.dist_qz['z'](*([_x] + v.values() + [np.ones((1, _x.shape[1]))]))
        q_mean, _ = transform(l1_v, x_u[0:1000])
        idx_keep = np.std(q_mean, axis=1) > 0.1
        
        # 2. Select dimensions
        for key in ['mean_b','mean_w','logvar_b','logvar_w']:
            l1_v[key] = l1_v[key][idx_keep,:]
        l1_w['w0'] = l1_w['w0'][:,idx_keep]
        
        # 3. Extract features
        x_mean_u, x_logvar_u = transform(l1_v, x_u)
        x_mean_l, x_logvar_l = transform(l1_v, x_l)
        x_unlabeled = {'mean':x_mean_u, 'logvar':x_logvar_u, 'y':y_u}
        x_labeled = {'mean':x_mean_l, 'logvar':x_logvar_l, 'y':y_l}
        
        valid_x, _ = transform(l1_v, valid_x)
        test_x, _ = transform(l1_v, test_x)
        
        n_x = np.sum(idx_keep)
        n_y = 10
        
        type_pz = 'gaussianmarg'
        type_px = 'gaussian'
        nonlinear = 'softplus'
        
        colorImg = False

    if dataset == 'svhn_2layer':
        
        size = 32
        dim_input = (size,size)
        
        # Load model for feature extraction
        path = 'models/tmp/svhn_z_x_300-500-500/'
        l1_v = ndict.loadz(path+'v.ndict.tar.gz')
        l1_w = ndict.loadz(path+'w.ndict.tar.gz')
        f_enc, f_dec = pp.PCA_fromfile(path+'pca_params.ndict.tar.gz', True)
        from anglepy.models.VAE_Z_X import VAE_Z_X
        n_x = l1_v['w0'].shape[1] #=600
        l1_model = VAE_Z_X(n_x=n_x, n_hidden_q=(600,600), n_z=300, n_hidden_p=(600,600), nonlinear_q='softplus', nonlinear_p='softplus', type_px='gaussian', type_qz='gaussianmarg', type_pz='gaussianmarg', prior_sd=1)
        
        # SVHN dataset
        import anglepy.data.svhn as svhn
        size = 32
        train_x, train_y, valid_x, valid_y, test_x, test_y = svhn.load_numpy_split(False, binarize_y=True, extra=False) #norb.load_resized(size, binarize_y=True)
        
        #train_x = np.hstack((_train_x, extra_x)) 
        #train_y = np.hstack((_train_y, extra_y))[:,:604000]
        
        # create labeled/unlabeled split in training set
        import anglepy.data.mnist as mnist
        x_l, y_l, x_u, y_u = mnist.create_semisupervised(train_x, train_y, n_labeled)
        
        # Extract features
        
        # 1. Determine which dimensions to keep
        def transform(v, _x):
            return l1_model.dist_qz['z'](*([f_enc(_x)] + v.values() + [np.ones((1, _x.shape[1]))]))
        
        # 2. We're keeping all latent dimensions
        
        # 3. Extract features
        x_mean_u, x_logvar_u = transform(l1_v, x_u)
        x_mean_l, x_logvar_l = transform(l1_v, x_l)
        x_unlabeled = {'mean':x_mean_u, 'logvar':x_logvar_u, 'y':y_u}
        x_labeled = {'mean':x_mean_l, 'logvar':x_logvar_l, 'y':y_l}
        
        valid_x, _ = transform(l1_v, valid_x)
        test_x, _ = transform(l1_v, test_x)
        
        n_x = l1_w['w0'].shape[1]
        n_y = 10
        
        type_pz = 'gaussianmarg'
        type_px = 'gaussian'
        nonlinear = 'softplus'
        
    # Init VAE model p(x,y,z)
    from anglepy.models.VAE_YZ_X import VAE_YZ_X
    uniform_y = True
    model = VAE_YZ_X(n_x, n_y, n_hidden, n_z, n_hidden, nonlinear, nonlinear, type_px, type_qz="gaussianmarg", type_pz=type_pz, prior_sd=1, uniform_y=uniform_y)
    v, w = model.init_w(1e-3)
    
    # Init q(y|x) model
    from anglepy.models.MLP_Categorical import MLP_Categorical
    n_units = [n_x]+list(n_hidden)+[n_y]
    model_qy = MLP_Categorical(n_units=n_units, prior_sd=1, nonlinearity=nonlinear)
    u = model_qy.init_w(1e-3)
    
    # Just test
    if False:
        u = ndict.loadz('u.ndict.tar.gz')
        v = ndict.loadz('v.ndict.tar.gz')
        w = ndict.loadz('w.ndict.tar.gz')
        pass
    
    # Progress hook
    t0 = time.time()
    
    def hook(t, u, v, w, ll):
        
        # Get classification error of validation and test sets
        def error(dataset_x, dataset_y):
            _, _, _z = model_qy.gen_xz(u, {'x':dataset_x}, {})
            return np.sum( np.argmax(_z['py'], axis=0) != np.argmax(dataset_y, axis=0)) / (0.0 + dataset_y.shape[1])
        
        valid_error = error(valid_x, valid_y)
        test_error = error(test_x, test_y)

        # Log
        ndict.savez(u, logdir+'u')
        ndict.savez(v, logdir+'v')
        ndict.savez(w, logdir+'w')
    
        dt = time.time() - t0
        
        print dt, t, ll, valid_error, test_error
        with open(logdir+'hook.txt', 'a') as f:
            print >>f, dt, t, ll, valid_error, test_error
        
        return valid_error

    # Optimize
    result = optim_vae_ss_adam(alpha, model_qy, model, x_labeled, x_unlabeled, n_y, u, v, w, n_minibatches=n_minibatches, n_passes=n_passes, hook=hook)
    
    return result
    
def optim_vae_ss_adam(alpha, model_qy, model, x_labeled, x_unlabeled, n_y, u_init, v_init, w_init, n_minibatches, n_passes, hook, n_reset=20, resample_keepmem=False, display=0):
    
    # Shuffle datasets
    ndict.shuffleCols(x_labeled)
    ndict.shuffleCols(x_unlabeled)
    
    # create minibatches
    minibatches = []

    n_labeled = x_labeled.itervalues().next().shape[1]
    n_batch_l = n_labeled / n_minibatches
    if (n_labeled%n_batch_l) != 0: raise Exception()
    
    n_unlabeled = x_unlabeled.itervalues().next().shape[1]
    n_batch_u = n_unlabeled / n_minibatches
    if (n_unlabeled%n_batch_u) != 0: raise Exception()
    
    n_tot = n_labeled + n_unlabeled

    # Divide into minibatches
    def make_minibatch(i):
        _x_labeled = ndict.getCols(x_labeled, i * n_batch_l, (i+1) * n_batch_l)
        _x_unlabeled = ndict.getCols(x_unlabeled, i * n_batch_u, (i+1) * n_batch_u)
        return [i, _x_labeled, _x_unlabeled]

    for i in range(n_minibatches):
        minibatches.append(make_minibatch(i))
    
    # For integrating-out approach
    L_inner = T.dmatrix()
    L_unlabeled = T.dot(np.ones((1, n_y)), model_qy.p * (L_inner - T.log(model_qy.p)))
    grad_L_unlabeled = T.grad(L_unlabeled.sum(), model_qy.var_w.values())
    f_du =  theano.function([model_qy.var_x['x']] + model_qy.var_w.values() + [model_qy.var_A, L_inner], [L_unlabeled] + grad_L_unlabeled)
    
    # Some statistics
    L = [0.]
    n_L = [0]
    
    def f_df(w, minibatch):
        
        u = w['u']
        v = w['v']
        w = w['w']
        
        i_minibatch = minibatch[0]
        _x_l = minibatch[1] #labeled
        x_minibatch_l = {'x': np.random.normal(_x_l['mean'], np.exp(0.5*_x_l['logvar'])), 'y': _x_l['y']}
        eps_minibatch_l = model.gen_eps(n_batch_l)
        
        _x_u = minibatch[2] #unlabeled
        x_minibatch_u = {'x': np.random.normal(_x_u['mean'], np.exp(0.5*_x_u['logvar'])), 'y': _x_u['y']}
        eps_minibatch_u = [model.gen_eps(n_batch_u) for i in range(n_y)]
        
        # === Get gradient for labeled data
        # gradient of -KL(q(z|y,x) ~p(x,y) || p(x,y,z))
        logpx, logpz, logqz, gv_labeled, gw_labeled = model.dL_dw(v, w, x_minibatch_l, eps_minibatch_l)        
        # gradient of classification error E_{~p(x,y)}[q(y|x)]
        logqy, _, gu_labeled, _ = model_qy.dlogpxz_dwz(u, x_minibatch_l, {})
        
        # Reweight gu_labeled and logqy
        #beta = alpha / (1.-alpha) * (1. * n_unlabeled / n_labeled) #old
        beta = alpha * (1. * n_tot / n_labeled)
        for i in u: gu_labeled[i] *= beta
        logqy *= beta
        
        L_labeled = logpx + logpz - logqz + logqy
        
        # === Get gradient for unlabeled data
        # -KL(q(z|x,y)q(y|x) ~p(x) || p(x,y,z))
        # Approach where outer expectation (over q(z|x,y)) is taken as explicit sum (instead of sampling)
        u = ndict.ordered(u)
        py = model_qy.dist_px['y'](*([x_minibatch_u['x']] + u.values() + [np.ones((1, n_batch_u))]))
        
        if True:
            # Original
            _L = np.zeros((n_y, n_batch_u))
            gv_unlabeled = {i: 0 for i in v}
            gw_unlabeled = {i: 0 for i in w}
            for label in range(n_y):
                new_y = np.zeros((n_y, n_batch_u))
                new_y[label,:] = 1
                eps = eps_minibatch_u[label]
                #logpx, logpz, logqz, _gv, _gw = model.dL_dw(v, w, {'x':x_minibatch['x'],'y':new_y}, eps)
                L_unweighted, L_weighted, _gv, _gw = model.dL_weighted_dw(v, w, {'x':x_minibatch_u['x'],'y':new_y}, eps, py[label:label+1,:])
                _L[label:label+1,:] = L_unweighted
                for i in v: gv_unlabeled[i] += _gv[i]
                for i in w: gw_unlabeled[i] += _gw[i]
        else:
            # New, should be more efficient. (But is not in practice)
            _y = np.zeros((n_y, n_batch_u*n_y))
            for label in range(n_y):
                _y[label,label*n_batch_u:(label+1)*n_batch_u] = 1
            _x = np.tile(x_minibatch_u['x'].astype(np.float32), (1, n_y))
            eps = model.gen_eps(n_batch_u*n_y)
            L_unweighted, L_weighted, gv_unlabeled, gw_unlabeled = model.dL_weighted_dw(v, w, {'x':_x,'y':_y}, eps, py.reshape((1, -1)))
            _L = L_unweighted.reshape((n_y, n_batch_u))
        
        r = f_du(*([x_minibatch_u['x']] + u.values() + [np.zeros((1, n_batch_u)), _L]))
        L_unlabeled = r[0]
        gu_unlabeled = dict(zip(u.keys(), r[1:]))
        
        # Get gradient of prior
        logpu, gu_prior = model_qy.dlogpw_dw(u)
        logpv, logpw, gv_prior, gw_prior = model.dlogpw_dw(v, w)
        
        # Combine gradients and objective
        gu = {i: ((gu_labeled[i] + gu_unlabeled[i]) * n_minibatches + gu_prior[i])/(-n_tot) for i in u}
        gv = {i: ((gv_labeled[i] + gv_unlabeled[i]) * n_minibatches + gv_prior[i])/(-n_tot) for i in v}
        gw = {i: ((gw_labeled[i] + gw_unlabeled[i]) * n_minibatches + gw_prior[i])/(-n_tot) for i in w}
        f = ((L_labeled.sum() + L_unlabeled.sum()) * n_minibatches + logpu + logpv + logpw)/(-n_tot)
        
        L[0] += ((L_labeled.sum() + L_unlabeled.sum()) * n_minibatches + logpu + logpv + logpw)/(-n_tot)
        n_L[0] += 1
        
        #ndict.pNorm(gu_unlabeled)
        
        return f, {'u': gu, 'v':gv, 'w':gw}
    
    w_init = {'u': u_init, 'v':v_init, 'w':w_init}
    
    optimizer = AdaM(f_df, w_init, minibatches, alpha=3e-4, beta1=0.9, beta2=0.999)
    
    for i in range(n_passes):
        w = optimizer.optimize(num_passes=1)
        LB = L[0]/(1.*n_L[0])
        testset_error = hook(i, w['u'], w['v'], w['w'], LB)
        L[0] = 0
        n_L[0] = 0
    
    return testset_error
