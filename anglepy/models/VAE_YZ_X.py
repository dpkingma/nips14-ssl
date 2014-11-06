import numpy as np
import theano
import theano.tensor as T
import collections as C
import anglepy as ap
import anglepy.ndict as ndict
from anglepy.misc import lazytheanofunc

import math, inspect

'''
Variational AE (VAE) with P(Z)P(Y)P(X|Y,Z)
'''

class VAE_YZ_X(ap.VAEModel):
    
    def __init__(self, n_x, n_y, n_hidden_q, n_z, n_hidden_p, nonlinear_q='tanh', nonlinear_p='tanh', type_px='bernoulli', type_qz='gaussianmarg', type_pz='gaussianmarg', prior_sd=1, uniform_y=False):
        self.constr = (__name__, inspect.stack()[0][3], locals())
        self.n_x = n_x
        self.n_y = n_y
        self.n_hidden_q = n_hidden_q
        self.n_z = n_z
        self.n_hidden_p = n_hidden_p
        self.nonlinear_q = nonlinear_q
        self.nonlinear_p = nonlinear_p
        self.type_px = type_px
        self.type_qz = type_qz
        self.type_pz = type_pz
        self.prior_sd = prior_sd
        self.uniform_y = uniform_y
        super(VAE_YZ_X, self).__init__()
    
    def factors(self, v, w, x, z, A):
            
        '''
        z['eps'] is the independent epsilons (Gaussian with unit variance)
        x['x'] is the data
        x['y'] is categorial data (1-of-K coded)
        
        The names of list z[...] may be confusing here: the latent variable z is not included in the list z[...],
        but implicitely computed from epsilon and parameters in w.

        z is computed with g(.) from eps and variational parameters
        let logpx be the generative model density: log p(y) + log p(x|y,z) where z=g(.)
        let logpz be the prior of Z plus the entropy of q(z|x,y): logp(z) + H_q(z|x)
        So the lower bound L(x) = logpx + logpz
        
        let logpv and logpw be the (prior) density of the parameters
        '''
        
        def f_softplus(x): return T.log(T.exp(x) + 1)# - np.log(2)
        def f_rectlin(x): return x*(x>0)
        def f_rectlin2(x): return x*(x>0) + 0.01 * x
        nonlinear = {'tanh': T.tanh, 'sigmoid': T.nnet.sigmoid, 'softplus': f_softplus, 'rectlin': f_rectlin, 'rectlin2': f_rectlin2}
        nonlinear_q = nonlinear[self.nonlinear_q]
        nonlinear_p = nonlinear[self.nonlinear_p]
        
        # Compute q(z|x,y)
        hidden_q = [nonlinear_q(T.dot(v['w0x'], x['x']) + T.dot(v['w0y'], x['y']) + T.dot(v['b0'], A))]
        for i in range(1, len(self.n_hidden_q)):
            hidden_q.append(nonlinear_q(T.dot(v['w'+str(i)], hidden_q[-1]) + T.dot(v['b'+str(i)], A)))
        
        q_mean = T.dot(v['mean_w'], hidden_q[-1]) + T.dot(v['mean_b'], A)
        if self.type_qz == 'gaussian' or self.type_qz == 'gaussianmarg':
            q_logvar = T.dot(v['logvar_w'], hidden_q[-1]) + T.dot(v['logvar_b'], A)
        else: raise Exception()
        
        # function for distribution q(z|x)
        theanofunc = lazytheanofunc('ignore', mode='FAST_RUN')
        self.dist_qz['z'] = theanofunc([x['x'], x['y']] + v.values() + [A], [q_mean, q_logvar])

        # Compute virtual sample
        _z = q_mean + T.exp(0.5 * q_logvar) * z['eps']
        
        # Compute log p(x|y,z)
        hidden_p = [nonlinear_p(T.dot(w['w0y'], x['y']) + T.dot(w['w0z'], _z) + T.dot(w['b0'], A))]
        for i in range(1, len(self.n_hidden_p)):
            hidden_p.append(nonlinear_p(T.dot(w['w'+str(i)], hidden_p[-1]) + T.dot(w['b'+str(i)], A)))
        
        if self.type_px == 'bernoulli':
            p = T.nnet.sigmoid(T.dot(w['out_w'], hidden_p[-1]) + T.dot(w['out_b'], A))
            _logpx = - T.nnet.binary_crossentropy(p, x['x'])
            self.dist_px['x'] = theanofunc([x['y'], _z] + w.values() + [A], p)
        elif self.type_px == 'gaussian'or self.type_px == 'sigmoidgaussian':
            x_mean = T.dot(w['out_w'], hidden_p[-1]) + T.dot(w['out_b'], A)
            if self.type_px == 'sigmoidgaussian':
                x_mean = T.nnet.sigmoid(x_mean)
            x_logvar = T.dot(w['out_logvar_w'], hidden_p[-1]) + T.dot(w['out_logvar_b'], A)
            _logpx = ap.logpdfs.normal2(x['x'], x_mean, x_logvar)
            self.dist_px['x'] = theanofunc([x['y'], _z] + w.values() + [A], [x_mean, x_logvar])
        else: raise Exception("")
        
        # Note: logpx is a row vector (one element per sample)
        logpx = T.dot(np.ones((1, self.n_x)), _logpx) # logpx = logp(y|w) + logp(x|z,w)
        
        # log p(y) (prior of y)
        _logpy = w['logpy']
        if self.uniform_y: _logpy *= 0
        py_model = T.nnet.softmax(T.dot(_logpy, A).T).T
        logpy = (- T.nnet.categorical_crossentropy(py_model.T, x['y'].T).T).reshape((1,-1))
        logpx += logpy
        self.dist_px['y'] = theanofunc(w.values() + [A], py_model)
        
        # log p(z) (prior of z)
        if self.type_pz == 'gaussianmarg':
            logpz = -0.5 * (np.log(2 * np.pi) + (q_mean**2 + T.exp(q_logvar))).sum(axis=0, keepdims=True)
        elif self.type_pz == 'gaussian':
            logpz = ap.logpdfs.standard_normal(_z).sum(axis=0, keepdims=True)
        elif self.type_pz == 'laplace':
            logpz = ap.logpdfs.standard_laplace(_z).sum(axis=0, keepdims=True)
        elif self.type_pz == 'studentt':
            logpz = ap.logpdfs.studentt(_z, T.dot(T.exp(w['logv']), A)).sum(axis=0, keepdims=True)
        else:
            raise Exception("Unknown type_pz")
        
        # loq q(z|x) (entropy of z)
        if self.type_qz == 'gaussianmarg':
            logqz = - 0.5 * (np.log(2 * np.pi) + 1 + q_logvar).sum(axis=0, keepdims=True)
        elif self.type_qz == 'gaussian':
            logqz = ap.logpdfs.normal2(_z, q_mean, q_logvar).sum(axis=0, keepdims=True)
        else: raise Exception()
        
        # Note: logpv and logpw are a scalars
        def f_prior(_w, prior_sd=self.prior_sd):
            return ap.logpdfs.normal(_w, 0, prior_sd).sum()
        logpv = 0
        logpv += f_prior(v['w0x'])
        logpv += f_prior(v['w0y'])
        for i in range(1, len(self.n_hidden_q)):
            logpv += f_prior(v['w'+str(i)])
        logpv += f_prior(v['mean_w'])
        if self.type_qz in ['gaussian','gaussianmarg']:
            logpv += f_prior(v['logvar_w'])
        
        logpw = 0
        logpw += f_prior(w['w0y'])
        logpw += f_prior(w['w0z'])
        for i in range(1, len(self.n_hidden_p)):
            logpw += f_prior(w['w'+str(i)])
        logpw += f_prior(w['out_w'])
        if self.type_px in ['sigmoidgaussian', 'gaussian']:
            logpw += f_prior(w['out_logvar_w'])
        if self.type_pz == 'studentt':
            logpw += f_prior(w['logv'])
            
        return logpv, logpw, logpx, logpz, logqz
    
    # Generate epsilon from prior
    def gen_eps(self, n_batch):
        z = {'eps': np.random.standard_normal(size=(self.n_z, n_batch))}
        return z
    
    # Generate variables
    def gen_xz(self, v, w, x, z, n_batch):
        
        v, w, x, z = ndict.ordereddicts((v, w, x, z))
        
        A = np.ones((1, n_batch))
        
        _z = {}

        # If x['x'] and x['y'] were given but not z['z']: generate z ~ q(z|x)
        if x.has_key('x') and x.has_key('y') and not z.has_key('z'):

            q_mean, q_logvar = self.dist_qz['z'](*([x['x'], x['y']] + v.values() + [A]))
            _z['mean'] = q_mean
            _z['logvar'] = q_logvar
            
            # Require epsilon
            if not z.has_key('eps'):
                z['eps'] = self.gen_eps(n_batch)['eps']
            
            z['z'] = q_mean + np.exp(0.5 * q_logvar) * z['eps']
            
        else:
            if not z.has_key('z'):
                if self.type_pz in ['gaussian','gaussianmarg']:
                    z['z'] = np.random.standard_normal(size=(self.n_z, n_batch))
                elif self.type_pz == 'laplace':
                    z['z'] = np.random.laplace(size=(self.n_z, n_batch))
                elif self.type_pz == 'studentt':
                    z['z'] = np.random.standard_t(np.dot(np.exp(w['logv']), A))
            if not x.has_key('y'):
                py = self.dist_px['y'](*(w.values() + [A]))
                _z['y'] = py
                x['y'] = np.zeros(py.shape)
                # np.random.multinomial requires loop. Faster possible?
                for i in range(py.shape[1]):
                    x['y'][:,i] = np.random.multinomial(n=1, pvals=py[:,i])
            
        # Generate from p(x|z)
        
        if self.type_px == 'bernoulli':
            p = self.dist_px['x'](*([x['y'], z['z']] + w.values() + [A]))
            _z['x'] = p
            if not x.has_key('x'):
                x['x'] = np.random.binomial(n=1,p=p)
        elif self.type_px == 'sigmoidgaussian' or self.type_px == 'gaussian':
            x_mean, x_logvar = self.dist_px['x'](*([x['y'], z['z']] + w.values() + [A]))
            _z['x'] = x_mean
            if not x.has_key('x'):
                x['x'] = np.random.normal(x_mean, np.exp(x_logvar/2))
                if self.type_px == 'sigmoidgaussian':
                    x['x'] = np.maximum(np.zeros(x['x'].shape), x['x'])
                    x['x'] = np.minimum(np.ones(x['x'].shape), x['x'])
        
        else: raise Exception("")
        
        return x, z, _z
    
    def variables(self):
        
        # Define parameters 'w'
        v = {}
        v['w0x'], v['w0y'] = T.dmatrices('w0x','w0y')
        v['b0'] = T.dmatrix('b0')
        for i in range(1, len(self.n_hidden_q)):
            v['w'+str(i)] = T.dmatrix('w'+str(i))
            v['b'+str(i)] = T.dmatrix('b'+str(i))
        v['mean_w'] = T.dmatrix('mean_w')
        v['mean_b'] = T.dmatrix('mean_b')
        if self.type_qz in ['gaussian','gaussianmarg']:
            v['logvar_w'] = T.dmatrix('logvar_w')
        v['logvar_b'] = T.dmatrix('logvar_b')
        
        w = {}
        w['w0y'], w['w0z'] = T.dmatrices('w0y','w0z')
        w['b0'] = T.dmatrix('b0')
        for i in range(1, len(self.n_hidden_p)):
            w['w'+str(i)] = T.dmatrix('w'+str(i))
            w['b'+str(i)] = T.dmatrix('b'+str(i))
        w['out_w'] = T.dmatrix('out_w')
        w['out_b'] = T.dmatrix('out_b')
        
        if self.type_px == 'sigmoidgaussian' or self.type_px == 'gaussian':
            w['out_logvar_w'] = T.dmatrix('out_logvar_w')
            w['out_logvar_b'] = T.dmatrix('out_logvar_b')
        
        w['logpy'] = T.dmatrix('logpy')
        
        if self.type_pz == 'studentt':
            w['logv'] = T.dmatrix('logv')

        # Define latent variables 'z'
        z = {'eps': T.dmatrix('eps')}
        
        # Define observed variables 'x'
        x = {}
        x['x'] = T.dmatrix('x')
        x['y'] = T.dmatrix('y')
        
        return v, w, x, z
    
    def init_w(self, std=1e-2):
        
        def rand(size):
            return np.random.normal(0, std, size=size)
        
        v = {}
        v['w0x'] = rand((self.n_hidden_q[0], self.n_x))
        v['w0y'] = rand((self.n_hidden_q[0], self.n_y))
        v['b0'] = rand((self.n_hidden_q[0], 1))
        for i in range(1, len(self.n_hidden_q)):
            v['w'+str(i)] = rand((self.n_hidden_q[i], self.n_hidden_q[i-1]))
            v['b'+str(i)] = rand((self.n_hidden_q[i], 1))
        
        v['mean_w'] = rand((self.n_z, self.n_hidden_q[-1]))
        v['mean_b'] = rand((self.n_z, 1))
        if self.type_qz in ['gaussian','gaussianmarg']:
            v['logvar_w'] = np.zeros((self.n_z, self.n_hidden_q[-1]))
        v['logvar_b'] = np.zeros((self.n_z, 1))
        
        w = {}
        if len(self.n_hidden_p) > 0:
            w['w0y'] = rand((self.n_hidden_p[0], self.n_y))
            w['w0z'] = rand((self.n_hidden_p[0], self.n_z))
            w['b0'] = rand((self.n_hidden_p[0], 1))
            for i in range(1, len(self.n_hidden_p)):
                w['w'+str(i)] = rand((self.n_hidden_p[i], self.n_hidden_p[i-1]))
                w['b'+str(i)] = rand((self.n_hidden_p[i], 1))
            w['out_w'] = rand((self.n_x, self.n_hidden_p[-1]))
            w['out_b'] = np.zeros((self.n_x, 1))
            if self.type_px in ['sigmoidgaussian', 'gaussian']:
                w['out_logvar_w'] = rand((self.n_x, self.n_hidden_p[-1]))
                w['out_logvar_b'] = np.zeros((self.n_x, 1))
        else:
            w['out_w'] = rand((self.n_x, self.n_z))
            w['out_b'] = np.zeros((self.n_x, 1))
            if self.type_px in ['sigmoidgaussian', 'gaussian']:
                w['out_logvar_w'] = rand((self.n_x, self.n_z))
                w['out_logvar_b'] = np.zeros((self.n_x, 1))
        
        w['logpy'] = np.zeros((self.n_y, 1))
        
        if self.type_pz == 'studentt':
            w['logv'] = np.zeros((self.n_z, 1))
        
        return v, w
    