import numpy as np
import theano
import theano.tensor as T
import collections as C
import anglepy as ap
import anglepy.ndict as ndict
from anglepy.misc import lazytheanofunc

import math, inspect

#import theano.sandbox.cuda.rng_curand as rng_curand

def shared32(x, name=None, borrow=False):
    return theano.shared(np.asarray(x, dtype='float32'), name=name, borrow=borrow)

'''
Fully connected deep variational auto-encoder (VAE_YZ_X)
'''

class GPUVAE_YZ_X(ap.GPUVAEModel):
    
    def __init__(self, get_optimizer, n_x, n_y, n_hidden_q, n_z, n_hidden_p, nonlinear_q='softplus', nonlinear_p='softplus', type_px='bernoulli', type_qz='gaussianmarg', type_pz='gaussianmarg', prior_sd=1, init_sd=1e-3, var_smoothing=0, n_mixture=0, uniform_y=False):
        self.constr = (__name__, inspect.stack()[0][3], locals())
        self.n_x = n_x
        self.n_y = n_y
        self.n_hidden_q = n_hidden_q
        self.n_z = n_z
        self.n_hidden_p = n_hidden_p
        self.dropout = False
        self.nonlinear_q = nonlinear_q
        self.nonlinear_p = nonlinear_p
        self.type_px = type_px
        self.type_qz = type_qz
        self.type_pz = type_pz
        self.prior_sd = prior_sd
        self.var_smoothing = var_smoothing
        self.n_mixture = n_mixture
        self.uniform_y = uniform_y
        
        # Init weights
        v, w = self.init_w(1e-2)
        for i in v: v[i] = shared32(v[i])
        for i in w: w[i] = shared32(w[i])
        self.v = v
        self.w = w
        
        super(GPUVAE_YZ_X, self).__init__(get_optimizer)
    
    def factors(self, x, z, A):
        
        v = self.v
        w = self.w
        
        '''
        z is unused
        x['x'] is the data
        
        The names of dict z[...] may be confusing here: the latent variable z is not included in the dict z[...],
        but implicitely computed from epsilon and parameters in w.

        z is computed with g(.) from eps and variational parameters
        let logpx be the generative model density: log p(x|z) where z=g(.)
        let logpz be the prior of Z plus the entropy of q(z|x): logp(z) + H_q(z|x)
        So the lower bound L(x) = logpx + logpz
        
        let logpv and logpw be the (prior) density of the parameters
        '''
        
        def f_softplus(x): return T.log(T.exp(x) + 1)# - np.log(2)
        def f_rectlin(x): return x*(x>0)
        def f_rectlin2(x): return x*(x>0) + 0.01 * x
        nonlinear = {'tanh': T.tanh, 'sigmoid': T.nnet.sigmoid, 'softplus': f_softplus, 'rectlin': f_rectlin, 'rectlin2': f_rectlin2}
        nonlinear_q = nonlinear[self.nonlinear_q]
        nonlinear_p = nonlinear[self.nonlinear_p]
        
        #rng = rng_curand.CURAND_RandomStreams(0)
        import theano.tensor.shared_randomstreams
        rng = theano.tensor.shared_randomstreams.RandomStreams(0)
        
        # Compute q(z|x,y)
        hidden_q = [nonlinear_q(T.dot(v['w0x'], x['x']) + T.dot(v['w0y'], x['y']) + T.dot(v['b0'], A))]
        for i in range(1, len(self.n_hidden_q)):
            hidden_q.append(nonlinear_q(T.dot(v['w'+str(i)], hidden_q[-1]) + T.dot(v['b'+str(i)], A)))
        
        q_mean = T.dot(v['mean_w'], hidden_q[-1]) + T.dot(v['mean_b'], A)
        if self.type_qz == 'gaussian' or self.type_qz == 'gaussianmarg':
            q_logvar = T.dot(v['logvar_w'], hidden_q[-1]) + T.dot(v['logvar_b'], A)
        else: raise Exception()
        
        # function for distribution q(z|x)
        theanofunc = lazytheanofunc('warn', mode='FAST_RUN')
        self.dist_qz['z'] = theanofunc([x['x'], x['y']] + [A], [q_mean, q_logvar])
        
        # Compute virtual sample
        eps = rng.normal(size=q_mean.shape, dtype='float32')
        _z = q_mean + T.exp(0.5 * q_logvar) * eps
        
        # Compute log p(x|z)
        hidden_p = [nonlinear_p(T.dot(w['w0y'], x['y']) + T.dot(w['w0z'], _z) + T.dot(w['b0'], A))]
        for i in range(1, len(self.n_hidden_p)):
            hidden_p.append(nonlinear_p(T.dot(w['w'+str(i)], hidden_p[-1]) + T.dot(w['b'+str(i)], A)))
            if self.dropout:
                hidden_p[-1] *= 2. * (rng.uniform(size=hidden_p[-1].shape, dtype='float32') > .5)
        
        if self.type_px == 'bernoulli':
            p = T.nnet.sigmoid(T.dot(w['out_w'], hidden_p[-1]) + T.dot(w['out_b'], A))
            _logpx = - T.nnet.binary_crossentropy(p, x['x'])
            self.dist_px['x'] = theanofunc([x['y'], _z] + [A], p)
        elif self.type_px == 'gaussian':
            x_mean = T.dot(w['out_w'], hidden_p[-1]) + T.dot(w['out_b'], A)
            x_logvar = T.dot(w['out_logvar_w'], hidden_p[-1]) + T.dot(w['out_logvar_b'], A)
            _logpx = ap.logpdfs.normal2(x['x'], x_mean, x_logvar)
            self.dist_px['x'] = theanofunc([x['y'], _z] + [A], [x_mean, x_logvar])
        elif self.type_px == 'laplace':
            x_mean = T.dot(w['out_w'], hidden_p[-1]) + T.dot(w['out_b'], A)
            x_logvar = T.dot(w['out_logvar_w'], hidden_p[-1]) + T.dot(w['out_logvar_b'], A)
            _logpx = ap.logpdfs.laplace(x['x'], x_mean, x_logvar)
            self.dist_px['x'] = theanofunc([x['y'], _z] + [A], [x_mean, x_logvar])
            
        else: raise Exception("")
            
        # Note: logpx is a row vector (one element per sample)
        logpx = T.dot(shared32(np.ones((1, self.n_x))), _logpx) # logpx = log p(x|z,w)
        
        # log p(y) (prior of y)
        #_logpy = w['logpy']
        #if self.uniform_y: _logpy *= 0
        #py_model = T.nnet.softmax(T.dot(_logpy, A).T).T
        #logpy = (- T.nnet.categorical_crossentropy(py_model.T, x['y'].T).T).reshape((1,-1))
        #logpx += logpy
        #self.dist_px['y'] = theanofunc([A], py_model)

        # log p(z) (prior of z)
        if self.type_pz == 'gaussianmarg':
            logpz = -0.5 * (np.log(2 * np.pi) + (q_mean**2 + T.exp(q_logvar))).sum(axis=0, keepdims=True)
        elif self.type_pz == 'gaussian':
            logpz = ap.logpdfs.standard_normal(_z).sum(axis=0, keepdims=True)
        elif self.type_pz == 'mog':
            pz = 0
            for i in range(self.n_mixture):
                pz += T.exp(ap.logpdfs.normal2(_z, T.dot(w['mog_mean'+str(i)], A), T.dot(w['mog_logvar'+str(i)], A)))
            logpz = T.log(pz).sum(axis=0, keepdims=True) - self.n_z * np.log(float(self.n_mixture))
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
        if self.type_px in ['sigmoidgaussian', 'gaussian','laplace']:
            logpw += f_prior(w['out_logvar_w'])
        if self.type_pz == 'studentt':
            logpw += f_prior(w['logv'])
            
        #return logpv, logpw, logpx, logpz, logqz
        return logpx, logpz, logqz
    
    # Generate epsilon from prior
    def gen_eps(self, n_batch):
        z = {'eps': np.random.standard_normal(size=(self.n_z, n_batch)).astype('float32')}
        return z
    
    # Generate variables
    def gen_xz(self, x, z, n_batch):
        
        x, z = ndict.ordereddicts((x, z))
        
        A = np.ones((1, n_batch)).astype(np.float32)
        for i in z: z[i] = z[i].astype(np.float32)
        for i in x: x[i] = x[i].astype(np.float32)
        
        _z = {}

        # If x['x'] was given but not z['z']: generate z ~ q(z|x)
        if x.has_key('x') and x.has_key('y') and not z.has_key('z'):

            q_mean, q_logvar = self.dist_qz['z'](*([x['x'], x['y']] + [A]))
            _z['mean'] = q_mean
            _z['logvar'] = q_logvar
            
            # Require epsilon
            if not z.has_key('eps'):
                eps = self.gen_eps(n_batch)['eps']
            
            z['z'] = q_mean + np.exp(0.5 * q_logvar) * eps
            
        else:
            if not z.has_key('z'):
                if self.type_pz in ['gaussian','gaussianmarg']:
                    z['z'] = np.random.standard_normal(size=(self.n_z, n_batch)).astype(np.float32)
                elif self.type_pz == 'laplace':
                    z['z'] = np.random.laplace(size=(self.n_z, n_batch)).astype(np.float32)
                elif self.type_pz == 'studentt':
                    z['z'] = np.random.standard_t(np.dot(np.exp(self.w['logv'].get_value()), A)).astype(np.float32)
                elif self.type_pz == 'mog':
                    i = np.random.randint(self.n_mixture)
                    loc = np.dot(self.w['mog_mean'+str(i)].get_value(), A)
                    scale = np.dot(np.exp(.5*self.w['mog_logvar'+str(i)].get_value()), A)
                    z['z'] = np.random.normal(loc=loc, scale=scale).astype(np.float32)
                else:
                    raise Exception('Unknown type_pz')
            if not x.has_key('y'):
                py = self.dist_px['y'](*([A]))
                _z['y'] = py
                x['y'] = np.zeros(py.shape)
                # np.random.multinomial requires loop. Faster possible?
                for i in range(py.shape[1]):
                    x['y'][:,i] = np.random.multinomial(n=1, pvals=py[:,i])
                
        # Generate from p(x|z)
        
        if self.type_px == 'bernoulli':
            p = self.dist_px['x'](*([x['y'], z['z']] + [A]))
            _z['x'] = p
            if not x.has_key('x'):
                x['x'] = np.random.binomial(n=1,p=p)
        elif self.type_px == 'laplace':
            x_mean, x_logvar = self.dist_px['x'](*([x['y'], z['z']] + [A]))
            _z['x'] = x_mean
            if not x.has_key('x'):
                x['x'] = np.random.laplace(x_mean, np.exp(0.5 * x_logvar))
        elif self.type_px == 'bounded01' or self.type_px == 'gaussian':
            x_mean, x_logvar = self.dist_px['x'](*([x['y'], z['z']] + [A]))
            _z['x'] = x_mean
            if not x.has_key('x'):
                x['x'] = np.random.normal(x_mean, np.exp(x_logvar/2))
                if self.type_px == 'bounded01':
                    x['x'] = np.maximum(np.zeros(x['x'].shape), x['x'])
                    x['x'] = np.minimum(np.ones(x['x'].shape), x['x'])
        
        else: raise Exception("")
        
        return x, z, _z
    
    def variables(self):
        
        z = {}

        # Define observed variables 'x'
        x = {'x': T.fmatrix('x'), 'y': T.fmatrix('y')}
        
        return x, z
    
    def init_w(self, std=1e-2):
        
        def rand(size):
            if len(size) == 2 and size[1] > 1:
                return np.random.normal(0, 1, size=size) / np.sqrt(size[1])
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
            if self.type_px in ['gaussian', 'laplace']:
                w['out_logvar_w'] = rand((self.n_x, self.n_hidden_p[-1]))
                w['out_logvar_b'] = np.zeros((self.n_x, 1))
            if self.type_px == 'bounded01':
                w['out_logvar_b'] = np.zeros((self.n_x, 1))
                w['out_unif'] = np.zeros((self.n_x, 1))   
        else:
            w['out_w'] = rand((self.n_x, self.n_z))
            w['out_b'] = np.zeros((self.n_x, 1))
            if self.type_px == 'gaussian':
                w['out_logvar_w'] = rand((self.n_x, self.n_z))
                w['out_logvar_b'] = np.zeros((self.n_x, 1))
            if self.type_px == 'bounded01':
                w['out_logvar_b'] = np.zeros((self.n_x, 1))
                w['out_unif'] = np.zeros((self.n_x, 1))

        #w['logpy'] = np.zeros((self.n_y, 1))

        if self.type_pz == 'studentt':
            w['logv'] = np.zeros((self.n_z, 1))

        return v, w
