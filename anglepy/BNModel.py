import numpy as np
import theano
import theano.tensor as T
import math
import theano.compile
import anglepy.ndict as ndict
from anglepy.misc import lazytheanofunc
import anglepy.logpdfs
import inspect

# Model
class BNModel(object):
    
    def __init__(self, theano_warning='raise', hessian=True):
        
        theanofunction = lazytheanofunc('warn', mode='FAST_RUN')
        theanofunction_silent = lazytheanofunc('ignore', mode='FAST_RUN')
        
        # Create theano expressions
        w, x, z = ndict.ordereddicts(self.variables())
        self.var_w, self.var_x, self.var_z, = w, x, z
        
        # Helper variables
        A = T.dmatrix('A')
        self.var_A = A
        
        # Get gradient symbols
        self.allvars = w.values()  + x.values() + z.values() + [A] # note: '+' concatenates lists
        self.allvars_keys = w.keys() + x.keys() + z.keys() + ['A']
        
        if False:
            # Put test values
            # needs fully implemented gen_xz(), which is not always the case
            # Also, the FD has no test values
            theano.config.compute_test_value = 'raise'
            _w = self.init_w()
            for i in _w: w[i].tag.test_value = _w[i]
            _x, _z, _ = self.gen_xz(_w, {}, {}, 10)
            _x, _z = self.xz_to_theano(_x, _z)
            for i in _x: x[i].tag.test_value = _x[i]
            for i in _z: z[i].tag.test_value = _z[i]
        
        # TODO: more beautiful/standardized way of setting distributions
        # (should be even simpler then this) 
        self.dist_px = {}
        self.dist_pz = {}
        
        logpw, logpx, logpz = self.factors(w, x, z, A)
        self.var_logpw, self.var_logpx, self.var_logpz = logpw, logpx, logpz
        
        # Complete-data likelihood estimate
        logpxz = logpx.sum() + logpz.sum()
        self.f_logpxz = theanofunction(self.allvars, [logpx, logpz])
        
        dlogpxz_dwz = T.grad(logpxz, w.values() + z.values())
        self.f_dlogpxz_dwz = theanofunction(self.allvars, [logpx, logpz] + dlogpxz_dwz)
        #self.f_dlogpxz_dw = theanofunction(allvars, [logpxz] + dlogpxz_dw)
        #self.f_dlogpxz_dz = theanofunction(allvars, [logpxz] + dlogpxz_dz)
        
        # prior
        dlogpw_dw = T.grad(logpw, w.values(), disconnected_inputs='ignore')
        self.f_logpw = theanofunction(w.values(), logpw)
        self.f_dlogpw_dw = theanofunction(w.values(), [logpw] + dlogpw_dw)
        
        if False:
            # MC-LIKELIHOOD
            logpx_max = logpx.max()
            logpxmc = T.log(T.exp(logpx - logpx_max).mean()) + logpx_max
            self.f_logpxmc = theanofunction(self.allvars, logpxmc)
            dlogpxmc_dw = T.grad(logpxmc, w.values(), disconnected_inputs=theano_warning)
            self.f_dlogpxmc_dw = theanofunction(self.allvars, [logpxmc] + dlogpxmc_dw)
        
        if True and len(z) > 0:
            # Fisher divergence (FD)
            gz = T.grad(logpxz, z.values())
            gz2 = [T.dmatrix() for _ in gz]
            fd = 0
            for i in range(len(gz)):
                fd += T.sum((gz[i]-gz2[i])**2)
            dfd_dw = T.grad(fd, w.values())
            self.f_dfd_dw = theanofunction(self.allvars + gz2, [logpx, logpz, fd] + dfd_dw)
            
        if False and hessian:
            # Hessian of logpxz wrt z (works best with n_batch=1)
            hessian_z = theano.gradient.hessian(logpxz, z_concat)
            self.f_hessian_z = theanofunction(self.allvars, hessian_z)
        
    # NOTE: IT IS ESSENTIAL THAT DICTIONARIES OF SYMBOLIC VARS AND RESPECTIVE NUMPY VALUES HAVE THE SAME KEYS
    # (OTHERWISE FUNCTION ARGUMENTS ARE IN INCORRECT ORDER)
    
    def variables(self): raise NotImplementedError()
    def factors(self): raise NotImplementedError()
    def gen_xz(self): raise NotImplementedError()
    def init_w(self): raise NotImplementedError()
    
    # Prediction
    #def distribution(self, w, x, z, name):
    #    x, z = self.xz_to_theano(x, z)
    #    w, x, z = ndict.ordereddicts((w, x, z))
    #    A = self.get_A(x)
    #    allvars = w.values() + x.values() + z.values() + [A]
    #    return self.f_dists[name](*allvars)
    
    # Numpy <-> Theano var conversion
    def xz_to_theano(self, x, z): return x, z
    def gwgz_to_numpy(self, gw, gz): return gw, gz
    
    # A = np.ones((1, n_batch))
    def get_A(self, x): return np.ones((1, x.itervalues().next().shape[1]))
        
    # Likelihood: logp(x,z|w)
    def logpxz(self, w, x, z):
        x, z = self.xz_to_theano(x, z)
        w, z, x = ndict.ordereddicts((w, z, x))
        A = self.get_A(x)
        allvars = w.values() + x.values() + z.values() + [A]
        logpx, logpz = self.f_logpxz(*allvars)
        if np.isnan(logpx).any() or np.isnan(logpz).any():
            print 'v: ', logpx, logpz
            print 'Values:'
            ndict.p(w)
            ndict.p(z)
            raise Exception("dlogpxz_dwz(): NaN found in gradients")
        
        return logpx, logpz
    
    # Gradient of logp(x,z|w) w.r.t. parameters and latent variables
    def dlogpxz_dwz(self, w, x, z):
        
        x, z = self.xz_to_theano(x, z)
        w, z, x = ndict.ordereddicts((w, z, x))
        A = self.get_A(x)
        allvars = w.values() + x.values() + z.values() + [A]
        
        # Check if keys are correct
        keys = w.keys() + x.keys() + z.keys() + ['A']
        for i in range(len(keys)):
            if keys[i] != self.allvars_keys[i]:
                "Input values are incorrect!"
                print 'Input:', keys
                print 'Should be:', self.allvars_keys
                raise Exception()
            
        r = self.f_dlogpxz_dwz(*allvars)
        logpx, logpz, gw, gz = r[0], r[1], dict(zip(w.keys(), r[2:2+len(w)])), dict(zip(z.keys(), r[2+len(w):]))
        
        if ndict.hasNaN(gw) or ndict.hasNaN(gz):
            if True:
                print 'NaN detected in gradients'
                raise Exception()
                for i in gw: gw[i][np.isnan(gw[i])] = 0
                for i in gz: gz[i][np.isnan(gz[i])] = 0
            else:
                print 'logpx: ', logpx
                print 'logpz: ', logpz
                print 'Values:'
                ndict.p(w)
                ndict.p(z)
                print 'Gradients:'
                ndict.p(gw)
                ndict.p(gz)
                raise Exception("dlogpxz_dwz(): NaN found in gradients")
        
        gw, gz = self.gwgz_to_numpy(gw, gz)
        return logpx, logpz, gw, gz
    
    '''
    # Gradient of logp(x,z|w) w.r.t. parameters
    def dlogpxz_dw(self, w, z, x):
        w, z, x = ndict.ordereddicts((w, z, x))
        r = self.f_dlogpxz_dw(*(w.values() + z.values() + x.values()))
        return r[0], dict(zip(w.keys(), r[1:]))
    
    # Gradient of logp(x,z|w) w.r.t. latent variables
    def dlogpxz_dz(self, w, z, x):
        w, z, x = ndict.ordereddicts((w, z, x))
        r = self.f_dlogpxz_dz(*(w.values() + z.values() + x.values()))
        return r[0], dict(zip(z.keys(), r[1:]))
    '''
    
    # Hessian of logpxz wrt z (works best with n_batch=1)
    def hessian_z(self, w, z, x):
        x, z = self.xz_to_theano(x, z)
        A = self.get_A(x)
        return self.f_hessian_z(*ndict.orderedvals((w, x, z))+[A])

    # Prior: logp(w)
    def logpw(self, w):
        logpw = self.f_logpw(*ndict.orderedvals((w,)))
        return logpw
    
    # Gradient of the prior: logp(w)
    def dlogpw_dw(self, w):
        w = ndict.ordered(w)
        r = self.f_dlogpw_dw(*(w.values()))
        return r[0], dict(zip(w.keys(), r[1:]))
    
    # MC likelihood: logp(x|w)
    def logpxmc(self, w, x, n_batch):
        x = self.tiled_x(x, n_batch)
        x, z, _ = self.gen_xz(w, x, {}, n_batch=n_batch)
        x, z = self.xz_to_theano(x, z)
        A = self.get_A(x)
        logpxmc = self.f_logpxmc(*ndict.orderedvals((w, x, z))+[A])
        return logpxmc
    
    # Gradient of MC likelihood logp(x|w) w.r.t. parameters
    def dlogpxmc_dw(self, w, x, n_batch):
        x = self.tiled_x(x, n_batch) 
        x, z, _ = self.gen_xz(w, x, {}, n_batch=n_batch)
        x, z = self.xz_to_theano(x, z)
        A = self.get_A(x)
        r = self.f_dlogpxmc_dw(*ndict.orderedvals((w, x, z))+[A])
        return r[0], dict(zip(ndict.ordered(w).keys(), r[1:]))
    
    # Gradient w.r.t. the Fisher divergence
    def dfd_dw(self, w, x, z, gz2):
        x, z = self.xz_to_theano(x, z)
        w, z, x, gz2 = ndict.ordereddicts((w, z, x, gz2))
        A = self.get_A(x)
        r = self.f_dfd_dw(*(w.values() + x.values() + z.values() + [A] + gz2.values()))
        logpx, logpz, fd, gw = r[0], r[1], r[2], dict(zip(w.keys(), r[3:3+len(w)]))
        
        if ndict.hasNaN(gw):
            if True:
                print 'NaN detected in gradients'
                raise Exception()
                for i in gw: gw[i][np.isnan(gw[i])] = 0
            else:
                
                print 'fd: ', fd
                print 'Values:'
                ndict.p(w)
                ndict.p(z)
                print 'Gradients:'
                ndict.p(gw)
                raise Exception("dfd_dw(): NaN found in gradients")
        
        gw, _ = self.gwgz_to_numpy(gw, {})
        return logpx, logpz, fd, gw
    
    # Helper function that creates tiled version of datapoint 'x' (* n_batch)
    def tiled_x(self, x, n_batch):
        x_tiled = {}
        for i in x:
            if (x[i].shape[1] != 1):
                raise Exception("{} {} {} ".format(x[i].shape[0], x[i].shape[1], n_batch))
            x_tiled[i] = np.dot(x[i], np.ones((1, n_batch)))
        return x_tiled
    