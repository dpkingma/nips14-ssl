import numpy as np
import theano
import theano.tensor as T
import math
import theano.compile
import anglepy.ndict as ndict
from anglepy.misc import lazytheanofunc
import anglepy.logpdfs
import inspect

# ====
# VARIATIONAL AUTO-ENCODER MODEL
# ====

# Model
class VAEModel(object):
    
    def __init__(self, theano_warning='raise'):
        
        theanofunction = lazytheanofunc('warn', mode='FAST_RUN')
        theanofunction_silent = lazytheanofunc('ignore', mode='FAST_RUN')
        
        # Create theano expressions
        v, w, x, z = ndict.ordereddicts(self.variables())
        self.var_v, self.var_w, self.var_x, self.var_z, = v, w, x, z
        
        # Helper variables
        A = T.dmatrix('A')
        self.var_A = A
        
        # Get gradient symbols
        allvars = v.values() + w.values() + x.values() + z.values() + [A] # note: '+' concatenates lists
        
        # TODO: more beautiful/standardized way of setting distributions
        # (should be even simpler than this) 
        self.dist_qz = {}
        self.dist_px = {}
        self.dist_pz = {}
        
        logpv, logpw, logpx, logpz, logqz = self.factors(v, w, x, z, A)
        
        # Log-likelihood lower bound
        self.f_L = theanofunction(allvars, [logpx, logpz, logqz])
        L = (logpx + logpz - logqz).sum()
        dL_dw = T.grad(L, v.values() + w.values())
        self.f_dL_dw = theanofunction(allvars, [logpx, logpz, logqz] + dL_dw)
        
        weights = T.dmatrix()
        dL_weighted_dw = T.grad((weights * (logpx + logpz - logqz)).sum(), v.values() + w.values())
        self.f_dL_weighted_dw = theanofunction(allvars + [weights], [logpx + logpz - logqz, weights*(logpx + logpz - logqz)] + dL_weighted_dw)
        
        # prior
        dlogpw_dw = T.grad(logpv + logpw, v.values() + w.values(), disconnected_inputs='ignore')
        self.f_logpw = theanofunction(v.values() + w.values(), [logpv, logpw])
        self.f_dlogpw_dw = theanofunction(v.values() + w.values(), [logpv, logpw] + dlogpw_dw)
        
        # distributions
        #self.f_dists = {}
        #for name in dists:
        #    _vars, dist = dists[name]
        #    self.f_dists[name] = theanofunction_silent(_vars, dist)
        
    # NOTE: IT IS ESSENTIAL THAT DICTIONARIES OF SYMBOLIC VARS AND RESPECTIVE NUMPY VALUES HAVE THE SAME KEYS
    # (OTHERWISE FUNCTION ARGUMENTS ARE IN INCORRECT ORDER)
    
    def variables(self): raise NotImplementedError()
    def factors(self): raise NotImplementedError()
    def gen_xz(self): raise NotImplementedError()
    def init_w(self): raise NotImplementedError()
    
    # Prediction
    # TODO: refactor to new solution
    def distribution(self, v, w, x, z, name):
        x, z = self.xz_to_theano(x, z)
        v, w, x, z = ndict.ordereddicts((v, w, x, z))
        A = self.get_A(x)
        allvars = v.values() + w.values() + x.values() + z.values() + [A]
        return self.f_dists[name](*allvars)
    
    # Numpy <-> Theano var conversion
    def xz_to_theano(self, x, z): return x, z
    def gw_to_numpy(self, gv, gw): return gv, gw
    
    # A = np.ones((1, n_batch))
    def get_A(self, x): return np.ones((1, x.itervalues().next().shape[1]))
        
    # Likelihood: logp(x,z|w)
    def L(self, v, w, x, z):
        x, z = self.xz_to_theano(x, z)
        v, w, z, x = ndict.ordereddicts((v, w, z, x))
        A = self.get_A(x)
        allvars = v.values() + w.values() + x.values() + z.values() + [A]
        logpx, logpz, logqz = self.f_L(*allvars)
        
        if np.isnan(logpx).any() or np.isnan(logpz).any() or np.isnan(logqz).any():
            print 'logp: ', logpx, logpz, logqz
            print 'Values:'
            ndict.p(v)
            ndict.p(w)
            ndict.p(x)
            ndict.p(z)
            raise Exception("delbo_dwz(): NaN found in gradients")
        
        return logpx, logpz, logqz
    
    
    def checknan(self, v, w, gv, gw):
        
        if ndict.hasNaN(gv) or ndict.hasNaN(gw):
                raise Exception("dL_dw(): NaN found in gradients")
                #print 'logpx: ', logpx
                #print 'logpz: ', logpz
                #print 'logqz: ', logqz
                print 'v:'
                ndict.p(v)
                print 'w:'
                ndict.p(w)
                print 'gv:'
                ndict.p(gv)
                print 'gw:'
                ndict.p(gw)
                raise Exception("dL_dw(): NaN found in gradients")
        
    # Gradient of logp(x,z|w) and logq(z) w.r.t. parameters
    def dL_dw(self, v, w, x, z):
        x, z = self.xz_to_theano(x, z)
        v, w, z, x = ndict.ordereddicts((v, w, z, x))
        A = self.get_A(x)
        allvars = v.values() + w.values() + x.values() + z.values() + [A]
        r = self.f_dL_dw(*allvars)
        logpx, logpz, logqz, gv, gw = r[0], r[1], r[2], dict(zip(v.keys(), r[3:3+len(v)])), dict(zip(w.keys(), r[3+len(v):3+len(v)+len(w)]))
        self.checknan(v, w, gv, gw)        
        gv, gw = self.gw_to_numpy(gv, gw)
        return logpx, logpz, logqz, gv, gw

    # Gradient of logp(x,z|w) and logq(z) w.r.t. parameters
    def dL_weighted_dw(self, v, w, x, z, weights):
        x, z = self.xz_to_theano(x, z)
        v, w, z, x = ndict.ordereddicts((v, w, z, x))
        A = self.get_A(x)
        allvars = v.values() + w.values() + x.values() + z.values() + [A]
        r = self.f_dL_weighted_dw(*(allvars+[weights]))
        L_unweighted, L_weighted, gv, gw = r[0], r[1], dict(zip(v.keys(), r[2:2+len(v)])), dict(zip(w.keys(), r[2+len(v):2+len(v)+len(w)]))
        self.checknan(v, w, gv, gw)
        gv, gw = self.gw_to_numpy(gv, gw)
        return L_unweighted, L_weighted, gv, gw
    
    # Prior: logp(w)
    def logpw(self, v, w):
        logpv, logpw = self.f_logpw(*ndict.orderedvals((v,w)))
        return logpv, logpw
    
    # Gradient of the prior: logp(w)
    def dlogpw_dw(self, v, w):
        r = self.f_dlogpw_dw(*ndict.orderedvals((v,w)))
        v, w = ndict.ordereddicts((v, w))
        return r[0], r[1], dict(zip(v.keys(), r[2:2+len(v)])), dict(zip(w.keys(), r[2+len(v):2+len(v)+len(w)]))
    
    # Helper function that creates tiled version of datapoint 'x' (* n_batch)
    def tiled_x(self, x, n_batch):
        x_tiled = {}
        for i in x:
            if (x[i].shape[1] != 1):
                raise Exception("{} {} {} ".format(x[i].shape[0], x[i].shape[1], n_batch))
            x_tiled[i] = np.dot(x[i], np.ones((1, n_batch)))
        return x_tiled
    