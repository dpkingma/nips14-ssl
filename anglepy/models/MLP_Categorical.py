import theano
import theano.tensor as T
import numpy as np

import sys; sys.path.append('../../shared')
import anglepy as ap
from anglepy.misc import lazytheanofunc
import anglepy.ndict as ndict

# Write anglepy model
class MLP_Categorical(ap.BNModel):
    def __init__(self, n_units, prior_sd=1, nonlinearity='tanh'):
        self.n_units = n_units
        self.prior_sd = prior_sd
        self.nonlinearity = nonlinearity
        super(MLP_Categorical, self).__init__()
    
    def variables(self):
        w = {}
        for i in range(len(self.n_units)-1):
            w['w'+str(i)] = T.dmatrix()
            w['b'+str(i)] = T.dmatrix()
        
        x = {}
        x['x'] = T.dmatrix()
        x['y'] = T.dmatrix()
        
        z = {}
        return w, x, z
    
    def factors(self, w, x, z, A):
        # Define logp(w)
        logpw = 0
        for i in range(len(self.n_units)-1):
            logpw += ap.logpdfs.normal(w['w'+str(i)], 0, self.prior_sd).sum()
            logpw += ap.logpdfs.normal(w['b'+str(i)], 0, self.prior_sd).sum()
        
        if self.nonlinearity == 'tanh':
            f = T.tanh
        elif self.nonlinearity == 'sigmoid':
            f = T.nnet.sigmoid
        elif self.nonlinearity == 'softplus':
            f = T.nnet.softplus
        else:
            raise Exception("Unknown nonlinarity "+self.nonlinearity)
        
        # Define logp(x)
        hiddens  = [T.dot(w['w0'], x['x']) + T.dot(w['b0'], A)]
        for i in range(1, len(self.n_units)-1):
            hiddens.append(T.dot(w['w'+str(i)], f(hiddens[-1])) + T.dot(w['b'+str(i)], A))
        
        self.p = T.nnet.softmax(hiddens[-1].T).T
        self.entropy = T.nnet.categorical_crossentropy(self.p.T, self.p.T).T
        
        logpx = (- T.nnet.categorical_crossentropy(self.p.T, x['y'].T).T).reshape((1,-1))
        
        # function for distribution q(z|x)
        theanofunc = lazytheanofunc('ignore', mode='FAST_RUN')
        self.dist_px['y'] = theanofunc([x['x']] + w.values() + [A], self.p)
        
        logpz = 0 * A
        return logpw, logpx, logpz
        
    def gen_xz(self, w, x, z, n_batch=0):
        if not x.has_key('x'):
            raise Exception('Not implemented')
        
        if n_batch == 0:
            n_batch = x['x'].shape[1]
        A = np.ones((1, n_batch))
        
        _z = {}
        if not x.has_key('y'):
            w = ndict.ordered(w)
            py = self.dist_px['y'](*([x['x']] + w.values() + [A]))
            _z['py'] = py
            x['y'] = np.zeros(py.shape)
            for i in range(n_batch):
                x['y'][:,i] = np.random.multinomial(n=1, pvals=py[:,i])
            
        return x, z, _z
    
    def init_w(self, init_sd=1e-2):
        w = {}
        for i in range(len(self.n_units)-1):
            w['w'+str(i)] = init_sd * np.random.standard_normal((self.n_units[i+1], self.n_units[i]))
            w['b'+str(i)] = np.zeros((self.n_units[i+1], 1))
            
        return w
