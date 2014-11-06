'''
Author: D.P. Kingma (2014)
This software is made available under the MIT License.
http://opensource.org/licenses/MIT

This initialization bias correction used in this algorithm is not yet published at the time of writing.
'''
import numpy as np

class AdaM(object):
    
    def __init__(self, f_df, w, minibatches, alpha=3e-4, beta1=0.9, beta2=0.999):
        self.f_df = f_df
        self.w = w
        self.minibatches = minibatches
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.m1 = {}
        self.m2 = {}
        self.t = 0
        for i in w:
            self.m1[i] = {}
            self.m2[i] = {}
            for j in w[i]:
                self.m1[i][j] = np.zeros(w[i][j].shape)
                self.m2[i][j] = np.zeros(w[i][j].shape)
    
    '''
    Do num_passes epochs
    '''
    def optimize(self, num_passes=1):
        f = 0
        for i_pass in range(num_passes):
            for i_batch in range(len(self.minibatches)):
                _f = self.optim_minibatch(i_batch)
                f += _f
        f /= 1. * num_passes
        return self.w
    
    '''
    Do a minibatch
    '''
    def optim_minibatch(self, i_batch):
        f, g = self.f_df(self.w, self.minibatches[i_batch])

        self.t += 1
        alpha_t = self.alpha * np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)
        
        # Update moments and parameters
        for i in g:
            for j in g[i]:
                self.m1[i][j] += (1-self.beta1) * (g[i][j] - self.m1[i][j])
                self.m2[i][j] += (1-self.beta2) * (g[i][j]**2 - self.m2[i][j])
                self.w[i][j] -= self.alpha * self.m1[i][j] / np.sqrt(self.m2[i][j])
        
        return f
    