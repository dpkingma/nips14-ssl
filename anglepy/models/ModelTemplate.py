import theano
import theano.tensor as T
import anglepy as ap

# Template for BNModel
class Model(ap.BNModel):
    def __init__(self):
        super(Model, self).__init__()
    
    def variables(self):
        w = {}
        x = {}
        z = {}
        return w, x, z
    
    def factors(self, w, x, z, A):
        logpw = 0
        logpx = 0
        logpz = 0
        return logpw, logpx, logpz, {}
        
    def gen_xz(self, w, x, z):
    	return x, z, {}

    def init_w(self, init_sd=1):
        return {}
      
    
