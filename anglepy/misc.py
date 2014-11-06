import numpy as np
import theano
import theano.tensor as T

def random_orthogonal(dim, special=True):
	if dim == 1:
		if np.random.uniform() < 0.5:
			return np.ones((1,1))
		return -np.ones((1,1))
	P = np.random.randn(dim, dim)
	while np.linalg.matrix_rank(P) != dim:
		P = np.random.randn(dim, dim)
	U, S, V = np.linalg.svd(P)
	P = np.dot(U, V)
	if special:
		# Make sure det(P) == 1
		if np.linalg.det(P) < 0:
			P[:, [0, 1]] = P[:, [1, 0]]
	return P

# Code From:
# https://gist.github.com/benanne/2025317
# (Checked for correctness against scipy)

# Log-gamma function for theano
def log_gamma_lanczos(z):
    # reflection formula. Normally only used for negative arguments,
    # but here it's also used for 0 < z < 0.5 to improve accuracy in this region.
    flip_z = 1 - z
    # because both paths are always executed (reflected and non-reflected),
    # the reflection formula causes trouble when the input argument is larger than one.
    # Note that for any z > 1, flip_z < 0.
    # To prevent these problems, we simply set all flip_z < 0 to a 'dummy' value.
    # This is not a problem, since these computations are useless anyway and
    # are discarded by the T.switch at the end of the function.
    flip_z = T.switch(flip_z < 0, 1, flip_z)
    small = np.log(np.pi) - T.log(T.sin(np.pi * z)) - log_gamma_lanczos_sub(flip_z)
    big = log_gamma_lanczos_sub(z)
    return T.switch(z < 0.5, small, big)
   
## version that isn't vectorised, since g is small anyway
def log_gamma_lanczos_sub(z): #expanded version
    # Coefficients used by the GNU Scientific Library
    g = 7
    p = np.array([0.99999999999980993, 676.5203681218851, -1259.1392167224028,
                  771.32342877765313, -176.61502916214059, 12.507343278686905,
                  -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7])                    
    z = z - 1
    x = p[0]
    for i in range(1, g+2):
        x += p[i]/(z+i)
    t = z + g + 0.5
    return np.log(np.sqrt(2*np.pi)) + (z + 0.5) * T.log(t) - t + T.log(x)

'''
def log_gamma_lanczos_sub(z): #vectorised version
    # Coefficients used by the GNU Scientific Library
    g = 7
    p = np.array([0.99999999999980993, 676.5203681218851, -1259.1392167224028,
                  771.32342877765313, -176.61502916214059, 12.507343278686905,
                  -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7])  
    z = z - 1
    
    zs = T.shape_padleft(z, 1) + T.shape_padright(T.arange(1, g+2), z.ndim)
    x = T.sum(T.shape_padright(p[1:], z.ndim) / zs, axis=0) + p[0]
	t = z + g + 0.5
    return np.log(np.sqrt(2*np.pi)) + (z + 0.5) * T.log(t) - t + T.log(x)
'''

# Lazy function compilation
# (it only gets compiled when it's actually called)
def lazytheanofunc(on_unused_input='warn', mode='FAST_RUN'):
	def theanofunction(*args, **kwargs):
		f = [None]
		if not kwargs.has_key('on_unused_input'):
			kwargs['on_unused_input'] = on_unused_input
		if not kwargs.has_key('mode'):
			kwargs['mode'] = mode
		def func(*args2, **kwargs2):
			if f[0] == None:
				f[0] = theano.function(*args, **kwargs)
			return f[0](*args2, **kwargs2)
		return func
	return theanofunction


