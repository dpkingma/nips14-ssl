import numpy as np
import matplotlib.pyplot as plt
import anglepy.ndict as ndict
import anglepy.mcmc
import theano.tensor as T
import theano

'''
MCMC tester
'''
	
def mcmc_test():
	
	np.random.seed(6)

	# Gaussian parameters
	n_dim = 10
	n_batch = 1000
	
	# Init multivariate Gaussian params
	mu = np.random.uniform(size=(n_dim, 1)) * 1
	mu_ = np.dot(mu, 1.*np.ones((1, n_batch)))
	cov = np.random.standard_normal(size=(n_dim, n_dim)) * 1
	cov = np.dot(cov, cov.T)
	#cov[np.arange(n_dim), np.arange(n_dim)] = 1.0
	cov_inv = np.linalg.inv(cov)
	eigenvalues, _ = np.linalg.eig(cov)
	print eigenvalues
	#raise Exception

	# Differentiate Gaussian PDF with Theano
	x = T.dmatrix()
	logpx = - 0.5 * (T.dot(cov_inv, x-mu_) * (x-mu_)).sum(axis=0)
	f_logpx = theano.function([x], [logpx])
	f_dlogpxdx = theano.function([x], [logpx, T.grad(logpx.sum(), x)])
	
	# Init MCMC sampler
	def f(_x):
		return f_logpx(_x['x'])
	def fgrad(_x):
		v, grad = f_dlogpxdx(_x['x'])
		return v, {'x':grad}
	
	stepsize = [1e-6]
	dostep = anglepy.mcmc.hmc_step_autotune(n_steps=20)
	#dostep = anglepy.mcmc.hmc_step_autotune2(n_steps=20)
	def mcmc_dostep(_x):
		return dostep(f, fgrad, _x)
	
	# Sample statistics computation
	def compute_stats(samples):
		mu_sample = samples.mean(axis=1, keepdims=True)
		cov_sample = np.cov(samples)
		mu_error = ((mu_sample-mu)**2).sum()
		cov_error = ((cov-cov_sample)**2).sum()
		return mu_sample.T, cov_sample, mu_error + cov_error
	
	# Start sampling
	x = {'x':np.random.normal(scale=0.1, size=(n_dim, n_batch))}
	
	# Burn-in
	for i in xrange(200):
		mcmc_dostep(x)
	samples = []
	for i in xrange(10000):
		acceptrate, _stepsize = mcmc_dostep(x)
		samples.append(np.copy(x['x']))
		if i%100 == 0:
			#print acceptrate, _stepsize, x['x'].mean()
			_, _, error = compute_stats(np.hstack(tuple(samples)))
			print 'Sample: ', i, 'Acceptrate: ', acceptrate, 'Stepsize:', _stepsize, 'Error:', error
		
	samples = np.hstack(tuple(samples))
	
	# Report
	mu_sample, cov_sample, error = compute_stats(samples)
	print '=== Target values'
	print 'true mean:', mu.T
	print 'true cov:\n', cov
	print '=== Empirical values'
	print 'empirial mean:\n', mu_sample
	print 'epmirical cov:\n', cov_sample
	print 'Error:', 
	