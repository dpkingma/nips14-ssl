import numpy as np
import anglepy.ndict as ndict

# FuncLikelihood
class FuncLikelihood():
	
	def __init__(self, x, model, n_batch):
		self.x = x
		self.model = model
		self.n_batch = n_batch
		self.n_datapoints = x.itervalues().next().shape[1]
		if self.n_datapoints%(self.n_batch) != 0:
			print self.n_datapoints, self.n_batch
			raise BaseException()
		self.blocksize = self.n_batch
		self.n_minibatches = self.n_datapoints/self.blocksize
	
	def subval(self, i, w, z):
		_x = ndict.getCols(self.x, i*self.n_batch, (i+1)*self.n_batch)
		_z = ndict.getCols(z, i*self.n_batch, (i+1)*self.n_batch)
		return self.model.logpxz(w, _x, _z)
	
	def subgrad(self, i, w, z):
		_x = ndict.getCols(self.x, i*self.n_batch, (i+1)*self.n_batch)
		_z = ndict.getCols(z, i*self.n_batch, (i+1)*self.n_batch)
		logpx, logpz, g, _ = self.model.dlogpxz_dwz(w, _x, _z)
		return logpx, logpz, g
	
	def val(self, w, z):
		if self.n_minibatches==1: return self.subval(0, w, z)
		logpx, logpz = tuple(zip(*[self.subval(i, w, z) for i in range(self.n_minibatches)]))
		return np.hstack(logpx), np.hstack(logpz)
	
	def grad(self, w, z):
		if self.n_minibatches==1: return self.subgrad(0, w, z)
		logpxi, logpzi, gwi, _ = tuple(zip(*[self.subgrad(i, w, z) for i in range(self.n_minibatches)]))
		return np.hstack(logpxi), np.hstack(logpzi), ndict.sum(gwi)
	
# FuncPosterior	
class FuncPosterior():
	def __init__(self, likelihood, model):
		self.ll = likelihood
		self.model = model
		self.n_minibatches = likelihood.n_minibatches
		self.blocksize = likelihood.blocksize
	
	def subval(self, i, w, z):
		prior = self.model.logpw(w)
		prior_weight = 1. / float(self.ll.n_minibatches)
		logpx, logpz = self.ll.subval(i, w, z)
		return logpx.sum() + logpz.sum() + prior_weight * prior
	
	def subgrad(self, i, w, z):
		logpx, logpz, gw = self.ll.subgrad(i, w, z)
		logpw, gw_prior = self.model.dlogpw_dw(w)
		prior_weight = 1. / float(self.ll.n_minibatches)
		for j in gw: gw[j] += prior_weight * gw_prior[j]
		return logpx.sum() + logpz.sum() + prior_weight * logpw, gw
	
	def val(self, w, z={}):
		logpx, logpz = self.ll.val(w, z)
		return logpx.sum() + logpz.sum() + self.model.logpw(w)
	
	def grad(self, w, z={}):
		logpx, logpz, gw = self.ll.grad(w, z)
		prior, gw_prior = self.model.dlogpw_dw(w)
		for i in gw: gw[i] += gw_prior[i]
		return logpx.sum() + logpz.sum() + prior, gw
	

# Parallel version of likelihood

# Before using, start ipython cluster, e.g.:
# shell>ipcluster start -n 4
from IPython.parallel.util import interactive
import IPython.parallel
class FuncLikelihoodPar():
	def __init__(self, x, model, n_batch):
		raise Exception("TODO")
		
		self.x = x
		self.c = c = IPython.parallel.Client()
		self.model = model
		self.n_batch = n_batch
		self.clustersize = len(c)
		
		print 'ipcluster size = '+str(self.clustersize)
		n_train = x.itervalues().next().shape[1]
		if n_train%(self.n_batch*len(c)) != 0: raise BaseException()
		self.blocksize = self.n_batch*len(c)
		self.n_minibatches = n_train/self.blocksize
		
		# Get pointers to slaves
		c.block = False
		# Remove namespaces on slaves
		c[:].clear()
		# Execute stuff on slaves
		module, function, args = self.model.constr
		c[:].push({'args':args,'x':x}).wait()
		commands = [
				'import os; cwd = os.getcwd()',
				'import sys; sys.path.append(\'../shared\')',
				'import anglepy.ndict as ndict',
				'import '+module,
				'my_n_batch = '+str(n_batch),
				'my_model = '+module+'.'+function+'(**args)'
		]
		for cmd in commands: c[:].execute(cmd).get()
		# Import data on slaves
		for i in range(len(c)):
			_x = ndict.getCols(x, i*(n_train/len(c)), (i+1)*(n_train/len(c)))
			c[i].push({'my_x':_x})
		c[:].pull(['my_x']).get()
		
	def subval(self, i, w, z):
		raise Exception("TODO")
		
		# Replaced my_model.nbatch with my_n_batch, this is UNTESTED
		
		@interactive
		def ll(w, z, k):
			_x = ndict.getCols(my_x, k*my_n_batch, (k+1)*my_n_batch) #@UndefinedVariable
			if z == None:
				return my_model.logpxmc(w, _x), None #@UndefinedVariable
			else:
				return my_model.logpxz(w, _x, z) #@UndefinedVariable
		
		tasks = []
		for j in range(len(self.c)):
			_z = z
			if _z != None:
				_z = ndict.getCols(z, j*self.n_batch, (j+1)*self.n_batch)
			tasks.append(self.c.load_balanced_view().apply_async(ll, w, _z, i))
		
		res = [task.get() for task in tasks]
		
		raise Exception("TODO: implementation with uncoupled logpx and logpz")
		return sum(res)
	
	def subgrad(self, i, w, z):
		
		@interactive
		def dlogpxz_dwz(w, z, k):
			_x = ndict.getCols(my_x, k*my_n_batch, (k+1)*my_n_batch).copy() #@UndefinedVariable
			if z == None:
				logpx, gw = my_model.dlogpxmc_dw(w, _x) #@UndefinedVariable
				return logpx, None, gw, None
			else:
				return my_model.dlogpxz_dwz(w, _x, z) #@UndefinedVariable
		
		tasks = []
		for j in range(len(self.c)):
			_z = z
			if _z != None:
				_z = ndict.getCols(z, j*self.n_batch, (j+1)*self.n_batch)
			tasks.append(self.c.load_balanced_view().apply_async(dlogpxz_dwz, w, _z, i))
		
		res = [task.get() for task in tasks]
		
		v, gw, gz = res[0]
		for k in range(1,len(self.c)):
			vi, gwi, gzi = res[k]
			v += vi
			for j in gw: gw[j] += gwi[j]
			for j in gz: gz[j] += gzi[j]
		return v, gw, gz
	

	def grad(self, w, z=None):
		v, gw, gz = self.subgrad(0, w, z)
		for i in range(1, self.n_minibatches):
			vi, gwi, gzi = self.subgrad(i, w, z)
			v += vi
			for j in gw: gw[j] += gwi[j]
			for j in gz: gz[j] += gzi[j]
		return v, gw, gz
	
	def val(self, w, z=None):
		logpx, logpz = self.subval(0, w, z)
		for i in range(1, self.n_minibatches):
			_logpx, _logpz = self.subval(i, w, z)
			logpx += _logpx
			logpz += _logpz
		return logpx, logpz
	
	def grad(self, w, z=None):
		logpx, logpz, gw, gz = self.subgrad(0, w, z)
		for i in range(1, self.n_minibatches):
			logpxi, logpzi, gwi, gzi = self.subgrad(i, w, z)
			logpx += logpxi
			logpz += logpzi
			for j in gw: gw[j] += gwi[j]
			for j in gz: gz[j] += gzi[j]
		return logpx, logpz, gw, gz
	
	# Helper function
	def getColsZX(self, w, z, i):
		_x = ndict.getCols(self.x, i*self.n_batch, (i+1)*self.n_batch)
		if z != None:
			_z = ndict.getCols(z, i*self.n_batch, (i+1)*self.n_batch)
		return _z, _x

# Monte Carlo FuncLikelihood
class FuncLikelihoodMC():
	
	def __init__(self, x, model, n_mc_samples):
		self.x = x
		self.model = model
		self.n_mc_samples = n_mc_samples
		self.n_minibatches = x.itervalues().next().shape[1]
	
	def subval(self, i, w):
		_x = ndict.getCols(self.x, i, i+1)
		return self.model.logpxmc(w, _x, self.n_mc_samples)
		
	def subgrad(self, i, w):
		_x = ndict.getCols(self.x, i, i+1)
		logpx, gw = self.model.dlogpxmc_dw(w, _x, self.n_mc_samples)
		return logpx, gw
		
	def val(self, w):
		logpx = [self.subval(i, w) for i in range(self.n_minibatches)]
		return np.hstack(logpx)
	
	def grad(self, w):
		logpxi, gwi = tuple(zip(*[self.subgrad(i, w) for i in range(self.n_minibatches)]))
		return np.hstack(logpxi), ndict.sum(gwi)

# FuncPosterior	
class FuncPosteriorMC():
	def __init__(self, likelihood, model):
		self.ll = likelihood
		self.model = model
		self.n_minibatches = likelihood.n_minibatches
	
	def subval(self, i, w):
		prior = self.model.logpw(w)
		prior_weight = 1. / float(self.ll.n_minibatches)
		logpx = self.ll.subval(i, w)
		return logpx.sum(), logpx.sum() + prior_weight * prior
	
	def subgrad(self, i, w):
		logpx, gw = self.ll.subgrad(i, w)
		v_prior, gw_prior = self.model.dlogpw_dw(w)
		prior_weight = 1. / float(self.ll.n_minibatches)
		v = logpx.sum() + prior_weight * v_prior
		for j in gw: gw[j] += prior_weight * gw_prior[j]
		return logpx.sum(), v, gw
	
	def val(self, w):
		logpx = self.ll.val(w)
		v = logpx.sum() + self.model.logpw(w)
		return logpx.sum(), v
	
	def grad(self, w):
		logpx, gw = self.ll.grad(w)
		v_prior, gw_prior = self.model.dlogpw_dw(w)
		v = logpx.sum() + v_prior
		for i in gw: gw[i] += gw_prior[i]
		return logpx.sum(), v, gw
