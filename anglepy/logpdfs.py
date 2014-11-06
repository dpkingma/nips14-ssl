import numpy as np
import theano
import theano.tensor as T
import math
import misc

# library with theano PDF functions

c = - 0.5 * math.log(2*math.pi)

def normal(x, mean, sd):
	return c - T.log(T.abs_(sd)) - (x - mean)**2 / (2 * sd**2)

def normal2(x, mean, logvar):
	return c - logvar/2 - (x - mean)**2 / (2 * T.exp(logvar))

def laplace(x, mean, logvar):
    sd = T.exp(0.5 * logvar)
    return - abs(x - mean) / sd - 0.5 * logvar - np.log(2)
    
def standard_normal(x):
	return c - x**2 / 2

# Centered laplace with unit scale (b=1)
def standard_laplace(x):
	return math.log(0.5) - T.abs_(x)

# Centered student-t distribution
# v>0 is degrees of freedom
# See: http://en.wikipedia.org/wiki/Student's_t-distribution
def studentt(x, v):
	gamma1 = misc.log_gamma_lanczos((v+1)/2.)
	gamma2 = misc.log_gamma_lanczos(0.5*v)
	
	return gamma1 - 0.5 * T.log(v * np.pi) - gamma2 - (v+1)/2. * T.log(1 + (x*x)/v)
