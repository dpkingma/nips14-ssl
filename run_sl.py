import numpy as np
import anglepy
import anglepy.ndict as ndict
from anglepy.models import GPUVAE_YZ_X
import sys

# Load MNIST data
dataset = sys.argv[1]
if dataset == 'mnist':
    import anglepy.data.mnist as mnist
    _, train_y, _, _, test_x, test_y = mnist.load_numpy(size=28, binarize_y=False)

    # Compute prior probabilities per class
    train_y = mnist.binarize_labels(train_y)
    prior_y = train_y.mean(axis=1).reshape((10,1))

    # Create model
    n_x = 28*28
    n_y = 10
    n_z = 50
    n_hidden = 500,500
    updates = None
    model = GPUVAE_YZ_X(updates, n_x, n_y, n_hidden, n_z, n_hidden, 'softplus', 'softplus', type_px='bernoulli', type_qz='gaussianmarg', type_pz='gaussianmarg', prior_sd=1, uniform_y=True)

    # Load parameters
    dir = 'models/mnist_yz_x_50-500-500/'
    ndict.set_value(model.v, ndict.loadz(dir+'v_best.ndict.tar.gz'))
    ndict.set_value(model.w, ndict.loadz(dir+'w_best.ndict.tar.gz'))

else:
    raise Exception("Unknown dataset")

# Make predictions on test set
def get_lowerbound():
    lb = np.zeros((n_y,test_x.shape[1]))
    for _class in range(n_y):
        y = np.zeros((n_y,test_x.shape[1]))
        y[_class,:] = 1
        _lb = model.eval({'x': test_x.astype(np.float32), 'y':y.astype(np.float32)}, {})
        lb[_class,:] = _lb
    return lb

def get_predictions(n_samples=1000, show_convergence=True):
    px = 0
    def get_posterior(likelihood, prior):
        posterior = (likelihood * prior)
        posterior /= posterior.sum(axis=0, keepdims=True)
        return posterior
    for i in range(n_samples):
        px += np.exp(get_lowerbound())
        if show_convergence:
            posterior = get_posterior(px / (i+1), prior_y)
            pred = np.argmax(posterior, axis=0)
            error_perc = 100* (pred != test_y).sum() / (1.*test_y.shape[0])
            print 'samples:', i, ', test-set error (%):', error_perc
    posterior = get_posterior(px / n_samples, prior_y)
    return np.argmax(posterior, axis=0)

n_samples = 1000
print 'Computing class posteriors using a marginal likelihood estimate with importance sampling using ', n_samples, ' samples.'
print 'This is slow, but could be sped up significantly by fitting a classifier to match the posteriors (of the generative model) in the training set.'
print 'For MNIST, this should converge to ~ 0.96 % error.'
result = get_predictions(n_samples)
print 'Done.'
print 'Result (test-set error %): ', result

'''
# Compare predictions with truth
print 'Predicting with 1, 10, 100 and 1000 samples'
for n_samples in [1,10,100,1000]:
    print 'Computing predictions with n_samples = ', n_samples
    predictions = get_predictions(n_samples)
    error_perc = 100* (predictions != test_y).sum() / (1.*test_y.shape[0])
    print 'Error rate is ', error_perc, '%'
'''
