import numpy as np
import anglepy.ndict as ndict

# Pre-processing routines

def PCA(x_in, cutoff=0.99, global_sd=True, toFloat=True):
    if toFloat: x_in = x_in / 256.
    x_center = x_in.mean(axis=1, keepdims=True)
    x = x_in - x_center
    if not global_sd:
        x_sd = x.std(axis=1, keepdims=True)
    else:
        x_sd = x.std()
    x /= x_sd
    x_cov = x.dot(x.T) / x.shape[1]
    print "Performing eigen-decomposition for PCA..."
    eigval, eigvec = np.linalg.eig(x_cov)
    print "Done."
    if cutoff <= 1:
        n_used = ((eigval.cumsum() / eigval.sum()) < cutoff).sum()
        print 'PCA cutoff:', cutoff, 'n_used:', n_used
    else:
        n_used = cutoff
    eigval = eigval[:n_used].reshape((-1,1))
    eigvec = eigvec[:,:n_used]
    f_enc, f_dec = PCA_encdec(eigvec, eigval, x_center, x_sd, toFloat)
    pca_params = {'eigval':eigval, 'eigvec':eigvec, 'x_center':x_center, 'x_sd':x_sd}
    
    return f_enc, f_dec, pca_params

def PCA_fromfile(fname, toFloat=False):
    pca = ndict.loadz(fname)
    return PCA_encdec(pca['eigvec'],pca['eigval'],pca['x_center'],pca['x_sd'], toFloat)
        
def PCA_encdec(eigvec, eigval, x_center, x_sd, toFloat=False):
    def f_enc(x, n_batch=1000):
        result = np.zeros((eigvec.shape[1], x.shape[1]))
        for i in range(0, x.shape[1], n_batch):
            _x = x[:,i:(i+n_batch)]
            if toFloat: _x = _x / 256.
            result[:,i:(i+n_batch)] = eigvec.T.dot((_x - x_center) / x_sd) / np.sqrt(eigval)
        return result 
    def f_dec(x, bounded01=True):
        result = eigvec.dot(x * np.sqrt(eigval)) * x_sd + x_center
        if bounded01: result = np.maximum(0, np.minimum(1, result))
        return result
    return f_enc, f_dec
    
def normalize_random(x, global_sd=True, toFloat=False):
    if toFloat: x = x / 256.
    x_center = x.mean(axis=1, keepdims=True)
    x = x - x_center
    if not global_sd:
        x_sd = x.std(axis=1, keepdims=True)
    else:
        x_sd = x.std()
    x /= x_sd
    u, s, v = np.linalg.svd(np.random.randn(x.shape[0], x.shape[0]))
    orth = np.dot(u, v)
    def f_enc(x):
        if toFloat: x = x / 256.
        result = orth.T.dot((x - x_center) / x_sd)
        return result
    def f_dec(x, bounded01=True):
        result = orth.dot(x) * x_sd + x_center
        if bounded01: result = np.maximum(0, np.minimum(1, result))
        return result
    return f_enc, f_dec, {'orth':orth, 'center':x_center, 'sd':x_sd}

def normalize(x, global_sd=True):
    x_center = x.mean(axis=1, keepdims=True)
    x = x - x_center
    if not global_sd:
        x_sd = x.std(axis=1, keepdims=True)
    else:
        x_sd = x.std()
    x /= x_sd
    def f_enc(x): return (x - x_center) / x_sd
    def f_dec(x): return x * x_sd + x_center
    return f_enc, f_dec, (x_center, x_sd)

def normalize_noise(x, noise_sd=0.01, global_sd=True, toFloat=False):
    if toFloat: x = x / 256.
    x_center = x.mean(axis=1, keepdims=True)
    x = x - x_center
    if not global_sd:
        x_sd = x.std(axis=1, keepdims=True)
    else:
        x_sd = x.std()
    x /= x_sd
    def f_enc(x):
        if toFloat: x = x / 256.
        result = (x - x_center) / x_sd
        return np.random.normal(loc=result, scale=noise_sd, size=result.shape)
    def f_dec(x, bounded01=True):
        result = x * x_sd + x_center
        if bounded01: result = np.maximum(0, np.minimum(1, result))
        return result
    return f_enc, f_dec, (x_center, x_sd)


def preprocess_normalize01(x, global_sd=True):
    x_center = x.mean(axis=1, keepdims=True)
    x = x - x_center
    if not global_sd:
        x_sd = x.std(axis=1, keepdims=True)
    else:
        x_sd = x.std()
    x /= x_sd
    def f_enc(x): return (x - x_center) / x_sd
    def f_dec(x): return np.maximum(np.minimum(x * x_sd + x_center, 1), 0)
    return f_enc, f_dec, (x_center, x_sd)


