import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
eps = 1e-6 # numerical tolerance

def thetaFromPhi(phi):
    # Input phi is (K,) array of cluster probabilities. Computes (K,) thetas vector.
    K = phi.shape[0]
    return np.log(phi/phi[K-1])

def softmax(x):
    """
    Compute softmax function for a batch of input values. 
    The first dimension of the input corresponds to the batch size. The second dimension
    corresponds to every class in the output.
    Args:
        x: A 2d numpy float array of shape batch_size x number_of_classes (K)

    Returns:
        A 2d numpy float array containing the softmax results of shape batch_size x number_of_classes
    """
    # *** START CODE HERE ***
    # Use log_sum_exp trick for numerical stability
    # x will be the output, before activation, of the second layer, so will have 10 entries per observation
    #x = np.random.randn(100, 10)
    b, k = x.shape # batch size, and number of categories
    a = np.amax(x, axis = 1) # shape is b
    s = np.exp(x - a.reshape(b,1))
    c = s.sum(axis=1)
    s2 = np.matmul(np.diag(1.0/c), s)
    return s2

def phiFromTheta(theta):
    # Inverse of thetaFromPhi.
    K = theta.shape[0]
    phi = softmax(theta.reshape(1,K)).reshape(K)
    return phi

def gm_pars1(c=4.0):
    # Parameters to initialize a 2d GM with 5 clusters with pretty symmetry. c determines distance of 4 means relative to origin.
    # Components have std dev of 1.
    K = 5
    d = 2
    mu = []
    Sigma = []
    corrs = np.array([-0.5, 0.5, -0.5, 0.5, 0.0])
    mus = np.array([[c, c], [-c, c], [-c, -c], [c, -c], [0.0,0.0]])
    for k in np.arange(K):
        mu.append(mus[k,:])
        Sigma1 = np.eye(d)
        Sigma1[0,1] = corrs[k]
        Sigma1[1,0] = corrs[k]
        Sigma.append(Sigma1)
    pars = {}
    pars['mu'] = mu
    pars['Sigma'] = Sigma
    pars['theta'] = np.zeros(K) # equal sized clusters 
    return pars

def accuracy(z, z_pred):
    # Compute accuracy of cluster predictions
    N = z.shape[0]
    return (z_pred == z).sum()/N

def unpackModelParameters(gm_model):
    # Input is object of class gm. Returns tuple of parameters
    return(gm_model.phi, gm_model.mu, gm_model.Sigma, gm_model.theta, gm_model.K, gm_model.d)

def packModelParameters(gm_model):
    # Input is object of class gm. Returns tuple of parameters
    mu, Sigma, theta, phi = gm_model.mu, gm_model.Sigma, gm_model.theta, gm_model.phi
    pars = {}
    pars['mu'] = mu
    pars['Sigma'] = Sigma
    pars['theta'] = theta
    pars['phi'] = phi
    return pars

def elboWeight(x, phi, mu, Sigma):
    # Compute the ELBO weight of (N,d) observation x. w_j(x) = phi_j*p_j(x)/p(x). 
    N = x.shape[0]
    K = phi.shape[0]
    w = np.repeat(phi.reshape(1,K), N, axis=0)
    p_j = np.zeros((N,K)) # entry i j is P(z=j|x_i)
    for k in np.arange(K):
        p_j[:,k] = np.exp(multivariate_normal.logpdf(x, mean=mu[k], cov=Sigma[k]))
    p = np.dot(p_j, phi)
    w = w*p_j/np.repeat(p.reshape(N,1), K, axis=1)
    w = np.diag(1.0/w.sum(axis=1))@w
    return w

def elboGradTheta(w, phi, mu, Sigma):
    N, K = w.shape
    gradTheta = w - np.repeat(phi.reshape(1,K), N, axis=0)
    return gradTheta
    
def elbo(x, phi, mu, Sigma):
    # Assumes x is (N, d)
    N = x.shape[0]
    K = phi.shape[0]
    w = elboWeight(x, phi, mu, Sigma)
    logp_j = np.zeros((N,K)) # entry i j is P(z=j|x_i)
    for k in np.arange(K):
        logp_j[:,k] = multivariate_normal.logpdf(x, mean=mu[k], cov=Sigma[k]) + np.log(phi[k])
    # Return ELBO/N
    return(np.sum(logp_j*w)/N)

def lTildeWeight(x, phi, mu, Sigma):
    # Compute the lTilde weight of (N,d) observation x. w_j(x) = phi_j*p_j(x)/p(x). 
    N = x.shape[0]
    K = phi.shape[0]
    w = np.repeat(phi.reshape(1,K), N, axis=0)
    d = mu[0].shape[0]
    logp_j = np.zeros((N,K)) # entry i j is P(z=j|x_i)
    for k in np.arange(K):
        logp_j[:,k] = multivariate_normal.logpdf(x, mean=mu[k], cov=Sigma[k]) + eps
    lbar = np.dot(logp_j, phi) # Size (N,)
    Delta = logp_j - np.repeat(lbar.reshape(N,1), K, axis=1)
    Delta = np.minimum(Delta, 5.0)
    sigma_l = np.minimum(np.dot(np.square(Delta), phi), 5.0)
    #w = w*(1 + 2.0*Delta/(2.0 + np.repeat(sigma_l.reshape(N,1), K, axis=1))))
    w = w*np.exp(Delta) + eps # This is a good one
    return np.diag(1.0/w.sum(axis=1))@w

def lTildeWeight4(x, phi, mu, Sigma): # This is the one in the actual derivation
    # Compute the lTilde weight of (N,d) observation x. w_j(x) = phi_j*p_j(x)/p(x). 
    N = x.shape[0]
    K = phi.shape[0]
    w = np.repeat(phi.reshape(1,K), N, axis=0)
    d = mu[0].shape[0]
    logp_j = np.zeros((N,K)) # entry i j is P(z=j|x_i)
    for k in np.arange(K):
        logp_j[:,k] = multivariate_normal.logpdf(x, mean=mu[k], cov=Sigma[k]) + eps
    lbar = np.dot(logp_j, phi) # Size (N,)
    Delta = logp_j - np.repeat(lbar.reshape(N,1), K, axis=1)
    Delta = np.minimum(Delta, 5.0)
    sigma_l = np.minimum(np.dot(np.square(Delta), phi), 5.0)
    w = np.maximum(w*(1.0 + 2.0*Delta/(2.0 + np.repeat(sigma_l.reshape(N,1), K, axis=1))), 0.0) + eps
    return np.diag(1.0/w.sum(axis=1))@w

def lTildeWeight2(x, phi, mu, Sigma):
    # Compute the lTilde weight of (N,d) observation x. w_j(x) = phi_j*p_j(x)/p(x). 
    N = x.shape[0]
    K = phi.shape[0]
    w = np.repeat(phi.reshape(1,K), N, axis=0)
    d = mu[0].shape[0]
    logp_j = np.zeros((N,K)) # entry i j is P(z=j|x_i)
    for k in np.arange(K):
        logp_j[:,k] = multivariate_normal.logpdf(x, mean=mu[k], cov=Sigma[k]) + eps
    lbar = np.dot(logp_j, phi) # Size (N,)
    Delta = logp_j - np.repeat(lbar.reshape(N,1), K, axis=1)
    Delta = np.minimum(Delta, 5.0)
    sigma_l = np.minimum(np.dot(np.square(Delta), phi), 5.0)
    w = w*np.exp(3.0*Delta) + eps # This is a good one
    return np.diag(1.0/w.sum(axis=1))@w

def lTildeWeight3(x, phi, mu, Sigma):
    # Compute the lTilde weight of (N,d) observation x. w_j(x) = phi_j*p_j(x)/p(x). 
    N = x.shape[0]
    K = phi.shape[0]
    w = np.repeat(phi.reshape(1,K), N, axis=0)
    d = mu[0].shape[0]
    logp_j = np.zeros((N,K)) # entry i j is P(z=j|x_i)
    for k in np.arange(K):
        logp_j[:,k] = multivariate_normal.logpdf(x, mean=mu[k], cov=Sigma[k]) + eps
    lbar = np.dot(logp_j, phi) # Size (N,)
    Delta = logp_j - np.repeat(lbar.reshape(N,1), K, axis=1)
    Delta = np.minimum(Delta, 5.0)
    sigma_l = np.minimum(np.dot(np.square(Delta), phi), 5.0)
    w = w*np.exp(2.0*Delta/(2.0 + np.repeat(sigma_l.reshape(N,1), K, axis=1))) + eps 
    return np.diag(1.0/w.sum(axis=1))@w

def logL(x, phi, mu, Sigma): #log marginal of the data
    # Assumes x is (N, d)
    N = x.shape[0]
    K = phi.shape[0]
    logp_j = np.zeros((N,K)) # entry i j is P(z=j|x_i)
    for k in np.arange(K):
        logp_j[:,k] = multivariate_normal.logpdf(x, mean=mu[k], cov=Sigma[k]) + np.log(phi[k])
    logL = np.log(np.dot(np.exp(logp_j), phi))
    # Return ELBO/N
    return(np.sum(logL)/N)
