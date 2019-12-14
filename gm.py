# modified from gmm.py from CS229 class at Stanford
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal as mnorm
from scipy.stats import multivariate_normal
from scipy.stats import invwishart
from numpy.random import multinomial as mult
import numpy as np
from utils import *
import os

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)
eps = 1e-6

class gm:
    """Mixture of Gaussians model.

    Example usage:
        > gm = Kuramoto(step_size=lr)
        > x = gm.sample(N=10)
        > z_pred = gm.predict(x)
    """
    def __init__(self, pars=None, K=4, d=2, n=None, c=1.0):
        """
        Args:
            pars: dictionary with entries theta, mu, Sigma. mu and Sigma are lists of length K, where K is the number of mixture
            components. Theta is K numpy array. K is default number of clusters. If pars is None random parameters are created
            for K clusters. d is the dimension per observation. Only used when pars is missing. Otherwise inferred. c is the
            variance used when sampling the means.
        """
        if pars is not None:
            self.mu = pars['mu']
            self.d = self.mu[0].shape[0]
            self.Sigma = pars['Sigma']
            self.K = len(self.mu)
            self.theta = pars['theta']
            self.theta[K-1] = 0.0 # enforce normalization
            #self.theta = np.concatenate([pars['theta'], np.array([0.0])])  # So theta is already of size K
            self.phi = phiFromTheta(self.theta)
        else:
            self.K = K
            self.d = d
            mu = []
            Sigma = []
            phi = np.random.dirichlet(alpha=np.repeat(1,K)) # need to convert to theta
            theta = thetaFromPhi(phi)
            mus = mnorm(np.zeros(d), c*np.eye(d), K)
            for k in np.arange(K):
                mu.append(mus[k,:].reshape(d))
                Sigma.append(invwishart.rvs(df=d+2, scale=np.eye(d)))
            self.mu = mu
            self.Sigma = Sigma
            self.theta = theta
            self.phi = phi
        self.n = n
        if n is None:
            self.n = np.zeros(self.K) # number of (expected) processed observations from each cluster. Used for learning.

    def sample(self, N=100, shuffle = True):
        """
        Generate N samples from the model. Returns three numpy arrays: z2 of size (N,) with cluster assignments,
        x = (N,d) with the observations, and z with the cluster counts (used for debugging only).
        """
        z = mult(N, self.phi, size=1).reshape(self.K) # these are the cluster assignments
        x = np.zeros((N, self.d))
        z2 = np.zeros(N, dtype=int) # To rexpress z as vector with cluster assignments
        ct = 0
        for k in np.arange(self.K):
            x[ct:ct+z[k],:] = mnorm(self.mu[k], self.Sigma[k], z[k])
            z2[ct:ct+z[k]] = k
            ct += z[k]
        if shuffle:
            idx = np.random.permutation(np.arange(N))
            z2 = z2[idx]
            x = x[idx,:]
        return(z2,x,z)

    def predict(self, x, clusterAssignments=True):
        """
        Takes in observations of size (N,d) and returns array of size (N,) with most likely cluster assignment according
        to current model parameters. If clusterAssignments is True, also returns the cluster assignments in (N,) array
        """
        N = x.shape[0]
        phi, mu, Sigma, theta, K, d = unpackModelParameters(self)
        logp = np.zeros((N, K)) # likelihood per cluster
        w = np.zeros((N, K)) # i,j entry is P(z_i=j|x_i).
        for k in np.arange(K):
            logp[:,k] = multivariate_normal.logpdf(x, mean=mu[k], cov=Sigma[k])
            w[:,k] = np.log(phi[k]) + logp[:,k]
        w = np.exp(w)
        w = np.diag(1/w.sum(axis=1))@w
        z_pred = np.zeros(N, dtype=int)
        if clusterAssignments:
            z_pred = np.argmax(w, axis=1)
        return(w,z_pred)

    def MLE(self, z, x):
        """
        Use fully observed dataset (z, x) to estimate parameters via Maximum Likelihood Estimation.
        """
        phi, mu, Sigma, theta, K, d = unpackModelParameters(self)
        n = np.zeros(K, dtype=int)
        mu = []
        Sigma = []
        for k in np.arange(K):
            I = (z == k)
            n_k = I.sum() + 1.0
            mu_k = (np.zeros(d)+x[I,:].sum(axis=0))/n_k
            mu.append(mu_k)
            mu_k = mu_k.reshape(1,d)
            Sigma_k = (eps*np.eye(d) + np.matmul((x[I,:]-mu_k).transpose(),(x[I,:]-mu_k)))/n_k
            n[k] = n_k
            Sigma.append(Sigma_k + eps*np.eye(d))
        self.phi = n/n.sum()
        self.mu = mu
        self.Sigma = Sigma
        self.theta = thetaFromPhi(self.phi)
        self.n = n # Reset observation counts.

    def SGDish(self, weight_fun, theta_fun, objective_fun, x, tol=-0.05, k_check=1000, alpha=1.0, max_itr=None):
        """ Implement SGD-like algo using weight_fun and theta_fun to compute part of the gradients. 
        Updates model parameters in place. x is the training data, a (N, d) numpy array. Algorithm stops when 
        the likelihood of the data changes by less than tol in the last k_check datapoints. Alpha is weight given
        to initial condition.
        """
        N = x.shape[0]
        if max_itr is None:
            max_itr = N*2.0 # Just one pass through data
        if N < k_check:
            k_check = N # Check convergence at most every pass through the data
        phi_old, mu_old, Sigma_old, theta_old, K, d = unpackModelParameters(self)
        n = self.n*alpha
        delta = 1.0
        ct1 = 0 # counter used to iterate through the data
        ct2 = 1
        itr = 0 # to stop in case of no convergence
        obj_cur = objective_fun(x, phi_old, mu_old, Sigma_old)
        obj_old = obj_cur - 1.0 # assume maximization context
        phi_new = n # current effective cluster counts
        n_new = n + 0.0
        print('starting objective: ' + str(np.round(obj_cur,6)))
        while (delta > tol and itr < max_itr):
            mu_new = []
            Sigma_new = []
            # Process new datapoint
            x_i = x[ct1,:].reshape(1,d)
            w = weight_fun(x_i, phi_old, mu_old, Sigma_old).reshape(K)
            n_new = n + w # effective ct for current observation
            phi_new = n_new/n_new.sum()
            for k in np.arange(K):
                mu_k_new = (n[k]*mu_old[k] + w[k]*x[ct1,:])/n_new[k]
                mu_k = mu_old[k].reshape(1,d)
                Sigma_k = n[k]*(Sigma_old[k] + np.matmul(mu_k.transpose(), mu_k))
                Sigma_k += w[k]*np.matmul(x_i.transpose(), x_i)
                Sigma_k = Sigma_k/n_new[k] - np.matmul(mu_k_new.reshape(1,d).transpose(), mu_k_new.reshape(1,d))
                Sigma_new.append(Sigma_k + eps*np.eye(d))
                mu_new.append(mu_k_new)
            if np.mod(ct2, k_check) == 0:
                # compute objective
                obj_old = obj_cur + 0.0
                obj_cur = objective_fun(x, phi_new, mu_new, Sigma_new)
                delta = obj_cur - obj_old
                print('objective: ' + str(np.round(obj_cur,6)) + ' after ' + str(itr+1) + ' datapoints.')
            ct1 += 1
            ct2 += 1
            itr += 1
            if ct1 > N - 1:
                ct1 = 0 # start going through data again
            phi_old = phi_new + 0.0
            mu_old = mu_new
            Sigma_old = Sigma_new
            n = n_new + 0.0
        # update parameters
        print('Stopped after ' + str(itr) + ' data points processed. Delta is ' + str(delta))
        self.mu = mu_new
        self.Sigma = Sigma_new
        self.phi = phi_new
        self.n = n_new
        self.theta = thetaFromPhi(phi_new)

def computeAccuracies(Ntrain=5000, N0=50, Ntest=1000, K=5, d=2, useDefaultPars=False, ns=20, cs=np.linspace(0.1, 2.0, 11)):
    # First explore default parameters as distance of mean from origina changes. ns is number of simulations per
    # value of parameters.
    nCs = cs.shape[0] # number of c values
    nModels = 6 # includes ground truth, and models right after initial estimate
    accuracies = np.zeros((nModels, nCs, ns)) # First model is ground truth, second is after EM initialization
    for i in np.arange(nCs):
        for s in np.arange(ns):
            if useDefaultPars:
                pars = gm_pars1(cs[i])
                gm_model = gm(pars) # ground truth: used to generate data
            else:
                gm_model = gm(K=K, d=d, c=cs[i]) # is the variance of random means
            K = gm_model.K
            d = gm_model.d
            gm_model2 = gm(K=K, d=d) # learner gm: will do ELBO
            # Generate data
            z0, x0, z_cts0 = gm_model.sample(N0) # used to initialize estimates. Assume fully observed   
            z_train, x_train, z_cts_train = gm_model.sample(Ntrain)
            z_test, x_test, z_cts_test = gm_model.sample(Ntest)
            # Initialize learner parameters
            gm_model2.MLE(z0,x0) # Max Likelihood estimation
            pars2 = packModelParameters(gm_model2)
            gm_model3 = gm(pars2) # learner gm2: will do exp(Delta)
            gm_model4 = gm(pars2) # learner gm3: will do exp(3.0*Delta)
            gm_model5 = gm(pars2) # learner gm4: will do exp(3.0*Delta + 0.2*sigma_l)
            # Predict after initialization
            z_pred0 = gm_model2.predict(x_test)[1]
            # Learn from training data
            print('Starting ELBO-based SGDish')
            gm_model2.SGDish(elboWeight, elboGradTheta, elbo, x_train)
            print('Starting lTilde-based SGDish algos')
            gm_model3.SGDish(lTildeWeight4, elboGradTheta, logL, x_train) # w as prescribed
            gm_model4.SGDish(lTildeWeight, elboGradTheta, logL, x_train) # w=phi*e^Delta
            gm_model5.SGDish(lTildeWeight3, elboGradTheta, logL, x_train)      #w=phi*e^{stuff}
            z_pred2 = gm_model2.predict(x_test)[1]
            z_pred3 = gm_model3.predict(x_test)[1] # After LTilde learning
            z_pred4 = gm_model4.predict(x_test)[1]
            z_pred5 = gm_model5.predict(x_test)[1]
            z_pred1 = gm_model.predict(x_test)[1]  # optimal prediction, by ground truth
            # Compute accuracy
            acc0 = accuracy(z_test, z_pred0) # right after parameter initialization
            acc1 = accuracy(z_test, z_pred1) # ground truth model
            acc2 = accuracy(z_test, z_pred2) 
            acc3 = accuracy(z_test, z_pred3) 
            acc4 = accuracy(z_test, z_pred4) 
            acc5 = accuracy(z_test, z_pred5)
            accuracies[0, i, s] = acc0
            accuracies[1, i, s] = acc1
            accuracies[2, i, s] = acc2
            accuracies[3, i, s] = acc3
            accuracies[4, i, s] = acc4
            accuracies[5, i, s] = acc5
        print("Value " + str(i) + " of c finished.")
    return(accuracies)

def plotAccuracies(cs, avg_acc, savePath='plot1.jpg'):
    plt.rcParams['legend.title_fontsize'] = 'x-large'
    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize=(14,10))
    plt.plot(cs, avg_acc[0,:], linewidth=4, label='initialization')
    plt.plot(cs, avg_acc[1,:], linewidth=4, label='ground truth model')
    plt.plot(cs, avg_acc[2,:], linewidth=4, label='true objective or ELBO')
    plt.plot(cs, avg_acc[3,:], linewidth=4, label='$l_0+l_2$')
    plt.plot(cs, avg_acc[4,:], linewidth=4, label='$w=\phi e^{\Delta}$')
    plt.plot(cs, avg_acc[5,:], linewidth=4, label='other variant')
    plt.legend(title='', fontsize = 'large', loc='upper left')
    plt.title('classification accuracies')
    #plt.title('')
    plt.xlabel('c')
    plt.ylabel('classification accuracy')
    #plt.xlim([0, tau])
    plt.savefig(savePath, dpi=300)
    plt.show()
    
def main():
    #print("Evaluating fixed model with K=5, d=2")
    #cs = np.linspace(0.5, 3.0, 11) # distance of mean of 4 clusters to origin
    #accuracies = computeAccuracies(Ntrain=4000, N0=50, Ntest=1000, ns=6, useDefaultPars=True, cs=cs)
    #avg_acc = np.mean(accuracies, axis=2)
    #plotAccuracies(cs, avg_acc, savePath='AccDefaultPars1DefaultAlgo.jpg')
    
    print("Evaluating sampled models now")
    cs2 = np.linspace(0.01, 1.0, 11) # distance of mean of 4 clusters to origin
    accuracies2 = computeAccuracies(Ntrain=10000, N0=200, Ntest=1000, ns=2, K=15, d=2, cs=cs2)
    avg_acc2 = np.mean(accuracies2, axis=2)
    plotAccuracies(cs2, avg_acc2, savePath='AccSampledModelsK15d5.jpg')
    
    avg_accuracies = []
    #avg_accuracies.append(avg_acc)
    avg_accuracies.append(avg_acc2)
    
    cs_used  = []
    #cs_used.append(cs)
    cs_used.append(cs2)
    
    return(cs_used, avg_accuracies)

if __name__ == '__main__':
    main()
    