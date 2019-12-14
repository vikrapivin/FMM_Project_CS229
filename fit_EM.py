import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.stats as st
import os


seedNum = 20191201
# uncomment if you want to set this as also the seed
# np.random.seed(seed=seedNum)

cwd = os.getcwd()
dirSave = os.path.join(cwd,"gen_seed_" + str(seedNum))

data = np.load(os.path.join(dirSave,'gen.npy'))
dataPts = data.shape[0]
dimOfData = data.shape[1]

dataShuffle = np.arange(dataPts)
np.random.shuffle(dataShuffle)

numberOfPeaks = 5 # corresponds to number of Gaussians
import pdb

# helping parameter initialization
labeled_data = np.zeros([numberOfPeaks,10*(dimOfData-1),dimOfData])
mean_guess = np.zeros([numberOfPeaks,dimOfData]) # number of gaussians x dim
sigma_guess = np.zeros([numberOfPeaks, dimOfData, dimOfData]) # number of gaussians x dim x dim
data_outer = np.zeros([numberOfPeaks,10*(dimOfData-1),dimOfData,dimOfData])
    
for ii in range(0, numberOfPeaks):
    labeled_data = np.load(os.path.join(dirSave,'gen'+ str(ii) +'.npy'))[0:10*(dimOfData-1)]

    data_outer[ii] = np.einsum('ij,ik->ijk',labeled_data,labeled_data)
    mean_guess[ii] = np.mean(labeled_data,axis=0)

    sigma_guess[ii,:,:] = np.mean(data_outer[ii] - np.outer(mean_guess[ii],mean_guess[ii]),axis=0)

means = mean_guess
sigmas = sigma_guess

# pdb.set_trace()

thetas = np.random.uniform(low=-1.0,high=1.0, size=numberOfPeaks-1) # uniform distribution

learningRateStart = 1

start_time = time.time()

# START SGD
for ii in range(0,1*dataPts):
    learningRate = learningRateStart/(1+ii/500)
    phis = np.zeros(numberOfPeaks-1)
    for jj in range(0,numberOfPeaks-1):
        phis[jj] = 1/( np.sum( np.exp(thetas-thetas[jj]) ) + np.exp( - thetas[jj] ) )      
    phis = np.concatenate((phis, [(1-np.sum(phis))])) # add phi_k

    prob_norm_term = np.zeros((numberOfPeaks)) # normalization parameter of p_j(x)
    curDataPt = np.copy(data[dataShuffle[ii%dataPts]])
    x_m_mean = np.copy(curDataPt[np.newaxis,:]-means) # x - mu_j
    prob_log_term = np.zeros((numberOfPeaks)) # argument of multivariate exponential

    for jj in range(0,numberOfPeaks):
        # prob_norm_term[jj] = np.log(1/((2*np.pi)**(dimOfData/2)*np.linalg.det(sigmas[jj])**(1/2)))
        # prob_log_term[jj] = ( -1/2 * np.sum(np.matmul(x_m_mean[jj],np.linalg.inv(sigmas[jj]) )* x_m_mean[jj] ) )
        prob_log_term[jj] = st.multivariate_normal.logpdf(curDataPt, mean=means[jj], cov=sigmas[jj])
    
    alpha = np.zeros(numberOfPeaks) # p_j(x)/p(x)
    for jj in range(0,numberOfPeaks):
        # alpha[jj] = 1/(np.sum(np.exp(prob_log_term + prob_norm_term - prob_log_term[jj] - prob_norm_term[jj] )*phis))
        alpha[jj] = 1/(np.sum(np.exp(prob_log_term - prob_log_term[jj] )*phis))
        
    # theta update rule (EM)
    thetas = thetas + learningRate*phis[0:-1]*(alpha[0:-1]-1)
    outprod_x = np.outer(curDataPt, curDataPt)
    
    for jj in range(0,numberOfPeaks):
        outprod_mu = np.outer(mean_guess[jj], mean_guess[jj])        
        mat_B = 2*learningRate*phis[jj]*alpha[jj] * (outprod_x - outprod_mu - sigmas[jj]) # making code more readable
        
        sigmas[jj] = sigmas[jj] + np.matmul(sigmas[jj], np.matmul(mat_B,sigmas[jj]))
        means[jj] = means[jj] + learningRate*phis[jj]*alpha[jj]*np.matmul(sigmas[jj],(curDataPt - means[jj]))
       
    if ii % 1000 == 0:  
        print(phis)
        print(means) 
        print(sigmas)  
        print(ii)
    

phiNumer = np.exp(thetas)
phiDeno = np.sum(phiNumer) + 1
phis = phiNumer/phiDeno
phis = np.concatenate((phis, [(1-np.sum(phis))]))
propDt = phis*dataPts
print(propDt)
print(thetas)
print(sigmas)
print(means)


if dimOfData == 2:
    numberOfGaussians = numberOfPeaks
    genPts = dict()
    for ii in range(0, numberOfGaussians):
        genPts[ii] = np.random.multivariate_normal(means[ii], sigmas[ii,:,:], np.floor(propDt[ii]).astype(int))
    xmin = -4
    xmax = 4
    ymin = -4
    ymax = 4
    plt.figure()
    plt.scatter(data[:,0],data[:,1])
    for ii in range(0, numberOfGaussians):
        curGen = genPts[ii]
        plt.scatter(curGen[:,0],curGen[:,1])
    axes = plt.gca()
    axes.set_xlim([xmin,xmax])
    axes.set_ylim([ymin,ymax])
    # plt.show()
    plt.savefig(os.path.join(dirSave,'fit_labeled_data.pdf'))
    plt.figure()
    for ii in range(0, numberOfGaussians):
        curGen = genPts[ii]
        plt.scatter(curGen[:,0],curGen[:,1])
    axes = plt.gca()
    axes.set_xlim([xmin,xmax])
    axes.set_ylim([ymin,ymax])
    plt.savefig(os.path.join(dirSave,'fit_labeled.pdf'))


end_time = time.time()
conv_time = end_time - start_time 
np.savetxt(os.path.join(dirSave,'convergence_time.txt'), np.array([[conv_time]]))
