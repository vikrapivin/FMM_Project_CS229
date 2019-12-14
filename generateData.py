import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.stats as st
import os
import pdb


seedNum = 20191213
np.random.seed(seed=seedNum)

cwd = os.getcwd()
dirSave = os.path.join(cwd,"gen_seed_" + str(seedNum) + "")
if not os.path.exists(dirSave):
    os.mkdir(dirSave)

print("Saving Generated Data in " + dirSave)

# edit below to give dimensions of the data, the number of Gaussians to generate from, and the total number of data points
#controls the dimension of the data
dimData = 2
#controls the number of Gaussians to generate from
numberOfGaussians = 5
#controls the total number of data points
numberOfDataPts = 1000*dimData*numberOfGaussians
#controls the variance of the means from np.zeros, and the means are normally distributed
varMeans = 3
#below is multiplied by the variance of the means in order to then use the wichart distribution to generate sigmas
#for each Gaussian
varSigmas = 0.01
#wichart distribution degree of freedom
wichartDofMultiple = 8

# minimum points per gaussian
minDataPtsBin = np.floor(0.5/numberOfGaussians*numberOfDataPts)

#array to store minimum pts
dataPtsPerBin = np.zeros(numberOfGaussians)

# how many points should be distributed if make sure each gaussian has minimum above
dataPtsToDistribute = numberOfDataPts-numberOfGaussians*minDataPtsBin
# how much deviation in each 
meanPtsToGive = dataPtsToDistribute/(numberOfGaussians)
devPtsToGive = np.sqrt(meanPtsToGive)*8

# distribute points in an approximate Poisson type sort of way
for ii in range(0, numberOfGaussians-1):
    dataPtsPerBin[ii] = np.floor(np.random.normal(meanPtsToGive, devPtsToGive, 1)) + minDataPtsBin
# last bin
dataPtsPerBin[numberOfGaussians-1] = numberOfDataPts - np.sum(dataPtsPerBin)
print("Number of Data Points in Each Gaussian: " + np.array2string(dataPtsPerBin, formatter={'float_kind':lambda x: "%.2f" % x}))


# generate the means from Normal distribution with identity variance times varMeans
means = np.random.multivariate_normal(np.zeros(dimData), varMeans*np.eye(dimData), numberOfGaussians)
print("The means for each Gaussian are: \n" + np.array2string(means, formatter={'float_kind':lambda x: "%.2f" % x}))


# generate the sigmas from Whichart distribution with identity variance times varMeans times varSigmas
sigmas = st.wishart.rvs(dimData*wichartDofMultiple, scale=varMeans*varSigmas*np.eye(dimData), size=numberOfGaussians)
print("The sigmas for each Gaussian are: \n" + np.array2string(sigmas, formatter={'float_kind':lambda x: "%.2f" % x}))

# generate the data pts
genPts = dict()
for ii in range(0, numberOfGaussians):
    genPts[ii] = np.random.multivariate_normal(means[ii], sigmas[ii,:,:], dataPtsPerBin[ii].astype(int))

# generate unlabeled list
totalDist = np.zeros((2, dimData))
for ii in range(0, numberOfGaussians):
    totalDist = np.concatenate((totalDist,genPts[ii]))
totalDist = totalDist[2:,:]

#plot if in 2D

if dimData == 2:
    xmin = -4
    xmax = 4
    ymin = -4
    ymax = 4
    plt.figure()
    plt.scatter(totalDist[:,0],totalDist[:,1])
    axes = plt.gca()
    axes.set_xlim([xmin,xmax])
    axes.set_ylim([ymin,ymax])
    plt.savefig(os.path.join(dirSave,'gen_full_unlabeled.pdf'))
    plt.savefig(os.path.join(dirSave,'gen_full_unlabeled.jpg'))

    plt.figure()
    for ii in range(0, numberOfGaussians):
        curGen = genPts[ii]
        plt.scatter(curGen[:,0],curGen[:,1])
    axes = plt.gca()
    axes.set_xlim([xmin,xmax])
    axes.set_ylim([ymin,ymax])
    plt.savefig(os.path.join(dirSave,'gen_full_labeled.pdf'))
    plt.savefig(os.path.join(dirSave,'gen_full_unlabeled.jpg'))

#**************************************
#*** Plotting generated + PDF Probs ***
#**************************************
     
phis_temp = np.zeros(numberOfGaussians)
x = y = np.linspace(-4,4,500)
xx,yy = np.meshgrid(x,y)
dummy_data = np.array([xx.ravel(),yy.ravel()]).T
    
prob_pdf = np.zeros([numberOfGaussians,xx.ravel().shape[0]])
for jj in range(0,numberOfGaussians):
    phis_temp[jj] =  dataPtsPerBin[jj]/dataPtsPerBin.sum()

plt.figure()
for jj in range(0, numberOfGaussians):
    prob_pdf[jj] = st.multivariate_normal.pdf(dummy_data, mean=means[jj], cov=sigmas[jj])*phis_temp[jj]
    cs = plt.contour(xx,yy,prob_pdf[jj].reshape(xx.shape))
# cb = plt.colorbar(cs,shrink=0.8,extend='both')
plt.scatter(totalDist[:,0],totalDist[:,1],marker='.')
plt.title('P(x) per Gaussian at ground truth')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('tight')
plt.savefig(os.path.join(dirSave,'generated_probability_per_gaussian.pdf'))
plt.savefig(os.path.join(dirSave,'generated_probability_per_gaussian.jpg'))
        
plt.figure()
plt.scatter(totalDist[:,0],totalDist[:,1],marker='.',alpha=0.4)
prob_pdf_total = np.log(np.sum(prob_pdf, axis=0))
cs = plt.contour(xx,yy,prob_pdf_total.reshape(xx.shape),levels = np.linspace(-10,0,20))
cb = plt.colorbar(cs)
plt.title('Total Probability at ground truth')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('tight')
plt.savefig(os.path.join(dirSave,'generated_total_log_probability.pdf'))
plt.savefig(os.path.join(dirSave,'generated_total_log_probability.jpg'))
 
#**************************************
#**************************************


np.save(os.path.join(dirSave,'gen.npy'), totalDist)
np.savetxt(os.path.join(dirSave,"gen.csv"), totalDist, delimiter=",")

np.savez(os.path.join(dirSave,'gen_params.npz'), seedNum=seedNum, dimData=dimData, means=means, sigmas=sigmas, numberOfGaussians=numberOfGaussians, numberOfDataPts=numberOfDataPts, varMeans=varMeans, varSigmas=varSigmas, wichartDofMultiple=wichartDofMultiple, dataPtsPerBin=dataPtsPerBin)


for ii in range(0, numberOfGaussians):
    np.save(os.path.join(dirSave,'gen'+ str(ii) +'.npy'), np.random.multivariate_normal(means[ii], sigmas[ii,:,:], dataPtsPerBin[ii].astype(int)))