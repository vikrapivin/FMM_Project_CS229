import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pdb
import scipy.stats as st


#fit everything with ability to use different methods
seedNum = 20191213
# uncomment if you want to set this as also the random seed
# np.random.seed(seed=seedNum)

def saveFits(plottype, thetas, means, sigmas, learningRate):
    saveFile = os.path.join(dirSave,'fitted_parameters_' + plottype + '.npz')
    np.savez(saveFile, thetas=thetas, means=means, sigmas=sigmas, learningRate=learningRate)
    return
    
def online_plotting_equiprob(numberOfPeaks, phis, means, sigmas, data, curFig):
    if curFig == 0 :
        curFig = 1000
    plt.figure(curFig)
    x = y = np.linspace(-4,4,20)
    xx,yy = np.meshgrid(x,y)
    dummy_data = np.array([xx.ravel(),yy.ravel()]).T
    
    prob_pdf = np.zeros([numberOfPeaks,xx.ravel().shape[0]])
    for jj in range(0, numberOfPeaks):
        prob_pdf[jj] = st.multivariate_normal.pdf(dummy_data, mean=means[jj], cov=sigmas[jj])*phis[jj]
    plt.clf()
    plt.scatter(data[:,0],data[:,1],marker='.',alpha=0.4)
    prob_pdf_total = np.log(np.sum(prob_pdf, axis=0))
    cs = plt.contour(xx,yy,prob_pdf_total.reshape(xx.shape),levels = np.linspace(-3,0,10))
    # cb = curFig.colorbar(cs)
    plt.title('Total Probability')
    plt.axis('tight')
    plt.show(block=False)
    #give time for plot to update
    plt.pause(1)
    return curFig

# **************************************
#*** plot original generated data vs (1) PDF probability per gaussian (2) total PDF probability ***
# **************************************
def plotting_equiprob(plottype, numberOfPeaks, thetas, means, sigmas, data):
    phis_temp = np.zeros(numberOfPeaks-1)
    x = y = np.linspace(-4,4,500)
    xx,yy = np.meshgrid(x,y)
    dummy_data = np.array([xx.ravel(),yy.ravel()]).T
    
    prob_pdf = np.zeros([numberOfPeaks,xx.ravel().shape[0]])
    for jj in range(0,numberOfPeaks-1):
        phis_temp[jj] = 1/( np.sum( np.exp(thetas-thetas[jj]) ) + np.exp( - thetas[jj] ) )   
    phis_temp = np.concatenate((phis_temp, [(1-np.sum(phis_temp))]))
    for jj in range(0, numberOfPeaks):
        prob_pdf[jj] = st.multivariate_normal.pdf(dummy_data, mean=means[jj], cov=sigmas[jj])*phis_temp[jj]
        # cs = plt.contour(xx,yy,prob_pdf[jj].reshape(xx.shape),levels = np.linspace(0,1,60))
        # cs = plt.contour(xx,yy,prob_pdf[jj].reshape(xx.shape),levels = np.linspace(-1000,0,500) )
        cs = plt.contour(xx,yy,prob_pdf[jj].reshape(xx.shape))
        # cb = plt.colorbar(cs, shrink=0.8, extend='both')
    plt.scatter(data[:,0],data[:,1],marker='.', alpha=0.4)
    plt.title('P(x) per Gaussian')
    plt.axis('tight')
    plt.savefig(os.path.join(dirSave,'fit_' + plottype + '_probability_per_gaussian.pdf'))
    plt.savefig(os.path.join(dirSave,'fit_' + plottype + '_probability_per_gaussian.jpg'))
    plt.close()
        
    plt.figure()
    plt.scatter(data[:,0],data[:,1],marker='.',alpha=0.4)
    prob_pdf_total = np.log(np.sum(prob_pdf, axis=0))
    cs = plt.contour(xx,yy,prob_pdf_total.reshape(xx.shape),levels = np.linspace(-10,0,20))
    # cs = plt.contour(xx,yy,prob_pdf_total.reshape(xx.shape),levels = np.linspace(-1000,0,500))
    cb = plt.colorbar(cs)
    plt.title('Total Probability for ' + plottype + ' model')
    plt.axis('tight')
    plt.savefig(os.path.join(dirSave,'fit_' + plottype + '_total_log_probability.pdf'))
    plt.savefig(os.path.join(dirSave,'fit_' + plottype + '_total_log_probability.jpg'))
    plt.close()


def plot2dimData(plottype, data, dataPts, numberOfGaussians, means, sigmas, thetas):
    phis = np.zeros(numberOfPeaks-1)
    for jj in range(0,numberOfPeaks-1):
        phis[jj] = 1/( np.sum( np.exp(thetas-thetas[jj]) ) + np.exp( - thetas[jj] ) )   
    phis = np.concatenate((phis, [(1-np.sum(phis))]))
    propDt = phis*dataPts
    genPts = dict()
    for ii in range(0, numberOfGaussians):
        genPts[ii] = np.random.multivariate_normal(means[ii], sigmas[ii], np.floor(propDt[ii]).astype(int))
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
    plt.savefig(os.path.join(dirSave,'fit_'+ plottype +'_labeled_data.pdf'))
    plt.savefig(os.path.join(dirSave,'fit_'+ plottype +'_labeled_data.jpg'))
    plt.close()
    plt.figure()
    for ii in range(0, numberOfGaussians):
        curGen = genPts[ii]
        plt.scatter(curGen[:,0],curGen[:,1])
    axes = plt.gca()
    axes.set_xlim([xmin,xmax])
    axes.set_ylim([ymin,ymax])
    plt.savefig(os.path.join(dirSave,'fit_'+ plottype +'_labeled.pdf'))
    plt.savefig(os.path.join(dirSave,'fit_'+ plottype +'_labeled.jpg'))
    plt.close()
    
    plotting_equiprob(plottype, numberOfGaussians, thetas, means, sigmas, data)
# the below procedure fits with EM and uses an approximation to the inverse matrix
# not the best results, but not the worse
def fit_with_em(learningRateStart, data, dataShuffle, dataPts, dimOfData, numberOfPeaks, thetas, means, sigmas):
    for ii in range(0,1*dataPts):
        learningRate = learningRateStart/(1+ii/500)
        phis = np.zeros(numberOfPeaks-1)
        for jj in range(0,numberOfPeaks-1):
            phis[jj] = 1/( np.sum( np.exp(thetas-thetas[jj]) ) + np.exp( - thetas[jj] ) )      
        phis = np.concatenate((phis, [(1-np.sum(phis))])) # add phi_k

        # prob_norm_term = np.zeros((numberOfPeaks)) # normalization parameter of p_j(x)
        curDataPt = np.copy(data[dataShuffle[ii%dataPts]])
        # x_m_mean = np.copy(curDataPt[np.newaxis,:]-means) # x - mu_j
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
            outprod_mu = np.outer(means[jj], means[jj])        
            mat_B = 2*learningRate*phis[jj]*alpha[jj] * (outprod_x - outprod_mu - sigmas[jj]) # making code more readable
            
            sigmas[jj] = sigmas[jj] + np.matmul(sigmas[jj], np.matmul(mat_B,sigmas[jj]))
            means[jj] = means[jj] + learningRate*phis[jj]*alpha[jj]*np.matmul(sigmas[jj],(curDataPt - means[jj]))
        
        # if ii % 1000 == 0:  
        #     print(phis)
        #     print(means) 
        #     print(sigmas)  
        #     print(ii)
    plottype = 'EM'
    if dimOfData == 2:
        plot2dimData(plottype, data, dataPts, numberOfPeaks, means, sigmas, thetas)
    saveFits(plottype, thetas, means, sigmas, learningRate)
    return thetas, means, sigmas

#below fits with a weighted algorithm that also removes the need for an inverse matrix
def fit_with_em_weighted(learningRateStart, data, dataShuffle, dataPts, dimOfData, numberOfPeaks, thetas, means, sigmas):
    curFig = 0
    sumParam = 1*np.ones(numberOfPeaks)
    for ii in range(0,1*dataPts):
        # add this so the algorithm settles down somewhere
        learningRate = learningRateStart/(1+ii/500)
        # calculate phis from current weight sum parameters
        phis = sumParam/np.sum(sumParam)
        curDataPt = np.copy(data[dataShuffle[ii%dataPts]])
        prob_log_term = np.zeros((numberOfPeaks)) # argument of multivariate exponential

        for jj in range(0,numberOfPeaks):
            prob_log_term[jj] = st.multivariate_normal.logpdf(curDataPt, mean=means[jj], cov=sigmas[jj])
            
        alpha = np.zeros(numberOfPeaks) # p_j(x)/p(x)
        for jj in range(0,numberOfPeaks):
            alpha[jj] = 1/(np.sum(np.exp(prob_log_term - prob_log_term[jj] )*phis))
        
        outprod_x = np.outer(curDataPt, curDataPt)
        sumParamTemp = np.copy(sumParam)
        
        for jj in range(0,numberOfPeaks):  
            outprod_mu_old = np.outer(means[jj], means[jj])   
            weights = learningRate*phis[jj]*alpha[jj]
            sumParam[jj] = sumParamTemp[jj] + weights
            means[jj] = 1/sumParam[jj]*( sumParamTemp[jj]*means[jj] + weights*curDataPt )
            outprod_mu = np.outer(means[jj], means[jj])   
            sigmas[jj] = 1/sumParam[jj]*( sumParamTemp[jj]*(sigmas[jj] + outprod_mu_old ) + weights*outprod_x )  - outprod_mu
        # uncomment lines below to enable real time observation, online plotting only support for 2D data
        # if ii % 1000 == 0:  
        #     print(phis)
        #     print(means) 
        #     print(sigmas)  
        #     print(ii)
        #     curFig = online_plotting_equiprob(numberOfPeaks, phis, means, sigmas, data[dataShuffle[0:1000]], curFig)
    # plt.figure(curFig)
    # plt.close()
    plottype = 'Expectation Maximization'
    if dimOfData == 2:
        plot2dimData(plottype, data, dataPts, numberOfPeaks, means, sigmas, thetas)
    saveFits(plottype, thetas, means, sigmas, learningRate)
    #find thetas from phis
    thetas = phis[0:-1]/phis[numberOfPeaks-1]
    thetas = np.log(thetas)
    return thetas, means, sigmas
# fit with l0 with inverse approximation
def fit_with_l0(learningRateStart, data, dataShuffle, dataPts, dimOfData, numberOfPeaks, thetas, means, sigmas):
    for ii in range(0,1*dataPts):
        learningRate = learningRateStart/(1+ii/500)
        phis = np.zeros(numberOfPeaks-1)
        for jj in range(0,numberOfPeaks-1):
            phis[jj] = 1/( np.sum( np.exp(thetas-thetas[jj]) ) + np.exp( - thetas[jj] ) )      
        phis = np.concatenate((phis, [(1-np.sum(phis))])) # add phi_k

        curDataPt = np.copy(data[dataShuffle[ii%dataPts]])
        prob_log_term = np.zeros((numberOfPeaks)) # argument of multivariate exponential will be put here

        for jj in range(0,numberOfPeaks):
            prob_log_term[jj] = st.multivariate_normal.logpdf(curDataPt, mean=means[jj], cov=sigmas[jj])
        lbar = np.sum(prob_log_term*phis)
        deltas = prob_log_term - lbar

        # theta update rule (l0)
        thetas = thetas + learningRate*phis[0:-1]*deltas[0:-1]

        outprod_x = np.outer(curDataPt, curDataPt)
        
        for jj in range(0,numberOfPeaks):
            outprod_mu = np.outer(means[jj], means[jj])
            #weight for this method
            weights = learningRate*phis[jj]
            mat_B = 2*weights* (outprod_x - outprod_mu - sigmas[jj])
            #approximation of inverse. Assume weights are small for this to be true.
            sigmas[jj] = sigmas[jj] + np.matmul(sigmas[jj], np.matmul(mat_B,sigmas[jj]))
            means[jj] = means[jj] + weights*np.matmul(sigmas[jj],(curDataPt - means[jj]))
        
        # if ii % 1000 == 0:  
        #     print(phis)
        #     print(means) 
        #     print(sigmas)  
        #     print(ii)
    plottype = 'l0'
    if dimOfData == 2:
        plot2dimData(plottype, data, dataPts, numberOfPeaks, means, sigmas, thetas)
    saveFits(plottype, thetas, means, sigmas, learningRate)
    return thetas, means, sigmas
# with with l0 using weights
def fit_with_l0_weighted(learningRateStart, data, dataShuffle, dataPts, dimOfData, numberOfPeaks, thetas, means, sigmas):
    sumParam = 1*np.ones(numberOfPeaks)
    for ii in range(0,1*dataPts):
        learningRate = learningRateStart/(1+ii/500)
        # phis for sum params
        phis = sumParam/np.sum(sumParam)

        curDataPt = np.copy(data[dataShuffle[ii%dataPts]])
        prob_log_term = np.zeros((numberOfPeaks)) # argument of multivariate exponential

        for jj in range(0,numberOfPeaks):
            prob_log_term[jj] = st.multivariate_normal.logpdf(curDataPt, mean=means[jj], cov=sigmas[jj])
        
        ljs = prob_log_term
        lbar = np.sum(ljs*phis)
        deltas =  ljs - lbar
            
        # theta update rule (EM)
        thetas = thetas + learningRate*phis[0:-1]*deltas[0:-1]
        outprod_x = np.outer(curDataPt, curDataPt)
        sumParamTemp = np.copy(sumParam)
        
        for jj in range(0,numberOfPeaks):  
            outprod_mu_old = np.outer(means[jj], means[jj])   
            weights = learningRate*phis[jj]
            sumParam[jj] = sumParamTemp[jj] + weights
            means[jj] = 1/sumParam[jj]*( sumParamTemp[jj]*means[jj] + weights*curDataPt )
            outprod_mu = np.outer(means[jj], means[jj])   
            sigmas[jj] = 1/sumParam[jj]*( sumParamTemp[jj]*(sigmas[jj] + outprod_mu_old ) + weights*outprod_x )  - outprod_mu
        
        # if ii % 1000 == 0:  
        #     print(phis)
        #     print(means) 
        #     print(sigmas)  
        #     print(ii)
    plottype = 'l0_weighted'
    if dimOfData == 2:
        plot2dimData(plottype, data, dataPts, numberOfPeaks, means, sigmas, thetas)
    saveFits(plottype, thetas, means, sigmas, learningRate)
    return thetas, means, sigmas
#fit with l_0 + l_2 with weights
def fit_with_ltilde(learningRateStart, data, dataShuffle, dataPts, dimOfData, numberOfPeaks, thetas, means, sigmas):
    for ii in range(0,1*dataPts):
        learningRate = learningRateStart/(1+ii/500)
        phis = np.zeros(numberOfPeaks-1)
        for jj in range(0,numberOfPeaks-1):
            phis[jj] = 1/( np.sum( np.exp(thetas-thetas[jj]) ) + np.exp( - thetas[jj] ) )      
        phis = np.concatenate((phis, [(1-np.sum(phis))])) # add phi_k

        curDataPt = np.copy(data[dataShuffle[ii%dataPts]])
        prob_log_term = np.zeros((numberOfPeaks)) # argument of multivariate exponential, logged
        for jj in range(0,numberOfPeaks):
            prob_log_term[jj] = st.multivariate_normal.logpdf(curDataPt, mean=means[jj], cov=sigmas[jj])
        
        ljs = prob_log_term
        lbar = np.sum(ljs*phis)
        deltas =  ljs - lbar
        
        #square of deltas
        deltassqrd = deltas*deltas
        
        # variance of log likelihood
        sigma_ll2 = np.sum(phis*deltassqrd)

        # theta update rule (ltilde)
        thetas = thetas + learningRate*phis[0:-1]*(deltas[0:-1] + (deltassqrd[0:-1] - sigma_ll2)/(2 + sigma_ll2) )
        outprod_x = np.outer(curDataPt, curDataPt)
        
        for jj in range(0,numberOfPeaks):
            outprod_mu = np.outer(means[jj], means[jj])
            # real update rule
            weights = learningRate*phis[jj] *(1 + (2*deltassqrd[jj])/(2 + sigma_ll2))
            #modified update rule, attempt for a better fit, but does not work that well
            # weights = learningRate*phis[jj] *(0 + (2*deltassqrd[jj])/(2 + sigma_ll2))
            mat_B = 2*weights* (outprod_x - outprod_mu - sigmas[jj])
            sigmas[jj] = sigmas[jj] + np.matmul(sigmas[jj], np.matmul(mat_B,sigmas[jj]))
            means[jj] = means[jj] + weights*np.matmul(sigmas[jj],(curDataPt - means[jj]))
        
        # if ii % 1000 == 0:  
        #     print(phis)
        #     print(means) 
        #     print(sigmas)  
        #     print(ii)
    plottype = 'Delta Method'
    if dimOfData == 2:
        plot2dimData(plottype, data, dataPts, numberOfPeaks, means, sigmas, thetas)
    saveFits(plottype, thetas, means, sigmas, learningRate)
    return thetas, means, sigmas
# fit with using the actual inverse method; done for comparison with approximation results
def fit_with_ltilde_inv(learningRateStart, data, dataShuffle, dataPts, dimOfData, numberOfPeaks, thetas, means, sigmas):
    # curFig = 0
    invSigmas = np.zeros(sigmas.shape)
    #compute inverse for sigmas and keep track to avoid doing a calculation twice
    for ii in range(0, numberOfPeaks):
        invSigmas[ii] = np.linalg.inv(sigmas[ii])
    
    for ii in range(0,1*dataPts):
        learningRate = learningRate/(1+ii/500)
        phis = np.zeros(numberOfPeaks-1)
        for jj in range(0,numberOfPeaks-1):
            phis[jj] = 1/( np.sum( np.exp(thetas-thetas[jj]) ) + np.exp( - thetas[jj] ) )      
        phis = np.concatenate((phis, [(1-np.sum(phis))])) # add phi_k

        curDataPt = np.copy(data[dataShuffle[ii%dataPts]])
        prob_log_term = np.zeros((numberOfPeaks)) # argument of multivariate exponential

        for jj in range(0,numberOfPeaks):
            prob_log_term[jj] = st.multivariate_normal.logpdf(curDataPt, mean=means[jj], cov=sigmas[jj])
        
        ljs = prob_log_term
        lbar = np.sum(ljs*phis)
        deltas =  ljs - lbar
        #try to remove very inconsistent points from current fit from driving it too much
        deltas = np.minimum(np.maximum(deltas, np.zeros(deltas.shape)), np.ones(deltas.shape))
        #square of deltas
        deltassqrd = deltas*deltas
        
        # variance of log likelihood
        sigma_ll2 = np.sum(phis*deltassqrd)
            
        # theta update rule (EM)
        # alpha = np.zeros(numberOfPeaks) # p_j(x)/p(x)
        # for jj in range(0,numberOfPeaks):
        #     alpha[jj] = 1/(np.sum(np.exp(prob_log_term - prob_log_term[jj] )*phis))
        # thetas = thetas + learningRate*phis[0:-1]*(alpha[0:-1]-1)

        # theta update rule (ltilde)
        thetas = thetas + learningRate*phis[0:-1]*(deltas[0:-1] + (deltassqrd[0:-1] - sigma_ll2)/(2 + sigma_ll2) )
        outprod_x = np.outer(curDataPt, curDataPt)
        
        for jj in range(0,numberOfPeaks):
            outprod_mu = np.outer(means[jj], means[jj])
            sigmaCurInv = np.copy(invSigmas[jj])
            oldSigma = np.copy(sigmas[jj])
            # true rule
            weight = learningRate*phis[jj] *(1 + (2*deltassqrd[jj])/(2 + sigma_ll2))

            #modified rule that kinda works with EM update rule
            # weight = learningRate*phis[jj] *np.exp(0 + (2*deltassqrd[jj])/(2 + sigma_ll2))

            # EM update rule
            # weight =  learningRate*phis[jj]*alpha[jj]

            mat_B = (outprod_x - oldSigma - outprod_mu)
            invSigmas[jj] = sigmaCurInv - 2*weight*mat_B
            sigmas[jj] = np.linalg.inv(invSigmas[jj])
            #mean update first part, inv(sigma_old)*mu
            meanFirst = np.matmul(sigmaCurInv,means[jj])
            #second part weight*(x-mu)
            meanSecond = weight*(curDataPt-means[jj])
            #last part. multiply the sum of the first two parts with new sigma
            means[jj] = np.matmul(sigmas[jj],meanFirst+meanSecond)
            #uncomment lines below if you would like real time feedback
    #     if ii % 500 == 0:  
    #         print(phis)
    #         print(means) 
    #         print(sigmas)  
    #         print(ii)
    #         curFig = online_plotting_equiprob(numberOfPeaks, phis, means, sigmas, data[dataShuffle[0:1000]], curFig)
    # plt.figure(curFig)
    # plt.close()
    plottype = 'ltilde_inv'
    if dimOfData == 2:
        plot2dimData(plottype, data, dataPts, numberOfPeaks, means, sigmas, thetas)
    saveFits(plottype, thetas, means, sigmas, learningRate)
    return thetas, means, sigmas
# fit with l_0 + l_2 method and using weights
def fit_with_ltilde_weighted(learningRateStart, data, dataShuffle, dataPts, dimOfData, numberOfPeaks, thetas, means, sigmas):
    # curFig=0
    sumParam = 1*np.ones(numberOfPeaks)
    for ii in range(0,1*dataPts):
        # phis for sum params
        phis = sumParam/np.sum(sumParam)
        curDataPt = np.copy(data[dataShuffle[ii%dataPts]])
        prob_log_term = np.zeros((numberOfPeaks)) # argument of multivariate exponential

        for jj in range(0,numberOfPeaks):
            prob_log_term[jj] = st.multivariate_normal.logpdf(curDataPt, mean=means[jj], cov=sigmas[jj])
            
        ljs = prob_log_term
        lbar = np.sum(ljs*phis)
        deltas =  ljs - lbar
        
        #square of deltas
        deltassqrd = deltas*deltas
        
        # variance of log likelihood
        sigma_ll2 = np.sum(phis*deltassqrd)
            
        # theta update rule (ltilde)
        thetas = thetas + learningRate*phis[0:-1]*(deltas[0:-1] + (deltassqrd[0:-1] - sigma_ll2)/(2 + sigma_ll2) )
        outprod_x = np.outer(curDataPt, curDataPt)
        sumParamTemp = np.copy(sumParam)
        
        for jj in range(0,numberOfPeaks):  
            outprod_mu_old = np.outer(means[jj], means[jj])   
            weights = learningRate*phis[jj] *(1 + (2*deltassqrd[jj])/(2 + sigma_ll2))
            sumParam[jj] = sumParamTemp[jj] + weights
            means[jj] = 1/sumParam[jj]*( sumParamTemp[jj]*means[jj] + weights*curDataPt )
            outprod_mu = np.outer(means[jj], means[jj])   
            sigmas[jj] = 1/sumParam[jj]*( sumParamTemp[jj]*(sigmas[jj] + outprod_mu_old ) + weights*outprod_x )  - outprod_mu
        #uncomment below if you would like realtime feedback
    #     if ii % 1000 == 0:  
    #         print(phis)
    #         print(means) 
    #         print(sigmas)  
    #         print(ii)
    #         curFig = online_plotting_equiprob(numberOfPeaks, phis, means, sigmas, data[dataShuffle[0:1000]], curFig)
    # plt.figure(curFig)
    # plt.close()
    plottype = 'ltilde_weighted'
    if dimOfData == 2:
        plot2dimData(plottype, data, dataPts, numberOfPeaks, means, sigmas, thetas)
    #find thetas from phis
    thetas = phis[0:-1]/phis[numberOfPeaks-1]
    thetas = np.log(thetas)
    saveFits(plottype, thetas, means, sigmas, learningRate)
    return thetas, means, sigmas
    
# inspired method that probably does not work
# not maintained
def fit_with_ltilde_insp1(learningRateStart, data, dataShuffle, dataPts, dimOfData, numberOfPeaks, thetas, means, sigmas):
    for ii in range(0,1*dataPts):
        learningRate = learningRateStart/(1+ii/500)
        phis = np.zeros(numberOfPeaks-1)
        for jj in range(0,numberOfPeaks-1):
            phis[jj] = 1/( np.sum( np.exp(thetas-thetas[jj]) ) + np.exp( - thetas[jj] ) )      
        phis = np.concatenate((phis, [(1-np.sum(phis))])) # add phi_k
        
        curDataPt = np.copy(data[dataShuffle[ii%dataPts]])
        prob_log_term = np.zeros((numberOfPeaks)) # argument of multivariate exponential

        for jj in range(0,numberOfPeaks):
            prob_log_term[jj] = st.multivariate_normal.logpdf(curDataPt, mean=means[jj], cov=sigmas[jj])
        
        ljs = prob_log_term
        lbar = np.sum(ljs*phis)
        deltas =  ljs - lbar
        
        #square of deltas
        # deltassqrd = deltas*deltas
        
        # variance of log likelihood
        # sigma_ll2 = np.sum(phis*deltassqrd)

        # theta update rule (ltilde)
        # thetas = thetas + learningRate*phis[0:-1]*(deltas[0:-1] + (deltassqrd[0:-1] - sigma_ll2)/(2 + sigma_ll2) )
        thetas = thetas + learningRate*phis[0:-1]*deltas[0:-1]
        outprod_x = np.outer(curDataPt, curDataPt)
        
        for jj in range(0,numberOfPeaks):
            outprod_mu = np.outer(means[jj], means[jj])  
            weight = 2*learningRate*phis[jj] *( deltas[jj] )
            # maybe independently calculate weight and just multiply both below.      
            # mat_B = 2*learningRate*phis[jj] *(1 + (2*deltassqrd[jj])/(2 + sigma_ll2))* (outprod_x - outprod_mu - sigmas[jj]) # making code more readable
            mat_B = weight* (outprod_x - outprod_mu - sigmas[jj]) # making code more readable
            
            sigmas[jj] = sigmas[jj] + np.matmul(sigmas[jj], np.matmul(mat_B,sigmas[jj]))
            # means[jj] = means[jj] + learningRate*phis[jj]*(1 + (2*deltassqrd[jj])/(2 + sigma_ll2))*np.matmul(sigmas[jj],(curDataPt - means[jj]))
            means[jj] = means[jj] + weight*np.matmul(sigmas[jj],(curDataPt - means[jj]))
        
        # if ii % 1000 == 0:  
        #     print(phis)
        #     print(means) 
        #     print(sigmas)  
        #     print(ii)
    plottype = 'ltilde_insp1'
    if dimOfData == 2:
        plot2dimData(plottype, data, dataPts, numberOfPeaks, means, sigmas, thetas)
    saveFits(plottype, thetas, means, sigmas, learningRate)
    return thetas, means, sigmas
# inspired method that probably does not work
# not maintained
def fit_with_ltilde_insp1_weighted(learningRateStart, data, dataShuffle, dataPts, dimOfData, numberOfPeaks, thetas, means, sigmas):
    sumParam = 1*np.ones(numberOfPeaks)
    for ii in range(0,1*dataPts):
        # learningRate = learningRateStart/(1+ii/500)
        phis = np.zeros(numberOfPeaks-1)
        for jj in range(0,numberOfPeaks-1):
            phis[jj] = 1/( np.sum( np.exp(thetas-thetas[jj]) ) + np.exp( - thetas[jj] ) )      
        phis = np.concatenate((phis, [(1-np.sum(phis))])) # add phi_k

        curDataPt = np.copy(data[dataShuffle[ii%dataPts]])
        prob_log_term = np.zeros((numberOfPeaks)) # argument of multivariate exponential

        for jj in range(0,numberOfPeaks):
            prob_log_term[jj] = st.multivariate_normal.logpdf(curDataPt, mean=means[jj], cov=sigmas[jj])
            
        ljs = prob_log_term
        lbar = np.sum(ljs*phis)
        deltas =  ljs - lbar
        
        #square of deltas
        # deltassqrd = deltas*deltas
        
        # variance of log likelihood
        # sigma_ll2 = np.sum(phis*deltassqrd)
            
        # theta update rule (ltilde)
        thetas = thetas + learningRate*phis[0:-1]*deltas[0:-1]
        outprod_x = np.outer(curDataPt, curDataPt)
        sumParamTemp = np.copy(sumParam)
        
        for jj in range(0,numberOfPeaks):  
            outprod_mu_old = np.outer(means[jj], means[jj])   
            
            weights = learningRate*phis[jj] *( deltas[jj] )
            sumParam[jj] = sumParamTemp[jj] + weights
            means[jj] = 1/sumParam[jj]*( sumParamTemp[jj]*means[jj] + weights*curDataPt )
            outprod_mu = np.outer(means[jj], means[jj])   
            sigmas[jj] = 1/sumParam[jj]*( sumParamTemp[jj]*(sigmas[jj] + outprod_mu_old ) + weights*outprod_x )  - outprod_mu
        
        # if ii % 1000 == 0:  
        #     print(phis)
        #     print(means) 
        #     print(sigmas)  
        #     print(ii)
    plottype = 'ltilde_insp1_weighted'
    if dimOfData == 2:
        plot2dimData(plottype, data, dataPts, numberOfPeaks, means, sigmas, thetas)
    saveFits(plottype, thetas, means, sigmas, learningRate)
    return thetas, means, sigmas
# inspired method that probably does not work
# not maintained
# not stable because abs(ljs) is unbounded and this assumes abs(weight) is small
def fit_with_ltilde_insp2(learningRateStart, data, dataShuffle, dataPts, dimOfData, numberOfPeaks, thetas, means, sigmas):
    for ii in range(0,1*dataPts):
        learningRate = learningRateStart/(1+ii/500)
        phis = np.zeros(numberOfPeaks-1)
        for jj in range(0,numberOfPeaks-1):
            phis[jj] = 1/( np.sum( np.exp(thetas-thetas[jj]) ) + np.exp( - thetas[jj] ) )      
        phis = np.concatenate((phis, [(1-np.sum(phis))])) # add phi_k
        
        curDataPt = np.copy(data[dataShuffle[ii%dataPts]])
        prob_log_term = np.zeros((numberOfPeaks)) # argument of multivariate exponential

        for jj in range(0,numberOfPeaks):
            prob_log_term[jj] = st.multivariate_normal.logpdf(curDataPt, mean=means[jj], cov=sigmas[jj])
        
        ljs = prob_log_term
        lbar = np.sum(ljs*phis)
        deltas =  ljs - lbar
        
        #square of deltas
        # deltassqrd = deltas*deltas
        
        # variance of log likelihood
        # sigma_ll2 = np.sum(phis*deltassqrd)

        # theta update rule (ltilde)
        # thetas = thetas + learningRate*phis[0:-1]*(deltas[0:-1] + (deltassqrd[0:-1] - sigma_ll2)/(2 + sigma_ll2) )
        thetas = thetas + learningRate*phis[0:-1]*deltas[0:-1]
        outprod_x = np.outer(curDataPt, curDataPt)
        
        for jj in range(0,numberOfPeaks):
            outprod_mu = np.outer(means[jj], means[jj])  
            weight = 2*learningRate*phis[jj] *( ljs[jj] )
            print(weight)
            # maybe independently calculate weight and just multiply both below.      
            # mat_B = 2*learningRate*phis[jj] *(1 + (2*deltassqrd[jj])/(2 + sigma_ll2))* (outprod_x - outprod_mu - sigmas[jj]) # making code more readable
            mat_B = weight* (outprod_x - outprod_mu - sigmas[jj]) # making code more readable
            
            sigmas[jj] = sigmas[jj] + np.matmul(sigmas[jj], np.matmul(mat_B,sigmas[jj]))
            # means[jj] = means[jj] + learningRate*phis[jj]*(1 + (2*deltassqrd[jj])/(2 + sigma_ll2))*np.matmul(sigmas[jj],(curDataPt - means[jj]))
            means[jj] = means[jj] + weight*np.matmul(sigmas[jj],(curDataPt - means[jj]))
        print(ii)
        print(means)
        print(sigmas)
        # if ii % 1000 == 0:  
        #     print(phis)
        #     print(means) 
        #     print(sigmas)  
        #     print(ii)
    plottype = 'ltilde_insp2'
    if dimOfData == 2:
        plot2dimData(plottype, data, dataPts, numberOfPeaks, means, sigmas, thetas)
    saveFits(plottype, thetas, means, sigmas, learningRate)
    return thetas, means, sigmas
# inspired method that probably does not work
# not maintained
def fit_with_ltilde_insp2_weighted(learningRateStart, data, dataShuffle, dataPts, dimOfData, numberOfPeaks, thetas, means, sigmas):
    sumParam = 1*np.ones(numberOfPeaks)
    for ii in range(0,1*dataPts):
        # learningRate = learningRateStart/(1+ii/500)
        phis = np.zeros(numberOfPeaks-1)
        for jj in range(0,numberOfPeaks-1):
            phis[jj] = 1/( np.sum( np.exp(thetas-thetas[jj]) ) + np.exp( - thetas[jj] ) )      
        phis = np.concatenate((phis, [(1-np.sum(phis))])) # add phi_k

        curDataPt = np.copy(data[dataShuffle[ii%dataPts]])
        prob_log_term = np.zeros((numberOfPeaks)) # argument of multivariate exponential

        for jj in range(0,numberOfPeaks):
            prob_log_term[jj] = st.multivariate_normal.logpdf(curDataPt, mean=means[jj], cov=sigmas[jj])
            
        ljs = prob_log_term
        lbar = np.sum(ljs*phis)
        deltas =  ljs - lbar
        
        #square of deltas
        # deltassqrd = deltas*deltas
        
        # variance of log likelihood
        # sigma_ll2 = np.sum(phis*deltassqrd)
            
        # theta update rule (ltilde)
        thetas = thetas + learningRate*phis[0:-1]*deltas[0:-1]
        outprod_x = np.outer(curDataPt, curDataPt)
        sumParamTemp = np.copy(sumParam)
        
        for jj in range(0,numberOfPeaks):  
            outprod_mu_old = np.outer(means[jj], means[jj])   
            
            weights = learningRate*phis[jj] *( ljs[jj] )
            sumParam[jj] = sumParamTemp[jj] + weights
            means[jj] = 1/sumParam[jj]*( sumParamTemp[jj]*means[jj] + weights*curDataPt )
            outprod_mu = np.outer(means[jj], means[jj])   
            sigmas[jj] = 1/sumParam[jj]*( sumParamTemp[jj]*(sigmas[jj] + outprod_mu_old ) + weights*outprod_x )  - outprod_mu
        
        # if ii % 1000 == 0:  
        #     print(phis)
        #     print(means) 
        #     print(sigmas)  
        #     print(ii)
    plottype = 'ltilde_insp2_weighted'
    if dimOfData == 2:
        plot2dimData(plottype, data, dataPts, numberOfPeaks, means, sigmas, thetas)
    saveFits(plottype, thetas, means, sigmas, learningRate)
    return thetas, means, sigmas

# start program
cwd = os.getcwd()
dirSave = os.path.join(cwd,"gen_seed_" + str(seedNum))

data = np.load(os.path.join(dirSave,'gen.npy'))
dataPts = data.shape[0]
dimOfData = data.shape[1]

dataShuffle = np.arange(dataPts)
np.random.shuffle(dataShuffle)

# code to get this parameter from saved generated parameters to be implemented
numberOfPeaks = 5 # corresponds to number of Gaussians

# parameter initialization as project was not about initializing parameters
numberOfStartingPoints = 10
labeled_data = np.zeros([numberOfPeaks,numberOfStartingPoints*(dimOfData-1),dimOfData])
mean_guess = np.zeros([numberOfPeaks,dimOfData]) # number of gaussians x dim
sigma_guess = np.zeros([numberOfPeaks, dimOfData, dimOfData]) # number of gaussians x dim x dim
data_outer = np.zeros([numberOfPeaks,numberOfStartingPoints*(dimOfData-1),dimOfData,dimOfData])
    
for ii in range(0, numberOfPeaks):
    labeled_data = np.load(os.path.join(dirSave,'gen'+ str(ii) +'.npy'))[0:numberOfStartingPoints*(dimOfData-1)]

    data_outer[ii] = np.einsum('ij,ik->ijk',labeled_data,labeled_data)
    mean_guess[ii] = np.mean(labeled_data,axis=0)

    sigma_guess[ii,:,:] = np.mean(data_outer[ii] - np.outer(mean_guess[ii],mean_guess[ii]),axis=0)

means = np.copy(mean_guess)
sigmas = np.copy(sigma_guess)
# debugging
# pdb.set_trace()

thetas_guess = np.random.uniform(low=0,high=0.1, size=numberOfPeaks-1) # uniform distribution of phis close to each other
thetas = np.copy(thetas_guess)

if dimOfData == 2:
    plottype_init = 'init_plot'
    plot2dimData(plottype_init, data, dataPts, numberOfPeaks, means, sigmas, thetas)

# START SGD methods; all of these methods are written in a hacky fashion as we had limited time, but EM_weighted and ltilde_weighted should work according to theory
# as we found out ltilde = l_0 + l_2 does not work very well
learningRate = 1.0
while True:
    try:
        thetas = np.copy(thetas_guess)
        means = np.copy(mean_guess)
        sigmas = np.copy(sigma_guess)
        start_time_EM = time.time()
        thetas, means, sigmas = fit_with_em(learningRate, data, dataShuffle, dataPts, dimOfData, numberOfPeaks, thetas, means, sigmas)
        end_time = time.time()
        conv_time = end_time - start_time_EM 
        np.savetxt(os.path.join(dirSave,'convergence_time_EM.txt'), np.array([[conv_time]]))
        phis = np.zeros(numberOfPeaks-1)
        for jj in range(0,numberOfPeaks-1):
            phis[jj] = 1/( np.sum( np.exp(thetas-thetas[jj]) ) + np.exp( - thetas[jj] ) )      
        phis = np.concatenate((phis, [(1-np.sum(phis))])) # add phi_k
        propDt = phis*dataPts
        print("Fitted EM")
        print(propDt)
        print(means)
        print(sigmas)
        print(thetas)
        break
    except ValueError as e:
        print(str(e))
        learningRate /= 5
        print("Fit Diverged in EM. Reduced Lambda to: "+ str(learningRate) + ".")

learningRate = 1.0
while True:
    try:
        start_time_l0 = time.time()
        thetas, means, sigmas = fit_with_em_weighted(learningRate, data, dataShuffle, dataPts, dimOfData, numberOfPeaks, thetas, means, sigmas)
        end_time = time.time()
        conv_time = end_time - start_time_l0 
        np.savetxt(os.path.join(dirSave,'convergence_time_EM_weights.txt'), np.array([[conv_time]]))
        phis = np.zeros(numberOfPeaks-1)
        for jj in range(0,numberOfPeaks-1):
            phis[jj] = 1/( np.sum( np.exp(thetas-thetas[jj]) ) + np.exp( - thetas[jj] ) )      
        phis = np.concatenate((phis, [(1-np.sum(phis))])) # add phi_k
        propDt = phis*dataPts
        print("Fitted EM weights")
        print(propDt)
        print(means)    
        print(sigmas)
        print(thetas)
        break
    except ValueError as e:
        print(str(e))
        learningRate /= 5
        print("Fit Diverged in EM weights. Reduced Lambda to: "+ str(learningRate) + ".")

learningRate = 1.0
while True:
    try:
        # print("Lambda is: " + np.array2string(np.array([[learningRate]]), formatter={'float_kind':lambda x: "%.10f" % x}))
        thetas = np.copy(thetas_guess)
        means = np.copy(mean_guess)
        sigmas = np.copy(sigma_guess)
        start_time_l0 = time.time()
        thetas, means, sigmas = fit_with_l0(learningRate, data, dataShuffle, dataPts, dimOfData, numberOfPeaks, thetas, means, sigmas)
        end_time = time.time()
        conv_time = end_time - start_time_l0 
        np.savetxt(os.path.join(dirSave,'convergence_time_l0.txt'), np.array([[conv_time]]))
        phis = np.zeros(numberOfPeaks-1)
        for jj in range(0,numberOfPeaks-1):
            phis[jj] = 1/( np.sum( np.exp(thetas-thetas[jj]) ) + np.exp( - thetas[jj] ) )      
        phis = np.concatenate((phis, [(1-np.sum(phis))])) # add phi_k
        propDt = phis*dataPts
        print("Fitted l0")
        print(propDt)
        print(means)    
        print(sigmas)
        print(thetas)
        break
    except ValueError as e:
        print(str(e))
        learningRate /= 5
        print("Fit Diverged in l0. Reduced Lambda to: "+ str(learningRate) + ".")
        print("Lambda is: \n" + np.array2string(np.array([[learningRate]]), formatter={'float_kind':lambda x: "%.10f" % x}))

learningRate = 1.0
# learningRate = 0.01
while True:
    try:
        thetas = np.copy(thetas_guess)
        means = np.copy(mean_guess)
        sigmas = np.copy(sigma_guess)
        # learningRate = 0.1
        start_time_l0 = time.time()
        thetas, means, sigmas = fit_with_l0_weighted(learningRate, data, dataShuffle, dataPts, dimOfData, numberOfPeaks, thetas, means, sigmas)
        end_time = time.time()
        conv_time = end_time - start_time_l0 
        np.savetxt(os.path.join(dirSave,'convergence_time_l0_weighted.txt'), np.array([[conv_time]]))
        phis = np.zeros(numberOfPeaks-1)
        for jj in range(0,numberOfPeaks-1):
            phis[jj] = 1/( np.sum( np.exp(thetas-thetas[jj]) ) + np.exp( - thetas[jj] ) )      
        phis = np.concatenate((phis, [(1-np.sum(phis))])) # add phi_k
        propDt = phis*dataPts
        print("Fitted l0 weighted")
        print(propDt)
        print(means)    
        print(sigmas)
        print(thetas)
        break
    except ValueError as e:
        print(str(e))
        learningRate /= 5
        print("Fit Diverged in l0 weighted. Reduced Lambda to: "+ str(learningRate) + ".")

learningRate = 1.0
# learningRate = 0.01
while True:
    try:
        thetas = np.copy(thetas_guess)
        means = np.copy(mean_guess)
        sigmas = np.copy(sigma_guess)
        # learningRate = 0.05
        start_time_l0 = time.time()
        thetas, means, sigmas = fit_with_ltilde(learningRate, data, dataShuffle, dataPts, dimOfData, numberOfPeaks, thetas, means, sigmas)
        end_time = time.time()
        conv_time = end_time - start_time_l0 
        np.savetxt(os.path.join(dirSave,'convergence_time_ltilde.txt'), np.array([[conv_time]]))
        phis = np.zeros(numberOfPeaks-1)
        for jj in range(0,numberOfPeaks-1):
            phis[jj] = 1/( np.sum( np.exp(thetas-thetas[jj]) ) + np.exp( - thetas[jj] ) )      
        phis = np.concatenate((phis, [(1-np.sum(phis))])) # add phi_k
        propDt = phis*dataPts
        print("Fitted ltilde")
        print(propDt)
        print(means)    
        print(sigmas)
        print(thetas)
        break
    except ValueError as e:
        print(str(e))
        learningRate /= 5
        print("Fit Diverged. in ltilde Reduced Lambda to: "+ str(learningRate) + ".")

learningRate = 1.0
# learningRate = 0.001
while True:
    try:
        thetas = np.copy(thetas_guess)
        means = np.copy(mean_guess)
        sigmas = np.copy(sigma_guess)
        # learningRate = 0.05
        start_time_l0 = time.time()
        thetas, means, sigmas = fit_with_ltilde_weighted(learningRate, data, dataShuffle, dataPts, dimOfData, numberOfPeaks, thetas, means, sigmas)
        end_time = time.time()
        conv_time = end_time - start_time_l0 
        np.savetxt(os.path.join(dirSave,'convergence_time_ltilde_weighted.txt'), np.array([[conv_time]]))
        phis = np.zeros(numberOfPeaks-1)
        for jj in range(0,numberOfPeaks-1):
            phis[jj] = 1/( np.sum( np.exp(thetas-thetas[jj]) ) + np.exp( - thetas[jj] ) )      
        phis = np.concatenate((phis, [(1-np.sum(phis))])) # add phi_k
        propDt = phis*dataPts
        print("Fitted ltilde weighted")
        print(propDt)
        print(means)    
        print(sigmas)
        print(thetas)
        break
    except ValueError as e:
        print(str(e))
        learningRate /= 5
        print("Fit Diverged in ltilde weighted. Reduced Lambda to: "+ str(learningRate) + ".")

#not supported
# learningRate = 1.0
# # learningRate = 0.01
# while True:
#     try:
#         thetas = np.copy(thetas_guess)
#         means = np.copy(mean_guess)
#         sigmas = np.copy(sigma_guess)
#         # learningRate = 0.05
#         start_time_l0 = time.time()
#         thetas, means, sigmas = fit_with_ltilde_insp1(learningRate, data, dataShuffle, dataPts, dimOfData, numberOfPeaks, thetas, means, sigmas)
#         end_time = time.time()
#         conv_time = end_time - start_time_l0 
#         np.savetxt(os.path.join(dirSave,'convergence_time_ltilde_insp1.txt'), np.array([[conv_time]]))
#         phis = np.zeros(numberOfPeaks-1)
#         for jj in range(0,numberOfPeaks-1):
#             phis[jj] = 1/( np.sum( np.exp(thetas-thetas[jj]) ) + np.exp( - thetas[jj] ) )      
#         phis = np.concatenate((phis, [(1-np.sum(phis))])) # add phi_k
#         propDt = phis*dataPts
#         print("Fitted ltilde_insp1")
#         print(propDt)
#         print(means)    
#         print(sigmas)
#         print(thetas)
#         break
#     except ValueError as e:
#         print(str(e))
#         learningRate /= 5
#         print("Fit Diverged. in ltilde_insp1 Reduced Lambda to: "+ str(learningRate) + ".")
#         time.sleep(1)
# learningRate = 1.0
# while True:
#     try:
#         thetas = np.copy(thetas_guess)
#         means = np.copy(mean_guess)
#         sigmas = np.copy(sigma_guess)
#         # learningRate = 0.05
#         start_time_l0 = time.time()
#         thetas, means, sigmas = fit_with_ltilde_insp1_weighted(learningRate, data, dataShuffle, dataPts, dimOfData, numberOfPeaks, thetas, means, sigmas)
#         end_time = time.time()
#         conv_time = end_time - start_time_l0 
#         np.savetxt(os.path.join(dirSave,'convergence_time_ltilde_insp1_weighted.txt'), np.array([[conv_time]]))
#         phis = np.zeros(numberOfPeaks-1)
#         for jj in range(0,numberOfPeaks-1):
#             phis[jj] = 1/( np.sum( np.exp(thetas-thetas[jj]) ) + np.exp( - thetas[jj] ) )      
#         phis = np.concatenate((phis, [(1-np.sum(phis))])) # add phi_k
#         propDt = phis*dataPts
#         print("Fitted ltilde_insp1 weighted")
#         print(propDt)
#         print(means)    
#         print(sigmas)
#         print(thetas)
#         break
#     except ValueError as e:
#         print(str(e))
#         learningRate /= 5
#         print("Fit Diverged in ltilde_insp1 weighted. Reduced Lambda to: "+ str(learningRate) + ".")
#this does not work as the assumption that abs(weight) is small is violated.
# learningRate = 1.0
# while True:
#     try:
#         thetas = np.copy(thetas_guess)
#         means = np.copy(mean_guess)
#         sigmas = np.copy(sigma_guess)
#         # learningRate = 0.05
#         start_time_l0 = time.time()
#         thetas, means, sigmas = fit_with_ltilde_insp2(learningRate, data, dataShuffle, dataPts, dimOfData, numberOfPeaks, thetas, means, sigmas)
#         end_time = time.time()
#         conv_time = end_time - start_time_l0 
#         np.savetxt(os.path.join(dirSave,'convergence_time_ltilde_insp2.txt'), np.array([[conv_time]]))
#         phis = np.zeros(numberOfPeaks-1)
#         for jj in range(0,numberOfPeaks-1):
#             phis[jj] = 1/( np.sum( np.exp(thetas-thetas[jj]) ) + np.exp( - thetas[jj] ) )      
#         phis = np.concatenate((phis, [(1-np.sum(phis))])) # add phi_k
#         propDt = phis*dataPts
#         print("Fitted ltilde_insp2")
#         print(propDt)
#         print(means)    
#         print(sigmas)
#         print(thetas)
#         break
#     except ValueError as e:
#         print(str(e))
#         learningRate /= 5
#         print("Fit Diverged. in ltilde_insp2 Reduced Lambda to: "+ str(learningRate) + ".")
# learningRate = 1.0
# while True:
#     try:
#         thetas = np.copy(thetas_guess)
#         means = np.copy(mean_guess)
#         sigmas = np.copy(sigma_guess)
#         # learningRate = 0.05
#         start_time_l0 = time.time()
#         thetas, means, sigmas = fit_with_ltilde_insp2_weighted(learningRate, data, dataShuffle, dataPts, dimOfData, numberOfPeaks, thetas, means, sigmas)
#         end_time = time.time()
#         conv_time = end_time - start_time_l0 
#         np.savetxt(os.path.join(dirSave,'convergence_time_ltilde_insp2_weighted.txt'), np.array([[conv_time]]))
#         phis = np.zeros(numberOfPeaks-1)
#         for jj in range(0,numberOfPeaks-1):
#             phis[jj] = 1/( np.sum( np.exp(thetas-thetas[jj]) ) + np.exp( - thetas[jj] ) )      
#         phis = np.concatenate((phis, [(1-np.sum(phis))])) # add phi_k
#         propDt = phis*dataPts
#         print("Fitted ltilde_insp2 weighted")
#         print(propDt)
#         print(means)    
#         print(sigmas)
#         print(thetas)
#         break
#     except ValueError as e:
#         print(str(e))
#         learningRate /= 5
#         print("Fit Diverged in ltilde_insp2 weighted. Reduced Lambda to: "+ str(learningRate) + ".")

learningRate = 1.0
# learningRate = 0.01
while True:
    try:
        thetas = np.copy(thetas_guess)
        means = np.copy(mean_guess)
        sigmas = np.copy(sigma_guess)
        start_time_l0 = time.time()
        thetas, means, sigmas = fit_with_ltilde_inv(learningRate, data, dataShuffle, dataPts, dimOfData, numberOfPeaks, thetas, means, sigmas)
        end_time = time.time()
        conv_time = end_time - start_time_l0 
        np.savetxt(os.path.join(dirSave,'convergence_time_ltilde_inv.txt'), np.array([[conv_time]]))
        phis = np.zeros(numberOfPeaks-1)
        for jj in range(0,numberOfPeaks-1):
            phis[jj] = 1/( np.sum( np.exp(thetas-thetas[jj]) ) + np.exp( - thetas[jj] ) )      
        phis = np.concatenate((phis, [(1-np.sum(phis))])) # add phi_k
        propDt = phis*dataPts
        print("Fitted ltilde inv")
        print(propDt)
        print(means)    
        print(sigmas)
        print(thetas)
        break
    except ValueError as e:
        print(str(e))
        learningRate /= 5
        print("Fit Diverged in ltilde inv. Reduced Lambda to: "+ str(learningRate) + ".")