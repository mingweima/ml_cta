import pandas as pd
import numpy as np
from scipy.io import loadmat, savemat
import scipy.stats as stats
from scipy.linalg import fractional_matrix_power as mpower

from numpy import diag, sqrt, sort, argsort, log, isnan, var, std, nanmean, kron
from numpy.random import randn, choice, random
from numpy.linalg import norm, inv, lstsq, svd, eig
from numpy.matlib import repmat


def main_simulation(T, N):
    """
    Input:

    T : # of time points, in form of list, ie. [T]
    N : # of assets, in form of list, ie. [N]

    Output:

    1. a .mat file titled with given T and N
    2. a dict named "output" that are directly returned by this function

    """

    # Calibration

    Tlist = T  # [120,240,480];
    Nlist = N  # [50,100,200];

    for tt in range(len(Tlist)):
        for nn in range(len(Nlist)):

            M = 1000  # %5000;  % # of MCs
            T = Tlist[tt]  # 600;   % # of time periods
            n = Nlist[nn]  # 200;%200;   % # of stocks
            p = 5  # of latent factors
            d = 4  # of factor proxies
            pmax = 10  # 1:pmax is the list of number of factors
            q = np.floor(T ** (1 / 4)) + 1

            # rng(1);
            # load parameters
            param = pd.read_pickle('Calibrated_Parameters.pkl')
            Sigmabeta = param['Sigmabeta']  # p*p
            beta0 = param['beta0']  # p*1
            Sigmau = param['Sigmau']  # n*n
            eta = param['eta']  # d*p
            gamma = param['gamma']  # p*1
            sigmaalpha = param['sigmaalpha']  # 1*1
            Sigmaw = param['Sigmaw']  # d*d
            Sigmav = param['Sigmav']  # p*p

            # create some zero matrix for later use
            mu = np.zeros([p, 1])  # mean vector of the factors
            gamma0 = 0
            alpha0 = 0
            Gammahat = np.zeros([d, pmax, M])
            GammaFM = np.zeros([d, M])
            GammaTrueFM = np.zeros([d, M])

            avarhat = np.zeros([d, pmax, M])
            avarhatFM = np.zeros([d, M])

            gthat = np.zeros([d, T, pmax, M])
            gtMP = np.zeros([d, M])
            gtavarMP = np.zeros([d, M])

            phat = np.zeros([3, M])
            gtavar = np.zeros([d, T, M])
            gttrue = np.zeros([d, T, M])

            xi = np.zeros([d, 1]);

            dlist = [int(x) for x in divisorGenerator(n)]  # all divisors of n
            dtemp = [abs(x - np.floor(np.sqrt(n))) for x in dlist]
            idx = dtemp.index(min(dtemp))

            beta = np.repeat(beta0.T, n, axis=0) + \
                   randn(n, p) @ (mpower((Sigmabeta - beta0 @ beta0.T), 0.5))  # factor loading (n,p)

            # % zero-beta rate + true risk premiums of the proxies
            Gammatrue = (eta @ gamma).reshape(-1, 1)

            Sigma = beta @ Sigmav @ beta.T + Sigmau  # n*n
            lambdastar = eta @ Sigmav @ beta.T @ inv(Sigma) @ (beta @ gamma)  # d*1

            # start MC steps

            for iMC in range(M):
                print('Currently on #', iMC + 1, '/', M, end='\r', flush=True)

                alpha = (alpha0 + randn(n, 1) @ sigmaalpha)  # exprnd(sigmaalpha,n,1)-sigmaalpha;
                vt = (mpower(Sigmav, 0.5)) @ randn(p, T)  # factor innovations (p,T)
                ut = (mpower(Sigmau, 0.5)) @ randn(n, T)  # residual innovations (n,T)

                wt = (mpower(Sigmaw[:d, :d], 0.5)) @ randn(d, T)  # proxies residual innovations(d,T)
                rt = np.repeat(np.ones([n, 1]) * gamma0 + alpha, T, axis=1) + \
                     np.repeat(beta @ gamma, T, axis=1) + beta @ vt + ut  # returns (n,T)
                gt = np.repeat(xi, T, axis=1) + eta @ vt + wt  # (d,T)
                gttrue[:, :, iMC] = eta @ vt  # (d,T)

                # Estimation

                res = PCAFM(rt, gt, pmax, q)
                resFMvsMP = FM(rt, vt, gt, q)

                # Get results from PCAFM and FM

                GammaFM[:, iMC] = res['GammaFM'].flatten()
                Gammahat[:, :, iMC] = res['Gammahat']

                GammaTrueFM[:, iMC] = resFMvsMP['Gammahat'].flatten()

                gtMP[:, iMC] = res['gtMP']
                gtavarMP[:, iMC] = res['avarMP'].flatten()
                avarhat[:, :, iMC] = res['avarhat']
                avarhatFM[:, iMC] = res['avarhatFM'].flatten()

                gthat[:, :, :, iMC] = res['gthat']
                phat[:, iMC] = res['phat'].flatten()

    output_list = ['GammaFM', 'avarhatFM', 'Gammahat', 'avarhat', 'gtMP', 'gtavarMP',
                   'gthat', 'Gammatrue', 'GammaTrueFM', 'd', 'p']

    output = {}
    for varname in output_list:
        output[varname] = eval(varname)
    pd.to_pickle(output, 'py_Simulation_Results_T_' + str(T) + '_N_' + str(n) + '.pkl')

    return output


def PCAFM(rt, gt, pmax, q):
    # This version gives average return std
    # This version allows for ridge
    # this fixed the output size of gtMp when n > T

    ''' 
    Input: 
    
    rt          : n by T matrix
    gt          : d by T factor proxies
    pmax        : the upperbound of the number of latent factors
    q           : # of lags used in Newy-West standard errors

    Output:
    
    GammaFM     : d by 1 vector of Fama and MacBeth estimator
    Gammahat    : d by pmax matrix of risk premia estimates
    avarhatFM   : d by 1 vector of the avar of risk premia using FM
    avarhat     : d by pmax matrix of the avar of risk premia estimates
    gthat       : d by T by pmax matrix of cleaned factor proxies
    alphahat    : n by pmax matrix of pricing error estimates
    
    gtMP        : d by 1 vector of the risk premia using MP
    avarMP      : d by 1 vector of avar of the risk premia using MP
    
    '''

    # INITIALIZATION

    n, T = rt.shape
    d = gt.shape[0]

    GammaFM = np.zeros(d)
    avarhatFM = np.zeros(d)
    gtMP = np.zeros(d)
    avarMP = np.zeros(d)
    Gammahat = np.zeros([d, pmax])
    avarhat = np.zeros([d, pmax])
    gthat = np.zeros([d, T, pmax])
    alphahat = np.zeros([n, pmax])

    # ESTIMATION

    ##############################################################################
    ### TODO: follow each step to implement the function                       ###
    ##############################################################################

    rbar = np.mean(rt, axis=1).reshape(-1, 1)
    rtbar = rt - np.mean(rt, axis=1).reshape(-1, 1)
    gtbar = gt - np.mean(gt, axis=1).reshape(-1, 1)

    # Fama MacBeth Method 
    ####################
    # calculate beta FM and GammaFM
    betaFM = rtbar @ gtbar.T @ inv(gtbar @ gtbar.T)
    GammaFM = inv(betaFM.T @ betaFM) @ betaFM.T @ rbar
    ####################

    # FM Standard Errors             
    Gammas = inv(betaFM.T @ betaFM) @ betaFM.T @ rt
    ####################
    # calculate avarhatFM
    avarhatFM = var(Gammas, axis=1) / (T)
    ####################

    # Mimicking Portfolio approach
    erhat = np.zeros([d, n])
    avarMP = np.zeros(d)

    if n < T:

        erhat = gtbar @ rtbar.T @ inv(rtbar @ rtbar.T)  # (d,n)
        zrbar = gtbar - erhat @ rtbar  # (d,T)
        varr = (T ** (-1)) * rtbar @ rtbar.T  # (n,n)

        for ll in range(d):
            vareta = inv(varr) @ (rtbar @ diag(zrbar[ll, :].T) \
                                  @ diag(zrbar[ll, :].T) @ rtbar.T) @ inv(varr) / T
            avarMP[ll] = (T ** (-1)) * (rbar.T @ vareta @ rbar + erhat[ll, :] @ varr @ erhat[ll, :].T)

    ####################
    # Calculate gtMP
    gtMP = np.mean(erhat @ rt, axis=1)

    # Giglio Xiu Method

    # step 1: SVD 
    S, V = np.linalg.eig(rtbar.T @ rtbar / n / T)

    for phat in range(pmax):  # % at least one factor up to pmax

        ##############################################################################
        # (1) TODO: Calculate estimators of factors (vhat) and estimators of beta (betahat)#
    ##############################################################################
        for i in range(phat + 1):
            if i == 0:
                vhat = V[:, i]
            else:
                vhat = np.column_stack((vhat, V[:, i]))
        vhat = np.sqrt(T) * vhat.T

        if phat == 0:
            vhat = vhat.reshape((phat + 1, vhat.shape[0]))

        betahat = (1 / T) * rtbar @ vhat.T
        
#############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
        
        Sigmavhat = vhat @ vhat.T / T

        # step 2: FM with PCs
        
    ############################################################################## 
        # (2) TODO: Calculate Gammatilde
    ##############################################################################

        Gammatilde = inv(betahat.T @ betahat) @ betahat.T @ rbar

#############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
        ####################

        # step 3: TS Regression
    ############################################################################## 
        # (3) TODO: Calculate etahat and what
    ##############################################################################

        etahat = gtbar @ vhat.T @ inv(vhat @ vhat.T)
        what = gtbar - etahat @ vhat
       
    #############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################  
    

        # step 4: Combine TS and CS estimates
    ############################################################################## 
        # (4) TODO: Calculate Gammahat[:, phat]
    ##############################################################################

        Gammahat[:, phat] = (etahat @ Gammatilde).flatten()

    #############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################


        # step 5: Newy-West Estimation of Avar

        Pi11hat = np.zeros([d * (phat + 1), d * (phat + 1)])
        Pi12hat = np.zeros([d * (phat + 1), (phat + 1)])
        Pi22hat = np.zeros([phat + 1, phat + 1])

        for t in range(T):

            if Pi12hat.shape[1] == 1:
                Pi11hat = Pi11hat + vec(what[:, t].reshape(-1, 1) @ vhat[:, t].T).reshape(-1, 1) @ vec(
                    what[:, t].reshape(-1, 1) @ vhat[:, t].T).T.reshape(1, -1) / T
                Pi12hat = Pi12hat + (
                ((vec(what[:, t].reshape(-1, 1) @ vhat[:, t].T).reshape(-1, 1) @ vhat[:, t].T / T))).reshape(-1, 1)
                Pi22hat = Pi22hat + (vhat[:, t] @ vhat[:, t].T / T)
            else:
                Pi11hat = Pi11hat + (vec(what[:, t].reshape(-1, 1) @ [vhat[:, t]]).reshape(-1, 1) @ vec(
                    what[:, t].reshape(-1, 1) @ [vhat[:, t]]).T.reshape(1, -1) / T)
                Pi12hat = Pi12hat + ((vec(what[:, t].reshape(-1, 1) @ [vhat[:, t]]).reshape(-1, 1) @ [vhat[:, t]] / T))
                Pi22hat = Pi22hat + (vhat[:, t].reshape(-1, 1) @ [vhat[:, t]] / T)

            for s in range(1, int(min(t, q)) + 1):

                if Pi12hat.shape[1] == 1:
                    Pi11hat = Pi11hat + (1 / T * (1 - s / (q + 1)) * (
                                vec(what[:, t].reshape(-1, 1) @ vhat[:, t].T) @ vec(
                            what[:, t - s].reshape(-1, 1) @ vhat[:, t - s].T).T + vec(
                            what[:, t - s].reshape(-1, 1) @ vhat[:, t - s].T) @ vec(
                            what[:, t].reshape(-1, 1) @ vhat[:, t].T).T))
                    Pi12hat = Pi12hat + (1 / T * (1 - s / (q + 1)) * (
                                vec(what[:, t].reshape(-1, 1) @ vhat[:, t].T).reshape(-1, 1) @ vhat[:, t - s].T + vec(
                            what[:, t - s].reshape(-1, 1) @ vhat[:, t - s].T).reshape(-1, 1) @ vhat[:, t].T)).reshape(
                        -1, 1)
                    Pi22hat = Pi22hat + (1 / T * (1 - s / (q + 1)) * (
                                vhat[:, t] @ vhat[:, t - s].T + vhat[:, t - s] @ vhat[:, t].T))
                else:
                    Pi11hat = Pi11hat + 1 / T * (1 - s / (q + 1)) * (
                                (vec(what[:, t].reshape(-1, 1) @ [vhat[:, t].T]).reshape(-1, 1)) @ [
                            vec(what[:, t - s].reshape(-1, 1) @ [vhat[:, t - s].T])] + vec(
                            what[:, t - s].reshape(-1, 1) @ [vhat[:, t - s].T]).reshape(-1, 1) @ [
                                    vec(what[:, t].reshape(-1, 1) @ [vhat[:, t].T]).T])
                    Pi12hat = Pi12hat + (1 / T * (1 - s / (q + 1)) * (
                                vec(what[:, t].reshape(-1, 1) @ [vhat[:, t].T]).reshape(-1, 1) @ [
                            vhat[:, t - s].T] + vec(what[:, t - s].reshape(-1, 1) @ [vhat[:, t - s].T]).reshape(-1,
                                                                                                                1) @ [
                                    vhat[:, t].T]))
                    Pi22hat = Pi22hat + (1 / T * (1 - s / (q + 1)) * (
                                vhat[:, t].reshape(-1, 1) @ [vhat[:, t - s].T] + vhat[:, t - s].reshape(-1, 1) @ [
                            vhat[:, t].T]))
        alphahat[:, phat] = (rbar - betahat @ Gammatilde).flatten()

        temp1 = diag(kron(Gammatilde.T @ inv(Sigmavhat), diag(np.ones(d))) @ Pi11hat @ kron(inv(Sigmavhat) @ Gammatilde,
                                                                                            diag(np.ones(d))) / T
                     + kron(Gammatilde.T @ inv(Sigmavhat), diag(np.ones(d))) @ Pi12hat @ etahat.T / T
                     + (kron(Gammatilde.T @ inv(Sigmavhat), diag(np.ones(d))) @ Pi12hat @ etahat.T).T / T
                     + etahat @ Pi22hat @ etahat.T / T)
        + diag(var(alphahat[:, phat]) * etahat @ inv(
            betahat.T @ betahat / n - np.mean(betahat, axis=0).T @ np.mean(betahat, axis=0)) @ etahat.T) / n

        avarhat[:, phat] = diag(
            kron(Gammatilde.T @ inv(Sigmavhat), diag(np.ones(d))) @ Pi11hat @ kron(inv(Sigmavhat) @ Gammatilde,
                                                                                   diag(np.ones(d))) / T
            + kron(Gammatilde.T @ inv(Sigmavhat), diag(np.ones(d))) @ Pi12hat @ etahat.T / T + (
                        kron(Gammatilde.T @ inv(Sigmavhat), diag(np.ones(d))) @ Pi12hat @ etahat.T).T / T
            + etahat @ Pi22hat @ etahat.T / T)

        # step 7: recovery of gt
    ############################################################################## 
        # (5) TODO: calculate gthat[:, :, phat]
    ##############################################################################
 
        
        
     #############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
        ####################



    # step 8: choose number of latent factors

    phat = np.zeros([3, 1])
    ppmax = 20
    obj = S[:ppmax] + 0.25 * np.array(range(1, ppmax + 1)).reshape(-1, 1) * (log(n) + log(T)) * (
                n ** (-0.5) + T ** (-0.5)) * np.median(S[:ppmax])
    phat[0] = list(obj.flatten()).index(obj.min()) - 1
    obj2 = S[:ppmax] + 0.5 * np.array(range(1, ppmax + 1)).reshape(-1, 1) * (log(n) + log(T)) * (
                n ** (-0.5) + T ** (-0.5)) * np.median(S[:ppmax])
    phat[1] = list(obj2.flatten()).index(obj2.min()) - 1
    obj3 = S[:ppmax] + 0.75 * np.array(range(1, ppmax + 1)).reshape(-1, 1) * (log(n) + log(T)) * (
                n ** (-0.5) + T ** (-0.5)) * np.median(S[:ppmax])
    phat[2] = list(obj3.flatten()).index(obj3.min()) - 1

    # OUTPUT

    res = {}

    res['GammaFM'] = GammaFM
    res['avarhatFM'] = avarhatFM
    res['Gammahat'] = Gammahat
    res['avarhat'] = avarhat
    res['gtMP'] = gtMP
    res['avarMP'] = avarMP

    res['alphahat'] = alphahat
    res['gthat'] = gthat
    res['phat'] = phat
    res['vt'] = vhat  # Corresponds to the largest phat

    return res


def FM(rt, vt, gt, q):
    """
    Input

    rt    : n by T matrix
    gt    : d by T factor proxies
    vt    : p by T matrix
    q     : is used in Newy-West

    Output:

    res   : a dict that stores multiple variables

    """

    n, T = rt.shape
    d = gt.shape[0]
    p = vt.shape[0]

    Gammahat = np.zeros(d)

    rtbar = rt - np.mean(rt, axis=1).reshape(-1, 1)  # (n,T)
    vtbar = vt - np.mean(vt, axis=1).reshape(-1, 1)  # (p,T)
    gtbar = gt - np.mean(gt, axis=1).reshape(-1, 1)  # (d,T)

    # step 1: TS Regression
    betahat = rtbar @ vtbar.T @ inv(vtbar @ vtbar.T)  # (n,p)

    # step 2: FM                           
    Gammatilde = inv(betahat.T @ betahat) @ betahat.T @ np.mean(rt, axis=1).reshape(-1, 1)

    # step 3: Regression
    etahat = gtbar @ vtbar.T @ inv(vtbar @ vtbar.T)  # (d,p)
    gthat = etahat @ vtbar  # (d,T)
    what = gtbar - gthat  # (d,T)

    # step 4: Adjustment
    Gammahat = etahat @ Gammatilde  # (d,1)

    # step 5: Newy-West Estimation of Avar
    Pi11hat = np.zeros([d * p, d * p])
    Pi12hat = np.zeros([d * p, p])
    Pi22hat = np.zeros([p, p])

    for t in range(T):
        Pi11hat = Pi11hat + (vec(what[:, t].reshape(-1, 1) @ [vtbar[:, t]]) @ \
                             vec(what[:, t].reshape(-1, 1) @ [vtbar[:, t]]) / T)
        Pi12hat = Pi12hat + ((vec(what[:, t].reshape(-1, 1) @ [vtbar[:, t]]).reshape(-1, 1) @ [vtbar[:, t]] / T))
        Pi22hat = Pi22hat + (vtbar[:, t].reshape(-1, 1) @ [vtbar[:, t]] / T)

        for s in range(1, int(min(t, q)) + 1):
            Pi11hat = Pi11hat + 1 / T * (1 - s / (q + 1)) * (
                        (vec(what[:, t].reshape(-1, 1) @ [vtbar[:, t].T]).reshape(-1, 1)) @ [
                    vec(what[:, t - s].reshape(-1, 1) @ [vtbar[:, t - s].T])] + vec(
                    what[:, t - s].reshape(-1, 1) @ [vtbar[:, t - s].T]).reshape(-1, 1) @ [
                            vec(what[:, t].reshape(-1, 1) @ [vtbar[:, t].T]).T])
            Pi12hat = Pi12hat + (1 / T * (1 - s / (q + 1)) * (
                        vec(what[:, t].reshape(-1, 1) @ [vtbar[:, t].T]).reshape(-1, 1) @ [vtbar[:, t - s].T] + vec(
                    what[:, t - s].reshape(-1, 1) @ [vtbar[:, t - s].T]).reshape(-1, 1) @ [vtbar[:, t].T]))
            Pi22hat = Pi22hat + (1 / T * (1 - s / (q + 1)) * (
                        vtbar[:, t].reshape(-1, 1) @ [vtbar[:, t - s].T] + vtbar[:, t - s].reshape(-1, 1) @ [
                    vtbar[:, t].T]))

    avarhat = diag(
        kron(Gammatilde.T, diag(np.ones(d))) @ Pi11hat @ kron(Gammatilde, diag(np.ones(d))) / T + kron(Gammatilde.T,
                                                                                                       diag(np.ones(
                                                                                                           d))) @ Pi12hat @ etahat.T / T + (
                    kron(Gammatilde.T, diag(np.ones(d))) @ Pi12hat @ etahat.T).T / T + etahat @ Pi22hat @ etahat.T / T)

    res = {}

    res['betahat'] = betahat
    res['Gammahat'] = Gammahat
    res['avarhat'] = avarhat

    return res


def vec(a):
    """
    Input a should be a numpy array
    This function should flatten the input array in column-major order and return
    a copy of the input array, flattened to one dimension.
    """
    return a.flatten('F')


def divisorGenerator(n):
    """
    In this function, we will output al divisors of n, the output should be in the order
    of from min to max, and if n = d^2, we only output d once.
    This function should use 'yield' to help fasten speed and save storage space.
    HINT: you can first yield small divisor and store big divisor, and then use 'reversed'
    to output in order.
    """

    large_divisors = []
    for i in range(1, int(np.sqrt(n) + 1)):
        if n % i == 0:
            yield i
            if i * i != n:
                large_divisors.append(n / i)
    for divisor in reversed(large_divisors):
        yield divisor
