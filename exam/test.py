# load necessary packages

import pandas as pd
import numpy as np
from scipy.io import loadmat, savemat
import scipy.stats as stats
from scipy.linalg import fractional_matrix_power as mpower


from numpy import diag, sqrt, sort, argsort, log, isnan, var, std, nanmean, kron
from numpy.random import randn, choice, random
from numpy.linalg import norm, inv, lstsq, svd, eig
from numpy.matlib import repmat

import warnings
warnings.filterwarnings('ignore')

def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


np.random.seed(5)
n    = 40
T    = 50
d    = 10
rt   = np.random.randn(n,T)
gt   = np.random.randn(d,T)
pmax = 15
q    = 2


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

# (1) TODO: Calculate estimators of factors (vhat) and estimators of beta (betahat)#

for phat in range(pmax):  # % at least one factor up to pmax

    for i in range(phat + 1):
        if i == 0:
            vhat = V[:, i]
        else:
            vhat = np.column_stack((vhat, V[:, i]))
    vhat = np.sqrt(T) * vhat.T

    if phat == 1:
        vhat = vhat.reshape((phat, vhat.shape[0]))

    betahat = (1 / T) * rtbar @ vhat.T

    Sigmavhat = vhat @ vhat.T / T

    # (2) TODO: Calculate Gammatilde

    Gammatilde = np.zeros([phat + 1, 1])
    print(Gammatilde.shape)
    Gammatilde = inv(betahat.T @ betahat) @ betahat.T @ rbar
    print(phat, Gammatilde.shape)

print(rbar.shape)


