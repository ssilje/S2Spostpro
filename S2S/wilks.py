# this script contains functions for the computation of a correlations map
# noting statisitcally significant grid points by controlling the false discovery rate
# at the desired threshold of the user. Following Wilks 2016
# https://doi.org/10.1175/BAMS-D-15-00267.1

import numpy as np
import xarray as xr
from scipy import stats

def correlation_map(xmap,y):
    """
    @author: Henrik Auestad
    correlates y with every instance of xmap
    along the zeroth axis

    args
        xmap      : np.array(), dim: (n,t)
        y         : np.array(), dim: (1,t)
    returns
        cmap      : np.array(), dim: (n,1)
        pmap      : np.array(), dim: (n,1)
    """
    n = xmap.shape[0]
    cmap = np.empty((n,))
    pmap = np.empty((n,))

    for ii in range(n):
        cmap[ii],pmap[ii] = stats.pearsonr(xmap[ii,:],y.squeeze())

    return cmap,pmap

def p_FDR(pmap,alpha_FDR):
    """
    @author: Henrik Auestad
    Calculates the required p-value to controll
    the false discovery rate at alpha_FDR

    args
        pmap      : np.array(), dim: (n,...)
        alpha_FDR : float
    returns
        p_FDR     : float
    """
    pvals = pmap[np.isfinite(pmap)]
    n = pvals.shape[0]
    if n==0:
        return 0.
    else:
        pvals.sort()
        pval = pvals[pvals <= alpha_FDR*np.arange(1,n+1,1)/n]
        if len(pval)==0:
            return 0.
        else:
            return np.max(pvals[pvals <= alpha_FDR*np.arange(1,n+1,1)/n])

def significans_map(lon,lat,pmap,alpha_FDR):
    """
    @author: Henrik Auestad
    return a set of coordinates for where the correlation is
    statistically significant

    args
        pmap      : np.array(), dim: (n,)
        alpha_FDR : float
    returns
        lon       : np.array(), dim: (m,)
        lat       : np.array(), dim: (m,)
    """
    lon,lat = np.meshgrid(lon,lat)
    p = p_FDR(pmap,alpha_FDR)

    pmap = pmap.flatten()
    lon  = lon.flatten()[pmap<=p]
    lat  = lat.flatten()[pmap<=p]

    return lon,lat

def xsignificans_map(pmap,alpha_FDR):
    """
    @author: Henrik Auestad
    return a set of coordinates for where the correlation is
    statistically significant. Xarray version.
    """
    p = p_FDR(pmap.values,alpha_FDR)
    return pmap.where(pmap<=p)

# def ci2p(lower,est,higher):
#     """
#     @author: Henrik Auestad
#
#     https://www.bmj.com/content/343/bmj.d2304
#
#     Hardcoded to only apply to 95% confidence intervals, assumes normal dist
#     uses log scale for ratio calc.
#     """
#
#     est    = xr.ufuncs.log(est)
#     higher = xr.ufuncs.log(higher)
#     lower  = xr.ufuncs.log(lower)
#
#     z = est/( ( higher - lower )/( 2 * 1.96 ) )
