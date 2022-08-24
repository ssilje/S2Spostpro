import numpy as np
import pandas as pd
import xarray as xr
import xskillscore as xs
import properscoring as ps
from scipy import stats

import S2S.xarray_helpers as xh

def centered_acc(x,y):
    """
    @author: Henrik Auestad
    Anomaly correlation after Wilks (2011, Chapter 8) - Eq. 8.

    D.S. Wilks, Chapter 8 - Forecast Verification,
    International Geophysics, Academic Press, Volume 100, 2011,
    Pages 301-394, https://doi.org/10.1016/B978-0-12-385022-5.00008-7.
    """
    idx_bool = ~np.logical_or(np.isnan(x),np.isnan(y))

    x = x[idx_bool]
    y = y[idx_bool]

    M = len(x)

    x_anom = (x - x.mean())
    y_anom = (y - y.mean())

    covar = x_anom * y_anom / M

    var_x = x_anom**2 / M
    var_y = y_anom**2 / M

    return covar.sum() / np.sqrt( ( var_x.sum() + var_y.sum() ) )

def uncentered_acc(x,y):
    """
    @author: Henrik Auestad
    Anomaly correlation after Wilks (2011, Chapter 8) - Eq. 8.64

    D.S. Wilks, Chapter 8 - Forecast Verification,
    International Geophysics, Academic Press, Volume 100, 2011,
    Pages 301-394, https://doi.org/10.1016/B978-0-12-385022-5.00008-7.
    """
    idx_bool = ~np.logical_or(np.isnan(x),np.isnan(y))

    x = x[idx_bool]
    y = y[idx_bool]

    M = len(x)

    x_anom = x
    y_anom = y

    covar = x_anom * y_anom / M

    var_x = x_anom**2 / M
    var_y = y_anom**2 / M

    return covar.sum() / np.sqrt( ( var_x.sum() + var_y.sum() ) )

def ACc(forecast,observations,weights=None,centered=True):
    """
    @author: Henrik Auestad
    Anomaly correlation after Wilks (2011, Chapter 8)

    D.S. Wilks, Chapter 8 - Forecast Verification,
    International Geophysics, Academic Press, Volume 100, 2011,
    Pages 301-394, https://doi.org/10.1016/B978-0-12-385022-5.00008-7.
    """
    if weights is None:
        weights = xr.full_like(observations,1.)

    forecast     = forecast * weights
    observations = observations * weights

    try:
        forecast = forecast.mean('member')
    except AttributeError:
        pass


    ds = xr.merge(
                    [
                        forecast.rename('fc'),
                        observations.rename('obs')
                    ],join='inner',compat='override'
                )

    if centered:
        ufunc = centered_acc
    else:
        ufunc = uncentered_acc

    r = xr.apply_ufunc(
            ufunc,ds.fc,ds.obs,
            input_core_dims = [['lat','lon'],['lat','lon']],
            output_core_dims = [[]],
            vectorize=True,dask='parallelized'
        )

    return r



def CRPS_ensemble(obs,fc,fair=True,axis=0):
    """
    @author: Ole Wulff
    @date: 2020-07-08

    implementation of fair (adjusted) CRPS based on equation (6) from Leutbecher (2018, QJRMS, https://doi.org/10.1002/qj.3387)
    version with fair=False tested against properscoring implementation crps_ensemble (see https://pypi.org/project/properscoring/)

    INPUT:
        obs: observations as n-dimensional array
        fc: forecast ensemble as (n+1)-dimensional where the extra dimension (axis) carries the ensemble members
        fair: if True returns the fair version of the CRPS accounting for the limited ensemble size (see Leutbecher, 2018)
              if False returns the normal CRPS
        axis: axis of fc array that contains the ensemble members, defaults to 0
    OUTPUT:
        CRPS: n-dimensional array
    TODO:
        implement weights for ensemble member weighting
    """
    odims = obs.shape
    M = fc.shape[axis]
    if axis != 0:
        fc = np.swapaxes(fc,axis,0)

    # flatten all dimensions except for the ensemble member dimension:
    fc_flat = fc.reshape([M,-1])
    obs_flat = obs.reshape([-1])

    dsum = np.array([abs(fc_flat[jj] - fc_flat[kk]) for kk in range(M) for jj in range(M)]).sum(axis=axis)
    if fair:
        CRPS = 1/M * (abs(fc_flat - obs_flat)).sum(axis=axis) - 1/(2*M*(M-1)) * dsum
    else:
        CRPS = 1/M * (abs(fc_flat - obs_flat)).sum(axis=axis) - 1/(2*M**2) * dsum

    # is this necessary or even a good idea at all?
#     del dsum, fc_flat, obs_flat

    return CRPS.reshape([*odims])

def crps_ensemble(obs,fc,fair=True):
    """
    @author: Henrik Auestad
    A xarray wrapper for CRPS_ensemble()
    """

    # obs = obs.broadcast_like(fc.mean('member'))
    obs,fc = xr.align(obs,fc,join='outer')

    return xr.apply_ufunc(
            CRPS_ensemble, obs, fc, fair,
            input_core_dims  = [['time'],['member','time'],[]],
            output_core_dims = [['time']],
            vectorize=True
        )

def fair_brier_score(observations,forecasts):
    """
    @author: Henrik Auestad (copied from xskilllscore)
    """
    M = forecasts['member'].size
    e = (forecasts == 1).sum('member')
    o = observations
    return (e / M - o) ** 2 - e * (M - e) / (M ** 2 * (M - 1))

def annual_bootstrap_CRPS(fc,obs,N):
    """
    @author: Henrik Auestad
    """
    y,d,m = fc.shape

    # generate random integers as indices for fc-obs pairs
    _idx_f = np.random.randint(low=0,high=y,size=(N,y))
    _idx_o = np.random.randint(low=0,high=y,size=(N,y))

    b_fc = np.transpose(fc[_idx_f,:,:],(3,0,1,2))

    # hardcode to crps
    score_fc = np.nanmean(CRPS_ensemble(obs[_idx_o,:],b_fc),(-2))
    score_cl = np.nanmean(ps.crps_gaussian(obs[_idx_o,:],mu=0,sig=1),(-2))
    print('.\n..\n...')
    return 1 - ( np.nanmean(score_fc,-1) / np.nanmean(score_cl,-1) )

def annual_bootstrap(observations,forecast,score_func,N=10000):
    """
    @author: Henrik Auestad
    """
    data  = xr.merge(
        [
            forecast.rename('fc'),
            observations.rename('obs')
        ],
    join='inner',
    compat='equals'
    )

    # split into weeks
    ds   = xh.unstack_time(data)

    if score_func=='CRPS':
        # get bootstrapped null-distribution
        return xr.apply_ufunc(
                annual_bootstrap_CRPS, ds.fc, ds.obs, N,
                input_core_dims  = [
                                    ['year','dayofyear','member'],
                                    ['year','dayofyear'],
                                    []
                                ],
                output_core_dims = [['sample']],
                vectorize = True, dask='parallelized'
            )
    else:
        return None

def ttest_1samp(a, popmean, dim):
    """
    Downloaded from:
    https://gist.github.com/kuchaale/293d2a16726a5d492be4f5bbae8d9111#file-xa_ttest-py
    at 3/10/21

    This is a two-sided test for the null hypothesis that the expected value
    (mean) of a sample of independent observations `a` is equal to the given
    population mean, `popmean`

    Inspired here: https://github.com/scipy/scipy/blob/v0.19.0/scipy/stats/stats.py#L3769-L3846

    Parameters
    ----------
    a : xarray
        sample observation
    popmean : float or array_like
        expected value in null hypothesis, if array_like than it must have the
        same shape as `a` excluding the axis dimension
    dim : string
        dimension along which to compute test

    Returns
    -------
    mean : xarray
        averaged sample along which dimension t-test was computed
    pvalue : xarray
        two-tailed p-value
    """
    n = a[dim].shape[0]
    df = n - 1
    a_mean = a.mean(dim)
    d = a_mean - popmean
    v = a.var(dim, ddof=1)
    denom = xrf.sqrt(v / float(n))

    t = d /denom
    prob = stats.distributions.t.sf(xr.ufuncs.fabs(t), df) * 2
    prob_xa = xr.DataArray(prob, coords=a_mean.coords)
    return a_mean, prob_xa

def _ttest_rel_(a,b,alternative='two-sided'):

    # dealing with nan, keeping only pairs
    idx = np.logical_and( np.isfinite(a),np.isfinite(b) )
    if idx.sum()>0:
        t,p = stats.ttest_rel(a[idx],b[idx],alternative=alternative)
    else:
        p = np.nan
    return p

def ttest_paired(a, b, dim, alternative='two-sided'):
    """
    @author: Henrik Auestad
    xarray wrapper for scipy.stats.ttest_rel

    ****from scipy.stats:

    Calculate the t-test on TWO RELATED samples of scores, a and b.

    This is a test for the null hypothesis that 2 related or repeated
    samples have identical average (expected) values.
    ****

    Convetion: b - a, i.e. alternative='greater' null hypothesis states
    that pop. mean of a is greater or equal to pop. mean of b

    Parameters
    ----------
    a : xarray.DataArray
        sample observations
    b : xarray.DataArray
        sample observations
    dim : list of strings
        dimension(s) along which to compute test

    ****from scipy.stats:

    alternative : {‘two-sided’, ‘less’, ‘greater’}, optional

    Defines the alternative hypothesis.
    The following options are available (default is ‘two-sided’):

        ‘two-sided’

        ‘less’: one-sided

        ‘greater’: one-sided
    ****

    Returns
    -------
    dist : xarray
        distance between sample means (b - a)
    pvalue : xarray
        two-tailed p-value
    """

    dist = b.mean(dim,skipna=True) - a.mean(dim,skipna=True)
    pvalue = xr.apply_ufunc(
        _ttest_rel_,
        a, b,
        input_core_dims  = [dim,dim],
        output_core_dims = [[]],
        vectorize = True,
        dask      = 'parallelized',
        kwargs    = {
            'alternative':alternative
        }
    )

    return dist, pvalue

def _ttest_ind_(a,b,alternative='two-sided',equal_var=True):

    t,p = stats.ttest_ind(
        a[np.isfinite(a)],
        b[np.isfinite(b)],
        alternative=alternative,
        equal_var=equal_var
    )

    return p

def ttest_upaired(a, b, dim, alternative='two-sided', welch=False):
    """
    @author: Henrik Auestad
    xarray wrapper for scipy.stats.ttest_ind

    ****from scipy.stats:

    Calculate the T-test for the means of two independent samples of scores.

    This is a two-sided test for the null hypothesis that 2 independent
    samples have identical average (expected) values. This test assumes
    that the populations have identical variances by default.
    ****

    Convetion: b - a, i.e. alternative='greater' null hypothesis states
    that pop. mean of a is greater or equal to pop. mean of b

    Parameters
    ----------
    a : xarray.DataArray
        sample observations
    b : xarray.DataArray
        sample observations
    dim : list of strings
        dimension(s) along which to compute test

    ****from scipy.stats:

    alternative : {‘two-sided’, ‘less’, ‘greater’}, optional

    Defines the alternative hypothesis.
    The following options are available (default is ‘two-sided’):

        ‘two-sided’

        ‘less’: one-sided

        ‘greater’: one-sided

    welch : bool, optional

    If True (default), perform a standard independent 2 sample test that
    assumes equal population variances [1]. If False, perform Welch’s t-test,
    which does not assume equal population variance [2].
    ****

    Returns
    -------
    dist : xarray
        distance between sample means (b - a)
    pvalue : xarray
        two-tailed p-value
    """

    dist = b.mean(dim,skipna=True) - a.mean(dim,skipna=True)
    pvalue = xr.apply_ufunc(
        _ttest_ind_,
        a, b,
        input_core_dims  = [dim,dim],
        output_core_dims = [[]],
        vectorize = True,
        dask      = 'parallelized',
        kwargs    = {
            'alternative':alternative,
            'equal_var':not welch
        }
    )

    return dist, pvalue

class SSCORE:
    """
    @author: Henrik Auestad
    Calculate monthly and seasonal means of scores with bootstrapped CIs.
    """

    def __init__(self,**kwargs):

        self.forecast     = self.try_key(kwargs,'forecast')
        self.observations = self.try_key(kwargs,'observations')

        self.data  = xr.merge(
                            [
                                self.forecast.rename('fc'),
                                self.observations.rename('obs')
                            ],
                        join='inner',
                        compat='equals'
                        )

    @staticmethod
    def skill_score(x,y):
        return 1 - np.nanmean(x,axis=-1)/np.nanmean(y,axis=-1)

    @staticmethod
    def try_key(dictionary,key):
        try:
            return dictionary[key]
        except KeyError:
            return None

    def bootstrap(self,N=1000,ci=.95,min_period=2):

        # split into weeks
        ds = xh.unstack_time(self.data)

        low_q,est,high_q,ny = xr.apply_ufunc(
                self.pull, ds.fc, ds.obs, N, ci, min_period,
                input_core_dims  = [
                                    ['dayofyear','year'],
                                    ['dayofyear','year'],
                                    [],
                                    [],
                                    []
                                ],
                output_core_dims = [[],[],[],[]],
                vectorize=True
            )
        out = xr.merge(
                    [
                        low_q.rename('low_q'),
                        est.rename('est'),
                        high_q.rename('high_q')
                    ], join='inner', compat='equals'
                )
        out = out.assign_coords(number_of_years=ny)
        return out

    def pull(self,fc,obs,N,ci=.95,min_period=2):

        # use only years of at least min_period observations
        fc_idx  = (~np.isnan(fc)).sum(axis=0)>min_period
        obs_idx = (~np.isnan(obs)).sum(axis=0)>min_period

        # use only coincinding years
        idx_bool = np.logical_and(
                            fc_idx,
                            obs_idx
                            )

        ny = idx_bool.sum()

        if idx_bool.sum()>1:

            fc = np.nanmean(fc[...,idx_bool],axis=0)
            obs = np.nanmean(obs[...,idx_bool],axis=0)

            y  = fc.shape[-1]

            # generate random integers as indices for fc-obs pairs
            _idx_ = np.random.randint(low=0,high=y,size=(N,y))

            # pick y random fc-obs pairs N times
            _fc_   = fc[_idx_]
            _obs_  = obs[_idx_]

            # calculate score N times
            score = np.sort(self.skill_score(_fc_,_obs_))

            # quantiles
            alpha  = (1-ci)/2

            high_q = score[ int( (N-1) * (1-alpha) ) ]
            low_q  = score[ int( (N-1) * alpha ) ]

        else:

            high_q = np.nan
            low_q  = np.nan

        if idx_bool.sum()>0:
            # actual score
            est_score = self.skill_score(fc,obs)
        else:
            est_score = np.nan

        return low_q,est_score,high_q,ny
