import datetime
import pandas as pd
from datetime import datetime
import numpy as np
import xarray as xr




def persistence(init_value,observations,window=30):
    """
    @author: Henrik Auestad
    
    Fits a persistence forecast to the observations at initalization time (init_value)
    predicting observations using cross validation.
    
    Input must be anomlies.
    
    args:
        init_value:     xarray.DataArray with time dimension, must be broadcasted to fit observations.
        observations:   xarray.DataArray with time dimension
        window:         default 30, number of consequtive days used for fitting
        
    returns:
       fitted model:   xarray.DataArray
    """

    print('\tmodels.persistence()')
    ds = xr.merge(
                    [
                        init_value.rename('iv'),
                        observations.rename('o')
                ],join='inner',compat='override'
            )

    ds  = xh.unstack_time(ds)

    rho = xr.apply_ufunc(
                correlation_CV,ds.iv,ds.o,ds.dayofyear,window,
                input_core_dims  = [
                                    ['year','dayofyear'],
                                    ['year','dayofyear'],
                                    ['dayofyear'],
                                    []
                                ],
                output_core_dims = [['year','dayofyear']],
                vectorize=True
    )

    rho = xh.stack_time(rho)

    try:
        rho = rho.drop('validation_time')
    except (AttributeError, ValueError):
        pass

    return rho * init_value

def correlation_CV(x,y,index,window=30):
    """
    @author: Henrik Auestad
    
    Computes correlation of x against y (Pearson), keeping dim -1 and -2.
    Dim -1 must be 'dayofyear', with the corresponding days given in index.
    Dim -2 must be 'year'.
    args:
        x:      np.array of float, with day of year as index -1 and year as
                index -2
        y:      like x
        index:  np.array of int, 1-dimensional holding dayofyear corresponding
                to dim -1 of x
    returns
        rho:   np.array of float, with day of year as index -1 and year as
                dim -2
    dimensions requirements:
        name            dim
        year            -2
        dayofyear       -1
    Returns 2 dimensional array (year,dayofyear)
    """
    rho   = []

    pad   = window//2

    if len(x.shape)==2:
        x     = np.pad(x,pad,mode='wrap')[pad:-pad,:]
        y     = np.pad(y,pad,mode='wrap')[pad:-pad,:]

    if len(x.shape)==3:
        x     = np.pad(x,pad,mode='wrap')[pad:-pad,pad:-pad,:]
        y     = np.pad(y,pad,mode='wrap')[pad:-pad,pad:-pad,:]

    index = np.pad(index,pad,mode='wrap')

    index[-pad:] += index[-pad-1]
    index[:pad]  -= index[-pad-1]

    for ii,idx in enumerate(index[pad:-pad]):

        # pool all values that falls within window
        xpool = x[...,np.abs(index-idx)<=pad]
        ypool = y[...,np.abs(index-idx)<=pad]

        yrho = []
        for yy in range(xpool.shape[-2]):

            # delete the relevant year from pool (for cross validation)
            filtered_xpool = np.delete(xpool,yy,axis=-2).flatten()
            filtered_ypool = np.delete(ypool,yy,axis=-2).flatten()

            idx_bool = np.logical_and(
                                np.isfinite(filtered_xpool),
                                np.isfinite(filtered_ypool)
                            )
            if idx_bool.sum()<2:
                r = np.nan

            else:
                r,p = stats.pearsonr(
                                    filtered_xpool[idx_bool],
                                    filtered_ypool[idx_bool]
                                )

            yrho.append(r)

        rho.append(np.array(yrho))

    return np.stack(rho,axis=-1)


def at_validation(obs,vt,ddays=1):
    """
    args:
        obs:   xarray.Dataset with time dimension
        vt:    xarray.DataArray time + step dimension
        ddays: int, tolerance, in days, for matching times
    returns
        observations: xarray.Dataset with time and step dimension
    """
    print('\t\txarray_helpers.at_validation()')

    obs  = obs.sortby('time')
    vt   = vt.sortby(['time','step'])
    vt   = vt.sel(time=slice(obs.time.min(),obs.time.max()))
    time = vt.time
    step = vt.step

    out  = []
    for t in time:
        o = obs.reindex(
                indexers   = {'time':(t+step).data},
                method     = 'nearest',
                tolerance  = pd.Timedelta(ddays,'D'),
                fill_value = np.nan
                )
        o = o.assign_coords(time=o.time-t).rename(time='step')
        out.append(o)

    out = xr.concat(out,time).sortby(['time','step'])
    #return out.assign_coords(validation_time=out.time+out.step)
    return assign_validation_time(out)
   
def assign_validation_time(ds):
    """
    @author: Henrik Auestad
    
    Add validation_time coordinates to xarray.Dataset/DataArray with 'time' and
    'step' dimensions.
    validation_time = time + step
    time: datetime-like
    step: pd.Timedelta
    args:
        ds: xarray.Dataset/DataArray with 'time' and 'step' dimensions
    returns:
        ds: xarray.Dataset/DataArray with 'validation_time' dimension
    """
    return ds.assign_coords(validation_time=ds.time+ds.step)


def o_climatology(da,window=30):
    """
    @Author: Henrik Auestad
    
    Climatology with centered initialization time, using cross validation
    using a 30-day window.
    args:
        da: xarray.DataArray with 'time' dimension
    returns:
        mean: xarray.DataArray, like da
        std: xarray.DataArray, like da
    time: datetime-like
    """

    print('\t\txarray_helpers.o_climatology()')

    da = unstack_time(da)

    # to all year,dayofyear matrices in da, apply runnning_clim_CV
    mean,std = xr.apply_ufunc(
            running_clim_CV, da, da.dayofyear,window,
            input_core_dims  = [['year','dayofyear'],['dayofyear'],[]],
            output_core_dims = [['year','dayofyear'],['year','dayofyear']],
            vectorize=True
        )
    
    ###### Question #####
    # do we need running_clim_CV in forecast mode? I tried to use running_clim, but got some dimesion errors. 
    
    #mean,std = xr.apply_ufunc(
    #        running_clim, da, da.dayofyear,window,
    #        input_core_dims  = [['year','dayofyear'],['dayofyear'],[]],
    #        output_core_dims = [['year','dayofyear'],['year','dayofyear']],
    #        vectorize=True
    #    )    

    
    # re-assing time dimension to da from year,dayofyear
    return stack_time(mean),stack_time(std)

def unstack_time(da):
    """
    @author: Henrik Auestad
    
    Splits time dimension in da into a 'year' and a 'dayofyear' dimension.
    Coordinate time must be datetime-like.
    args:
        da: xarray.DataArray, requires dims: time
    returns:
        da: xarray.DataArray, new dimensions: year, dayofyear
    """
    da   = da.sortby('time')
    time = da.time

    # create an mulitindex mapping time -> (year,dayofyear)
    stacked_time = pd.MultiIndex.from_arrays(
                                            [
                                                da.time.dt.year.to_pandas(),
                                                da.time.dt.dayofyear.to_pandas()
                                            ],names=('year','dayofyear')
                                        )

    # re-assing time from datetime like to multiindex (year,dayofyear) and
    # and split mutliindex into year and dauofyear dimension
    return da.assign_coords(time=stacked_time).unstack('time')


def running_clim(x,index,window=30):
    """
    Computes mean and standard deviation over x keeping dim -1. Dim -1 must be
    'dayofyear', with the corresponding days given in index.
    args:
        x:      np.array of float, with day of year as index -1
        index:  np.array of int, 1-dimensional holding dayofyear corresponding
                to dim -1 of x
    returns
        mean:   np.array of float, with day of year as index -1 and year as
                dim -2
        std:   np.array of float, with day of year as index -1 and year as
                dim -2
    dimensions requirements:
        name            dim
        dayofyear      -1
        member          0
    """
    mean  = []
    std   = []

    pad   = window//2

    x     = np.pad(x,pad,mode='wrap')[pad:-pad,pad:-pad,:]
    x     = np.pad(x,pad,mode='wrap')[pad:-pad,:]
    index = np.pad(index,pad,mode='wrap')

    index[-pad:] += index[-pad-1]
    index[:pad]  -= index[-pad-1]

    for ii,idx in enumerate(index[pad:-pad]):

        # pool all values that falls within window
        pool = x[...,np.abs(index-idx)<=pad]

        if np.isfinite(pool).sum() > 1:
            mean.append(np.full_like(pool[0][...,0],np.nanmean(pool)))
            std.append(np.full_like(pool[0][...,0],np.nanstd(pool)))
        else:
            mean.append(np.full_like(pool[0][...,0],np.nan))
            std.append(np.full_like(pool[0][...,0],np.nan))

    return np.stack(mean,axis=-1),np.stack(std,axis=-1)

def running_clim_CV(x,index,window=30):
    """
    Like running_clim() but leaves out current year in computation for
    cross validation
    Computes mean and standard deviation over x keeping dim -1 and -2.
    Dim -1 must be 'dayofyear', with the corresponding days given in index.
    Dim -2 must be 'year'.
    args:
        x:      np.array of float, with day of year as index -1 and year as
                index -2
        index:  np.array of int, 1-dimensional holding dayofyear corresponding
                to dim -1 of x
    returns
        mean:   np.array of float, with day of year as index -1 and year as
                dim -2
        std:   np.array of float, with day of year as index -1 and year as
                dim -2
    dimensions requirements:
        name            dim
        year            -2
        dayofyear       -1
    """
    mean  = []
    std   = []

    pad   = window//2

    x     = np.pad(x,pad,mode='wrap')[pad:-pad,:]
    index = np.pad(index,pad,mode='wrap')

    index[-pad:] += index[-pad-1]
    index[:pad]  -= index[-pad-1]

    for ii,idx in enumerate(index[pad:-pad]):

        # pool all values that falls within window
        pool = x[...,np.abs(index-idx)<=pad]

        ymean,ystd = [],[]
        for yy in range(pool.shape[-2]):

            # delete the relevant year from pool (for cross validation)
            filtered_pool = np.delete(pool,yy,axis=-2)

            if np.isfinite(filtered_pool).sum() > 1:
                ymean.append(np.nanmean(filtered_pool))
                ystd.append(np.nanstd(filtered_pool))
            else:
                ymean.append(np.nan)
                ystd.append(np.nan)

        mean.append(np.array(ymean))
        std.append(np.array(ystd))

    return np.stack(mean,axis=-1),np.stack(std,axis=-1)

def stack_time(da):
    """
    Stacks 'year' and 'dayofyear' dimensions in a xarray.DataArray to a 'time'
    dimension.
    args:
        da: xarray.DataArray, requires dims: year and dayofyear
    returns:
        da: xarray.DataArray, new dimension: time
    """

    da = da.stack(time=('year','dayofyear'))

    time = []
    for year,dayofyear in zip(da.year.values,da.dayofyear.values):
        year       = pd.to_datetime(year, format='%Y')
        dayofyear  = pd.Timedelta(dayofyear-1,'D')
        time.append(year+dayofyear)

    da = da.assign_coords(time=time)

    return da



#######################################################
##### ##### ##### Program starts ##### ##### #####

steps    = pd.to_timedelta([7,14,21,28,35],'D')
ds_step = xr.Dataset({"step": steps})


# make a dummy teperature dataset with daily values the past 10 years. The last time is the time it should be made forecast from. 

dates = pd.date_range("20100101", "20200101", freq="D")
ds_date = xr.Dataset({"time": dates})

s = np.random.uniform(0,10,dates.size) # values between 0 and 10

d = {'sst'     : s,
        'time'  : dates}

df = pd.DataFrame(d)
df.index = df["time"]
del df["time"]

data = df.to_xarray()

# Calculating running means. Question: the timeseries is still with daily output. Should it be with 7 days interval?

data_rolling=data.rolling(time=7,center=True).mean()


print('\tAssign step dimension to observations')

## make data with step dimension

data_step = at_validation(
    data_rolling,
    ds_date.time + ds_step.step,
    ddays=1)


data_step = data_step.drop('validation_time') # Needed to do? 

print('\tCompute climatology')

data_step_mean,data_step_std = o_climatology(data_step)

# the unstack_time and stack_time is changing the order (and length of time). 
# So need to do this on the data_step to get the same order as the data_step_mean and data_step_std

data_step= unstack_time(data_step)
data_step = stack_time(data_step)
data_step_a = ( data_step - data_step_mean ) / data_step_std

# make a dummy teperature dataset with daily values the pnext 46 days. 
# The last time from the obeerved temperature is the time it should be made forecast from. 

dates_fc = pd.date_range(dates[-1].strftime("%m%d%y"), periods=46, freq="D")
ds_fc_date = xr.Dataset({"time": dates_fc})
s = np.empty(dates_fc.shape)
s[:] = np.NaN

# Question: what climatology to use when calculation the anomalies. This is the inital time to be made forecast from

# set the inital temperature anomaly 

s[0] = (data.sst.sel(time=dates[-1]).values - data_step_mean.sst.sel(time='2020-12-31',step='7 days').values) / data_step_std.sst.sel(time='2020-12-31',step='7 days').values

d = {'sst'     : s,
        'time'  : dates_fc}

df = pd.DataFrame(d)    

ds_fc = xr.Dataset({"time": dates_fc})
df.index = df["time"]
del df["time"]
data_fc = df.to_xarray()


# make the forecast with step dimension. This is removing the inital time value..? how to fix this?

data_fc_step = at_validation(
    data_fc,
    ds_fc_date.time + ds_step.step,
    ddays=1)
data_fc_step = data_fc_step.drop('validation_time')


# Not working now. The persistence model must be changed to be in forecast mode

pers  = persistence(
  init_value   = data_fc_step.sst,
  observations = data_step_a.sst
)


