import numpy as np
import xarray as xr
import pandas as pd
from datetime import date, datetime, timedelta, timezone
import S2S.xarray_helpers    as xh
def Hindcast_seasonal(
    data_dir,
    variable,
    system,
    t_start,
    t_end,
    lat_bounds,
    lon_bounds,
    ):
    
    dates = pd.date_range(t_start, t_end, freq = 'MS')
    data = []
    for dd in dates:
            #d = dd.strftime('%Y-%m-%d')
            file = '%s/%s_%s_%s_%s%s'%(data_dir,variable,system,str(dd.year),str(dd.month),'.nc4')
    
    
            dataopen = xr.open_dataset(file)
            dataopen = dataopen.sel(g0_lat_2=slice(lat_bounds[1],lat_bounds[0]),g0_lon_3=slice(lon_bounds[0],lon_bounds[1]))
            #dataopen = dataopen.sel(step=slice('1 days','60 days'))
            dataopen = dataopen.rename({'g0_lat_2':'lat'})
            dataopen = dataopen.rename({'g0_lon_3':'lon'})
            dataopen = dataopen.rename({'TP_GDS0_SFC':'tp'})
            dataopen = dataopen.rename({'forecast_time1':'step'})
            dataopen = dataopen.rename({'ensemble0':'member'})
    
            time = np.empty((dataopen.step.shape[0]), dtype=datetime)
            time[:] = dd
            dataopen = dataopen.assign_coords(time=("time", time))
          
            data.append(dataopen)


    hindcast_data = xr.concat(data,dim='time')


    
    return hindcast_data

data_hindcast = Hindcast_seasonal(
    data_dir = '/nird/projects/NS9853K/DATA/SFE/Systems_daily_nc',
    variable = 'total_precipitation',
    system = 'ecmwf',
    t_start = '19930101',
    t_end = '19930201',
    lat_bounds = [0, 40],
    lon_bounds = [-10, 30],
    )

hindcast           = xh.assign_validation_time(data_hindcast)
