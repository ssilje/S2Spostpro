import xarray as xr
import time
import pandas as pd
import numpy as np
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar
import glob
from S2S.local_configuration import config

def run():
    path         = config['NORKYST']
    arb_filename = 'norkyst800_sst_*_daily_mean_at-BW-loc.nc'

    # n_workers = 8
    # n_threads = 2
    # processes = True
    #
    # cluster = LocalCluster(
    #                         n_workers=n_workers,
    #                         threads_per_worker=n_threads,
    #                         processes=processes,
    #                         lifetime='1 hour',
    #                         lifetime_stagger='10 minutes',
    #                         lifetime_restart=True
    #                     )
    #
    # client = Client(cluster)

    # data   = xr.open_mfdataset(
    #                             path+arb_filename
    #                             # chunks={'location':20},
    #                             # parallel=True
    #                         )

    d = []
    for filename in glob.glob(path+arb_filename):
        time = filename[-34:-24]
        d.append(xr.open_dataset(filename).assign_coords(time=[pd.Timestamp(time)]))
    ds = xr.concat(d,'time')
    print('done')
    ds['sst'] = ds['__xarray_dataarray_variable__']
    ds = ds.drop(['__xarray_dataarray_variable__'])
    data = ds.sst
    data = data.sortby('time').drop('radius')

    # for loc in data.location.values:
    #     data.sel(location=loc).to_netcdf(path+'NorKyst800_'+str(loc)+'.nc')
    #     print(loc)
    #
    # cluster.close()
    # client.close()

    return data

#
#     # groups = list(data.groupby('time.month'))
#     #
#     # for n,g in groups:
#     #     print(n,np.isfinite(g.values).sum().compute())
#     print(data)

    # for loc in data.location.values:
    #     print(loc)
    #     data.sel(location=loc).to_netcdf(path+'NorKyst800_'+str(loc)+'.nc')

    # print(data)
    #
    # cluster.close()
    # client.close()

# NorKyst-800m_ZDEPTHS_avg.an.2013052612.nc

# def test():
#
#     path         = config['NORKYST']
#     arb_filename = 'norkyst800_sst_*_daily_mean_at-BW.nc'
#
#     n_workers = 8
#     n_threads = 2
#     processes = True
#
#     cluster = LocalCluster(
#                             n_workers=n_workers,
#                             threads_per_worker=n_threads,
#                             processes=processes,
#                             lifetime='1 hour',
#                             lifetime_stagger='10 minutes',
#                             lifetime_restart=True
#                         )
#
#     client = Client(cluster)
#
#     data   = xr.open_mfdataset(
#                                 path+arb_filename,
#                                 chunks={'location':9},
#                                 parallel=True
#                             )
#
#     # groups = list(data.groupby('time.month'))
#     #
#     # for n,g in groups:
#     #     print(n,np.isfinite(g.values).sum().compute())
#     print(data)

    # for loc in data.location.values:
    #     print(loc)
    #     data.sel(location=loc).to_netcdf(path+'NorKyst800_'+str(loc)+'.nc')

    # print(data)
    #
    # cluster.close()
    # client.close()
