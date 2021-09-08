import netCDF4
import os
from S2S.data_handler import BarentsWatch

import xarray as xr
import pandas as pd
import numpy as np
import json
import glob

from S2S.local_configuration import config

from dask.distributed import LocalCluster, Client

def lin_weight(data,dist,r):
    """
    Linearly weighted mean where the closest points are given more weight.
    """
    weights = 1 - dist/r
    return ( data * weights ).sum() / weights.sum()

def distance(lat1, lon1, lat2, lon2):
    """
    Returns the great circle distance between lat1,lon1 and lat2,lon2.
    Function is positive definite.
    """
    p = np.pi/180
    a = 0.5 - np.cos((lat2-lat1)*p)/2 +\
        np.cos(lat1*p) * np.cos(lat2*p) * (1-np.cos((lon2-lon1)*p))/2
    return 12742 * np.arcsin(np.sqrt(a))

def norkyst_to_location(data,lat,lon,landmask,reflat,reflon):

    appr_data = []
    rs        = []

    landmask  = landmask.astype('int')

    for rlon,rlat in zip(reflon,reflat):

        dist = distance(rlat,rlon,lat,lon)

        over_land     = 0
        target_radius = 0.
        while not over_land:

            target_radius += 1.

            idx  = dist < target_radius

            lm   = landmask[idx]

            over_land = lm.sum()

        lm = lm.astype('bool')

        # wr: within radius
        data_wr = data[idx][lm]
        dist_wr = dist[idx][lm]

        appr_data.append( lin_weight( data_wr, dist_wr, target_radius ) )
        rs.append( target_radius )

    return np.array(appr_data),np.array(rs)

def norkyst_to_location_1D(data,lat,lon,landmask,rlat,rlon):

    landmask  = landmask.astype('int')

    dist = distance(rlat,rlon,lat,lon)

    over_land     = 0
    target_radius = 0.
    while not over_land:

        target_radius += 1.

        idx  = dist < target_radius

        lm   = landmask[idx]

        over_land = lm.sum()

    lm = lm.astype('bool')

    # wr: within radius
    data_wr = data[idx][lm]
    dist_wr = dist[idx][lm]

    return lin_weight( data_wr, dist_wr, target_radius ),target_radius

def download():

    nird_path = '/projects/NS9853K/DATA/norkyst800/processed/'
    if not os.path.exists(nird_path):
        os.mkdir(nird_path)

    url   = 'https://thredds.met.no/thredds/dodsC/sea/norkyst800mv0_24h/'
    times = pd.date_range(start='2012-06-27',end='2019-02-26',freq='D')

    filenames = [   url\
                    +'NorKyst-800m_ZDEPTHS_avg.an.'\
                    +time.strftime('%Y%m%d')\
                    +'12.nc'\
                    for time in times
                ]



    for filename,time in zip(filenames,times):
        nird_filename = nird_path\
                        +'norkyst800_sst_'\
                        +time.strftime('%Y-%m-%d')\
                        +'_daily_mean.nc'

        if not os.path.exists(nird_filename):
            try:
                with netCDF4.Dataset(filename) as file:

                    print(filename)

                    ds = xr.open_dataset(
                                            xr.backends.NetCDF4DataStore(file),
                                            decode_times=False
                                        )

                    ds = ds.temperature.sel(depth=0)
                    ds = ds.rename(
                                    {
                                        'longitude':'lon',
                                        'latitude':'lat'
                                    }
                                )
                    ds = ds.rename('sst')
                    ds = ds.assign_coords(time=[time])

                    ds.to_netcdf(nird_filename)

                    working_file = filename

            except OSError:
                print(
                        'OSError: File of time '\
                        +time.strftime('%Y-%m-%d')\
                        +' not found'
                    )
                pass
        else:
            pass

    try:
        working_file

    except NameError:
        for filename in filenames:
            try:
                with netCDF4.Dataset(filename) as file:
                    ds = xr.open_dataset(
                                            xr.backends.NetCDF4DataStore(file),
                                            decode_times=False
                                        )
                    working_file = filename
                    break

            except OSError:
                pass

    with netCDF4.Dataset(working_file) as file:

        print('Get landmask')

        nird_filename = nird_path + 'norkyst800_landmask.nc'

        ds = xr.open_dataset(
                                xr.backends.NetCDF4DataStore(file),
                                decode_times=False
                            )

        ds = ds.mask.rename(
                        {
                            'longitude':'lon',
                            'latitude':'lat'
                        }
                    )

        ds.to_netcdf(nird_filename)


def download_new():

    nird_path = '/projects/NS9853K/DATA/norkyst800/new/'
    if not os.path.exists(nird_path):
        os.mkdir(nird_path)

    url   = 'https://thredds/fileServer/fou-hi/norkyst800m/'
    times = pd.date_range(start='2017-02-28',end='2021-09-07',freq='D')

    filenames = [   url\
                    +'NorKyst-800m_ZDEPTHS_avg.fc.'\
                    +time.strftime('%Y%m%d')\
                    +'00.nc'\
                    for time in times
                ]

    for filename,time in zip(filenames,times):

        nird_filename = nird_path\
                        +'norkyst800_sst_'\
                        +time.strftime('%Y-%m-%d')\
                        +'_daily_mean.nc'

        if not os.path.exists(nird_filename):

            try:
                with netCDF4.Dataset(filename) as file:

                    print(filename)

                    ds = xr.open_dataset(
                                            xr.backends.NetCDF4DataStore(file),
                                            decode_times=False
                                        )
                    print(ds)
                    exit()
                    ds = ds.temperature.sel(depth=0)
                    ds = ds.rename(
                                    {
                                        'longitude':'lon',
                                        'latitude':'lat'
                                    }
                                )
                    ds = ds.rename('sst')
                    ds = ds.assign_coords(time=[time])

                    ds.to_netcdf(
                                    nird_path\
                                    +'norkyst800_sst_'\
                                    +time.strftime('%Y-%m-%d')\
                                    +'_daily_mean.nc'
                                )
            except KeyError:
                print('OSError')
                pass

def to_bw2():

    path      = config['NORKYST']
    filenames = glob.glob(path+'norkyst800_sst_2017-03-0*daily_mean*')
    out_path  = path+'new/'

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    cluster = LocalCluster(
                                n_workers=4,
                                threads_per_worker=4,
                                processes=True,
                                lifetime='30 minutes',
                                lifetime_stagger='5 minute',
                                lifetime_restart=True
                            )

    client = Client(cluster)

    bw_obs = BarentsWatch().load('all',no=0).sortby('time').isel(time=0)
    lons   = bw_obs.lon
    lats   = bw_obs.lat

    del bw_obs

    print('Open datasets')
    data   = xr.open_mfdataset(
                                filenames,
                                chunks={'time':1},
                                parallel=True,
                                concat_dim='time',
                                join='outer'
                            )

    print('Get landmask')
    land = xr.open_dataset(path+'norkyst800_landmask.nc')

    print('Interpoolating to BarentsWatch locations')
    out,r = xr.apply_ufunc(
            norkyst_to_location,
            data.sst,
            data.lat,
            data.lon,
            land.mask,
            lats.chunk({'location':5}),
            lons.chunk({'location':5}),
            input_core_dims=[
                                ['X','Y'],['X','Y'],['X','Y'],['X','Y'],
                                ['location'],['location'],
                            ],
            output_core_dims=[['location'],['location']],
            vectorize=True,
            dask='parallelized'
        )

    print('Re-chunk')
    out = out.to_dataset(name='sst').chunk({'location':1,'time':-1})
    for loc in out.location.values:
        out.to_netcdf(out_path+'NorKyst800_'+str(loc)+'.nc')
        print(loc,'stored')

    cluster.close()
    client.close()

    # out = out.to_dataset(name='sst').chunk({'time':-1,'location':1})
    # print('Storing by location')
    # # locations,datasets = zip(*out.groupby('location'))
    # # xr.save_mfdataset(datasets,out_paths)
    # for loc in out.location.values:
    #
    #     out.sel(location=loc).to_netcdf(path+'NorKyst800_'+str(loc)+'.nc')


    # ds = ds.sst.rolling(time=7,center=True).mean()

def to_bw():

    path      = config['NORKYST']
    filenames = glob.glob(path+'norkyst800_sst_*daily_mean*')
    out_path  = path+'new/'

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    cluster = LocalCluster(
                                n_workers=8,
                                threads_per_worker=2,
                                processes=True,
                                lifetime='30 minutes',
                                lifetime_stagger='1 minute',
                                lifetime_restart=True
                            )

    client = Client(cluster)

    bw_obs = BarentsWatch().load('all',no=0).sortby('time').isel(time=0)
    lons   = bw_obs.lon
    lats   = bw_obs.lat

    del bw_obs

    print('Open datasets')
    data   = xr.open_mfdataset(
                                filenames,
                                chunks={'time':1},
                                parallel=True,
                                concat_dim='time',
                                join='outer'
                            )

    print('Get landmask')
    land = xr.open_dataset(path+'norkyst800_landmask.nc')

    print('Interpoolating to BarentsWatch locations')
    for loc in lons.location.values:


        out,r = xr.apply_ufunc(
                norkyst_to_location_1D,
                data.sst,
                data.lat,
                data.lon,
                land.mask,
                lats.sel(location=loc),
                lons.sel(location=loc),
                input_core_dims=[
                                    ['X','Y'],['X','Y'],['X','Y'],['X','Y'],
                                    [],[],
                                ],
                output_core_dims=[[],[]],
                vectorize=True,
                dask='parallelized'
            )

        out = out.to_dataset(name='sst')\
                .expand_dims('location').assign_coords(location=[loc])
        out.to_netcdf(out_path+'NorKyst800_'+str(loc)+'.nc')
        print(loc,'stored')

    cluster.close()
    client.close()
