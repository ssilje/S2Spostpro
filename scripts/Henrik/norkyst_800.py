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
    # rs        = []

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
        # rs.append( target_radius )

    return np.array(appr_data)# ,np.array(rs)

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

def check_empty():

    path      = config['NORKYST']
    filenames = [
        path+'norkyst800_sst_'+date.strftime('%Y-%m-%d')+'_daily_mean.nc'\
        for date in pd.date_range(start='2012-01-01',end='2020-01-01',freq='D')
    ]

    for filename in filenames:

        try:

            out_filename = filename[:-3]+'_atBW'+'.nc'

            # print('Open dataset: ',out_filename.split('/')[-1])
            data = xr.open_dataset(out_filename)

            if len(data.sizes)==0:
                print('Empty: ',out_filename.split('/')[-1])
                print(data)
                os.remove(out_filename)

        except FileNotFoundError:
            print('FileNotFoundError: ',out_filename.split('/')[-1])

def assign_bw_coords():

    coords = BarentsWatch().load('all',no=0).isel(time=0).location
    coords = coords.drop('time')

    path      = config['NORKYST']
    filenames = glob.glob(path+'*_atBW.nc')

    for filename in filenames:
        data = xr.open_dataset(filename)
        data = data.assign_coords(location=coords.sel(location=data.location))

        o_filename = filename[:-3]+'c.nc'
        data.to_netcdf(o_filename)

def get_hardanger():

    path      = config['NORKYST']
    filenames = glob.glob(path+'*_atBWc.nc')
    extent    = [4.5,7.1,59.3,61]

    for filename in filenames:

        o_filename = filename.split('/')
        o_filename.insert(-1, 'hardanger')
        o_filename = '/'.join(o_filename)

        data = xr.open_dataset(filename)

        data = data.where(extent[0] < data.lon, drop=True)
        data = data.where(extent[2] < data.lat, drop=True)

        data = data.where(extent[1] > data.lon, drop=True)
        data = data.where(extent[3] > data.lat, drop=True)

        data.to_netcdf(o_filename)
        print(o_filename)

def stack_hardanger():

    path      = '/projects/NS9853K/DATA/norkyst800/processed/hardanger/'
    filenames = glob.glob(path+'*_atBWc.nc')

    data = xr.open_mfdataset(filenames,concat_dim='time')
    data.to_netcdf(path+'norkyst800_sst_daily_mean_hardanger_atBW.nc')


def to_bw2(download=False):

    path      = config['NORKYST']
    filenames = [
        path+'norkyst800_sst_'+date.strftime('%Y-%m-%d')+'_daily_mean.nc'\
        for date in pd.date_range(start='2012-01-01',end='2020-01-01',freq='D')
    ]

    # filenames = glob.glob(path+'norkyst800_sst_*daily_mean.nc')
    out_path  = path+'new/'


    if not os.path.exists(out_path):
        os.mkdir(out_path)

    # cluster = LocalCluster(
    #                             n_workers=4,
    #                             threads_per_worker=2,
    #                             processes=True,
    #                             lifetime='30 minutes',
    #                             lifetime_stagger='1 minute',
    #                             lifetime_restart=True
    #                         )
    #
    # client = Client(cluster)

    bw_obs = BarentsWatch().load('all',no=0).sortby('time').isel(time=0)
    lons   = bw_obs.lon
    lats   = bw_obs.lat

    del bw_obs

    land = xr.open_dataset(path+'norkyst800_landmask.nc')

    for filename in filenames:

        out_filename = filename[:-3]+'_atBW'+'.nc'

        if not os.path.exists(out_filename):

            try:

                print('Open dataset',filename.split('/')[-1])
                data = xr.open_dataset(filename)
                print(data)

                xr.Dataset().to_netcdf(out_filename)

                try:
                    print('Interpoolating to BarentsWatch locations')
                    xr.apply_ufunc(
                            norkyst_to_location,
                            data.sst,
                            data.lat,
                            data.lon,
                            land.mask,
                            lats,
                            lons,
                            input_core_dims=[
                                                ['X','Y'],['X','Y'],['X','Y'],['X','Y'],
                                                ['location'],['location'],
                                            ],
                            output_core_dims=[['location']],
                            vectorize=True,
                            dask='parallelized'
                        ).to_dataset(name='sst').to_netcdf(out_filename)

                    print(out_filename.split('/')[-1],' Stored')

                except ValueError:
                    print('Alignment error:')
                    print(data)

            except FileNotFoundError:
                print(filename.split('/')[-1],' Not Found')
                pass



    # cluster.close()
    # client.close()

    # while not os.path.exists(path+'norkyst800_sst_daily_mean.nc') or download:
    #
    #     print('Open datasets')
    #     data = xr.open_mfdataset(
    #             filenames,
    #             chunks={'time':10},
    #             parallel=True,
    #             concat_dim='time',
    #             join='inner'
    #         )
    #
    #     download = False
    #
    #     print(data)
    #
    #     print('Get landmask')
    #     land = xr.open_dataset(path+'norkyst800_landmask.nc')
    #
    #     print('Get BarentsWatch')
    #     bw_obs = BarentsWatch().load('all',no=0).sortby('time').isel(time=0)
    #     lons   = bw_obs.lon
    #     lats   = bw_obs.lat
    #
    #     del bw_obs
    #
    #     print('Interpoolating to BarentsWatch locations')
    #     out,r = xr.apply_ufunc(
    #             norkyst_to_location,
    #             data.sst,
    #             data.lat,
    #             data.lon,
    #             land.mask,
    #             lats,
    #             lons,
    #             input_core_dims=[
    #                                 ['X','Y'],['X','Y'],['X','Y'],['X','Y'],
    #                                 ['location'],['location'],
    #                             ],
    #             output_core_dims=[['location'],['location']],
    #             vectorize=True,
    #             dask='parallelized'
    #         )
    #
    #     out.to_dataset(name='sst').to_netcdf(path+'norkyst800_sst_daily_mean_interpolated.nc')

    # print('Open interpolated')
    # data = xr.open_dataset(
    #     path+'norkyst800_sst_daily_mean_interpolated.nc',
    #     chunks={'location':100}
    # )
    #
    # print(data)
    #
    # locations,datasets = zip(*data.groupby('location'))
    # out_fnames = [out_path+'NorKyst800_'+loc+'.nc' for loc in locations]
    # xr.save_mfdataset(datasets,out_fnames)

    ## for loc in out.location.values:
    ##     out.to_netcdf(out_path+'NorKyst800_'+str(loc)+'.nc')
    ##     print(loc,'stored')

    # cluster.close()
    # client.close()

    # out = out.to_dataset(name='sst').chunk({'time':-1,'location':1})
    # print('Storing by location')
    # # locations,datasets = zip(*out.groupby('location'))
    # # xr.save_mfdataset(datasets,out_paths)
    # for loc in out.location.values:
    #
    #     out.sel(location=loc).to_netcdf(path+'NorKyst800_'+str(loc)+'.nc')


    # ds = ds.sst.rolling(time=7,center=True).mean()

def stack(download=False):

    path      = config['NORKYST']
    filenames = [
        path+'norkyst800_sst_'+date.strftime('%Y-%m-%d')+'_daily_mean_atBW.nc'\
        for date in pd.date_range(start='2012-01-01',end='2020-01-01',freq='D')
    ]

    files = []
    for filename in filenames:

        try:

            print('Open dataset',filename.split('/')[-1])
            files.append(xr.open_dataset(filename))


        except FileNotFoundError:
            print(filename.split('/')[-1],' Not Found')
            pass

    data = xr.concat(files,'time')
    del files

    for loc in data.location.values:
        data.sel(location=loc).to_netcdf(path+'NorKyst800_'+str(loc)+'.nc')
        print('NorKyst800_'+str(loc)+'.nc Stored')
        # 13246
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

# def to_bw2(download=False):
#
#     path      = config['NORKYST']
#     filenames = glob.glob(path+'norkyst800_sst_*daily_mean*')
#     out_path  = path+'new/'
#
#     if not os.path.exists(out_path):
#         os.mkdir(out_path)
#
#     cluster = LocalCluster(
#                                 n_workers=2,
#                                 threads_per_worker=2,
#                                 processes=True,
#                                 lifetime='30 minutes',
#                                 lifetime_stagger='1 minute',
#                                 lifetime_restart=True
#                             )
#
#     client = Client(cluster)
#
#     while not os.path.exists(path+'norkyst800_sst_daily_mean.nc') or download:
#
#         print('Open datasets')
#         xr.open_mfdataset(
#                 filenames,
#                 chunks={'time':10},
#                 parallel=True,
#                 concat_dim='time',
#                 join='inner'
#             ).to_netcdf(path+'norkyst800_sst_daily_mean.nc')
#
#         download = False
#
#     data = xr.open_dataset(
#         path+'norkyst800_sst_daily_mean.nc',
#         chunks={'time':50}
#     )
#
#     print(data)
#
#     print('Get landmask')
#     land = xr.open_dataset(path+'norkyst800_landmask.nc')
#
#     print('Get BarentsWatch')
#     bw_obs = BarentsWatch().load('all',no=0).sortby('time').isel(time=0)
#     lons   = bw_obs.lon
#     lats   = bw_obs.lat
#
#     del bw_obs
#
#     while not os.path.exists(path+'norkyst800_sst_daily_mean_interpolated.nc') or download:
#
#         print('Interpoolating to BarentsWatch locations')
#         out,r = xr.apply_ufunc(
#                 norkyst_to_location,
#                 data.sst,
#                 data.lat,
#                 data.lon,
#                 land.mask,
#                 lats,
#                 lons,
#                 input_core_dims=[
#                                     ['X','Y'],['X','Y'],['X','Y'],['X','Y'],
#                                     ['location'],['location'],
#                                 ],
#                 output_core_dims=[['location'],['location']],
#                 vectorize=True,
#                 dask='parallelized'
#             )
#
#         out.to_dataset(name='sst').to_netcdf(path+'norkyst800_sst_daily_mean_interpolated.nc')
#
#     print('Open interpolated')
#     data = xr.open_dataset(
#         path+'norkyst800_sst_daily_mean_interpolated.nc',
#         chunks={'location':100}
#     )
#
#     print(data)
#
#     locations,datasets = zip(*data.groupby('location'))
#     out_fnames = [out_path+'NorKyst800_'+loc+'.nc' for loc in locations]
#     xr.save_mfdataset(datasets,out_fnames)
#
#     ## for loc in out.location.values:
#     ##     out.to_netcdf(out_path+'NorKyst800_'+str(loc)+'.nc')
#     ##     print(loc,'stored')
#
#     cluster.close()
#     client.close()
#
#     # out = out.to_dataset(name='sst').chunk({'time':-1,'location':1})
#     # print('Storing by location')
#     # # locations,datasets = zip(*out.groupby('location'))
#     # # xr.save_mfdataset(datasets,out_paths)
#     # for loc in out.location.values:
#     #
#     #     out.sel(location=loc).to_netcdf(path+'NorKyst800_'+str(loc)+'.nc')
#
#
#     # ds = ds.sst.rolling(time=7,center=True).mean()
