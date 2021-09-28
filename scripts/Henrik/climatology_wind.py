import xarray as xr
import pandas as pd
import numpy as np
import os
import xskillscore as xs
import glob
import dask
import seaborn as sns

import json
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from S2S.local_configuration import config
from S2S.graphics      import latex, graphics

from matplotlib.colors import BoundaryNorm

from S2S.data_handler  import ERA5, BarentsWatch, Archive
from S2S.process       import Hindcast, Observations, Grid2Point
from S2S               import models
import S2S.scoring     as sc
import S2S.xarray_helpers as xh

from dask.distributed import LocalCluster, Client

def compute_climatology():

    mparser = {
                '1':'JAN','2':'FEB','3':'MAR',
                '4':'APR','5':'MAY','6':'JUN',
                '7':'JUL','8':'AUG','9':'SEP',
                '10':'OCT','11':'NOV','12':'DEC',
                'DJF':'DJF','JJA':'JJA',
                'MAM':'MAM','SON':'SON'
            }

    parser = {
        'longitude':'lon',
        'latitude' :'lat'
        }

    database_path = '/projects/NS9853K/DATA/SFE/ERA_daily_nc/'
    out_filename  = '_era_05deg_6hourly_1990-2020.nc'
    path          = '/projects/NS9853K/DATA/processed/wind_S2S/wind10m/NA/'
    abs_filename  = path+'abs_era_1deg_6hourly_1990-2020.nc'

    cluster = LocalCluster(
                                n_workers=8,
                                threads_per_worker=2,
                                processes=True,
                                lifetime='30 minutes',
                                lifetime_stagger='1 minute',
                                lifetime_restart=True
                            )

    client = Client(cluster)

    # for component in ['u','v']:
    #     fnames = []
    #     for year in np.arange(1990,2020):
    #         fnames.extend(
    #             glob.glob(database_path+'*'+component+'*wind_'+str(year)+'*')
    #         )
    #
    #     clim_t_start  = (1990,1,1)
    #     clim_t_end    = (2020,1,1)
    #     bounds1  = (-15,28,35,73)
    #     bounds2  = (345,360,35,73)
    #
    #     lons = slice(-30,30)
    #     lats = slice(35,75)
    #
    #     out_path = path+component+out_filename
    #
    #     data = xr.open_mfdataset(
    #                             fnames,
    #                             chunks={'time':200},
    #                             parallel=True,
    #                             concat_dim='time',
    #                             join='inner'
    #                         ).rename(parser)
    #
    #     data = data.assign_coords(
    #         lon=(((data.lon + 180) % 360) - 180)
    #     ).sortby('lon').sortby('lat')
    #
    #     print(data)
    #
    #     with dask.config.set(**{'array.slicing.split_large_chunks': True}):
    #         out = data.sel(lon=lons,lat=lats)
    #     print(out)
    #     out.to_netcdf(out_path)
    #
    #     print(out_path)
    #
    # print('open u and v')
    # u = xr.open_dataset(path+'u'+out_filename,chunks={'time':200})
    # v = xr.open_dataset(path+'v'+out_filename,chunks={'time':200})
    #
    # xr.apply_ufunc(
    #                 xh.absolute,u.u10.rename('U10'),v.v10.rename('U10'),
    #                 input_core_dims  = [[],[]],
    #                 output_core_dims = [[]],
    #                 dask='parallelized'
    #             ).to_netcdf(abs_filename)

    print('open data')
    abs = xr.open_dataset(abs_filename)

    print('max day')
    max_day = abs.groupby('time.date').max()

    del abs

    print('asssing coords')
    max_day = max_day.assign_coords(
        date=[pd.Timestamp(date) for date in max_day.date.values]
    ).rename({'date':'time'})

    dim     = 'time.season'
    dim_lab = dim.split('.')[-1]

    print('plot')
    latex.set_style(style='white')
    subplots = graphics.fg(max_day,dim)
    fig,axes = plt.subplots(
        subplots[0],
        subplots[1],
        figsize=latex.set_size(
            width='thesis',
            subplots=(subplots[0],subplots[1])
        ),
        subplot_kw=dict(projection=ccrs.PlateCarree())
    )

    axes = axes.flatten()

    print('groupby')
    # z    = max_day.groupby(dim).max(skipna=True).U10
    z    = max_day.U10.groupby(dim).quantile(0.5)

    cmap   = sns.color_palette("Spectral", as_cmap=True).reversed()
    levels = np.arange(1,27,2)
    norm   = BoundaryNorm(levels,cmap.N)

    for ax,lab in zip(axes,z[dim_lab].values):

        cs = ax.contourf(
            z.lon,
            z.lat,
            z.sel({dim_lab:lab}),
            levels=levels,
            norm=norm,
            cmap=cmap,
            transform=ccrs.PlateCarree()
        )

        ax.coastlines(resolution='10m', color='black',\
                            linewidth=0.2)

        ax.set_title(mparser[str(lab)])

    cbar = fig.colorbar(cs,ax=axes.ravel().tolist(),boundaries=levels)
    cbar.ax.set_title('[m/s]')
    fig.suptitle('Median of daily max wind in ERA5 1990-2020')
    graphics.save_fig(fig,'wind_climatology/max_day_'+dim_lab+'_5q')

    ############### mean day ############################

    print('open data')
    abs = xr.open_dataset(abs_filename)

    print('mean day')
    max_day = abs.groupby('time.date').mean()

    del abs

    print('asssing coords')
    max_day = max_day.assign_coords(
        date=[pd.Timestamp(date) for date in max_day.date.values]
    ).rename({'date':'time'})

    dim     = 'time.season'
    dim_lab = dim.split('.')[-1]

    print('plot')
    latex.set_style(style='white')
    subplots = graphics.fg(max_day,dim)
    fig,axes = plt.subplots(
        subplots[0],
        subplots[1],
        figsize=latex.set_size(
            width='thesis',
            subplots=(subplots[0],subplots[1])
        ),
        subplot_kw=dict(projection=ccrs.PlateCarree())
    )

    axes = axes.flatten()

    print('groupby')
    # z    = max_day.groupby(dim).max(skipna=True).U10
    z    = max_day.U10.groupby(dim).quantile(0.5)

    cmap   = sns.color_palette("Spectral", as_cmap=True).reversed()
    levels = np.arange(1,27,2)
    norm   = BoundaryNorm(levels,cmap.N)

    for ax,lab in zip(axes,z[dim_lab].values):

        cs = ax.contourf(
            z.lon,
            z.lat,
            z.sel({dim_lab:lab}),
            levels=levels,
            norm=norm,
            cmap=cmap,
            transform=ccrs.PlateCarree()
        )

        ax.coastlines(resolution='10m', color='black',\
                            linewidth=0.2)

        ax.set_title(mparser[str(lab)])

    cbar = fig.colorbar(cs,ax=axes.ravel().tolist(),boundaries=levels)
    cbar.ax.set_title('[m/s]')
    fig.suptitle('Median of daily mean wind in ERA5 1990-2020')
    graphics.save_fig(fig,'wind_climatology/mean_day_'+dim_lab+'_5q')

    client.close()
    cluster.close()
