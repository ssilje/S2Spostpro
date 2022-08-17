import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm
from S2S.data_handler import BarentsWatch
import S2S.xarray_helpers    as xh
import os
from S2S.graphics import latex
from S2S.data_handler import ERA5

from S2S.xarray_helpers import o_climatology


def get_era():

    bounds   = (3.25,7.75,58.3,62.)
    var      = 'sst'

    clim_t_start  = (2000,6,15)
    clim_t_end    = (2020,8,25)

    print('Get era')
    era = ERA5(high_res=True).load(
        var         = var,
        start_time  = clim_t_start,
        end_time    = clim_t_end,
        bounds      = bounds
    )

    print('ERA: resample and roll')
    era = era.resample(time='D').mean()
    era = era.sst.rolling(time=7,center=True).mean()
    return era - 273.15

def get_norkyst():

    path = '/projects/NS9853K/DATA/norkyst800/processed/hardanger/norkyst800_sst_daily_mean_hardanger_atBW.nc'

    with xr.open_dataarray(path) as data:
        data = data.squeeze()
    return data

def plot():

    # Saltkjelen I, Ljonesbjørgene

    tmp_path = '/projects/NS9853K/DATA/norkyst800/processed/'

    if True:

        bw = BarentsWatch().load(location=['Ljonesbjørgene','Aga Ø'],no=380,data_label='DATA').sst
        bw = bw.assign_coords(loc_name=('location',['Ljonesbjørgene','Aga Ø']))
        nk = get_norkyst()

        bw = bw.rolling(time=7,center=True).mean()
        nk = nk.rolling(time=7,center=True).mean()

        era = get_era().dropna('time',how='all')

        era = era.sel(lat=60.0,lon=[5.,5.5],method='nearest').sortby('lon')
        era = era.assign_coords(loc_name=('lon',['Aga Ø','Ljonesbjørgene']))

        era_mean,era_std = o_climatology(era)
        nk_mean,nk_std = o_climatology(nk)
        bw_mean,bw_std = o_climatology(bw)

        era = ( era - era_mean )/era_std
        nk  = ( nk- nk_mean )/nk_std
        bw  = ( bw - bw_mean )/bw_std

        era.to_netcdf(tmp_path+'era_fig2_tmp.nc')
        nk.to_netcdf(tmp_path+'nk_fig2_tmp.nc')
        bw.to_netcdf(tmp_path+'bw_fig2_tmp.nc')

    else:

        era = xr.open_dataarray(tmp_path+'era_fig2_tmp.nc')
        nk  = xr.open_dataarray(tmp_path+'nk_fig2_tmp.nc')
        bw  = xr.open_dataarray(tmp_path+'bw_fig2_tmp.nc')


    #era = era.interp(method='nearest')

    #     mean_bw,std_bw = xh.o_climatology(bw)
    #     mean_nk,std_nk = xh.o_climatology(nk)
    #
    #     # mean_bw = mean_bw.groupby('time').mean()
    #     # std_bw = std_bw.groupby('time').mean()
    #
    #     mean_nk = mean_nk.groupby('time').mean()
    #     std_nk = std_nk.groupby('time').mean()
    #
    #     bw_a = ( bw - mean_bw ) / std_bw
    #     nk_a = ( nk - mean_nk ) / std_nk
    #
    #     data = xr.merge(
    #         [bw_a.rename('BarentsWatch'),nk_a.rename('NorKyst')],
    #         join='inner'
    #     )
    #
    #     data.to_netcdf('./fig2_temp.nc')
    # else:
    #     with xr.open_dataset('./fig2_temp.nc') as data:
    #         data = data

    # cmap   = plt.get_cmap('Set2')
    # levels = np.arange(0.5,14,2)
    # norm   = BoundaryNorm(levels,cmap.N)

    latex.set_style(style='white')
    fig,axes = plt.subplots(2,1,\
        figsize=latex.set_size(width=345,subplots=(2,1),fraction=0.95)
    )

    for ax,loc,name in zip(axes,bw.location,bw.loc_name.values):

        # c_p = {'DJF':'blue','MAM':'yellow','JJA':'orange','SON':'green'}

        bw_  = bw.sel(location=loc)
        nk_  = nk.sel(location=loc)
        era_ = era.where(era.loc_name==name,drop=True)
        # c = [c_p[x] for x in d.time.dt.season.values]
        # c = [x for x in d.time.dt.month.values]

        # cs = ax.plot(
        #     d.BarentsWatch,
        #     d.NorKyst,
        #     c=c,
        #     s=0.9,
        #     cmap=cmap,
        #     norm=norm
        # )
        ms = 0.5
        cs = ax.plot(
            bw_.time,
            bw_,
            'o',
            c='blue',
            markersize=ms,
            label='BarentsWatch',
            zorder=20,
        )
        cs = ax.plot(
            nk_.time,
            nk_,
            'o',
            c='k',
            markersize=ms,
            label='NorKyst',
            zorder=10,
        )
        cs = ax.plot(
            era_.time,
            era_,
            'o',
            c='grey',
            markersize=ms,
            label='ERA5',
            zorder=5,
        )

        if name=='Ljonesbjørgene':
            ax.legend(fontsize=7)
        ax.set_xlabel('Time',fontsize=8)
        ax.set_ylabel('[$^\circ$C]',fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.tick_params(axis='both', which='minor', labelsize=7)
        ax.set_title(name,fontsize=10)

    # cbar=fig.colorbar(
    #     cs,
    #     ax=axes.ravel().tolist(),
    #     fraction=0.046,
    #     pad=0.06,
    #     orientation='vertical'
    # )
    #
    # cbar.set_ticks(np.arange(-0.5,13.5,2))
    # # cbar.set_ticklabels(['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])
    # cbar.set_ticklabels(['JAN-FEB','MAR-APR','MAY-JUN','JUL-AUG','SEP-OCT','NOV-DEC'])

    latex.save_figure(fig,'/nird/home/heau/figures_paper/fig2')
