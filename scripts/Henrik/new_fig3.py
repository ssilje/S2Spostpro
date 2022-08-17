import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from glob import glob
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm, TwoSlopeNorm
from S2S.data_handler import BarentsWatch
import os
from S2S.graphics import latex

from S2S.xarray_helpers import o_climatology
from scripts.Henrik.distance_from_coast import get_shore
import cmocean

def get_norkyst():

    path = '/projects/NS9853K/DATA/norkyst800/processed/hardanger/norkyst800_sst_daily_mean_hardanger_atBW.nc'

    with xr.open_dataarray(path) as data:
        data = data.squeeze()
        data = data.rolling(time=7,center=True).mean()
        data = data.where(data.time.dt.weekday==2,drop=True)
        mean,std = o_climatology(data,window=30,cross_validation=False)

    return data,mean,std

def get_era(location):
    extent = [3.25,7.75,58.3,62.]
    path   = '/projects/NS9853K/DATA/SFE/ERA_daily_nc/sea_surface_temperature_*.nc'
    p      = glob(path)


    if not os.path.exists('/projects/NS9853K/DATA/norkyst800/processed/era3tmp.nc'):
        with xr.open_mfdataset(
            path,
            combine='nested',
            concat_dim='time'
        ) as data:
            data = data.sst
            data = data.where(data.time.dt.year >= 2012,drop=True)
            data = data.sortby('time')
            data = data.resample(time='D').mean()
            data = data.rolling(time=7,center=True).mean()
            data = data.where(data.time.dt.weekday==2,drop=True)
            data = data.rename({'latitude':'lat','longitude':'lon'})
            data = data.where(data.lat>=extent[2],drop=True)
            data = data.where(data.lat<=extent[3],drop=True)
            data = data.where(data.lon>=extent[0],drop=True)
            data = data.where(data.lon<=extent[1],drop=True)
            data = data - 273.15
            data = data.interpolate_na(dim='lon',fill_value='extrapolate')
            data = data.interp(dict(lon=location.lon,lat=location.lat))
            data.to_netcdf('/projects/NS9853K/DATA/norkyst800/processed/era3tmp.nc')
    else:

        with xr.open_dataarray('/projects/NS9853K/DATA/norkyst800/processed/era3tmp.nc') as data:

            data = data

    mean,std = o_climatology(data,window=30,cross_validation=False)

    return data,mean,std

def get_barentswatch():

    extent = [4.25,6.75,59.3,61.]
    data = BarentsWatch().load(location='all',no=0,data_label='DATA')

    data = data.where(data.lat>=extent[2],drop=True)
    data = data.where(data.lat<=extent[3],drop=True)
    data = data.where(data.lon>=extent[0],drop=True)
    data = data.where(data.lon<=extent[1],drop=True)
    data = data.sst

    mean,std = o_climatology(data,window=30,cross_validation=False)

    print(data)


    return data,mean,std

def main():

    bw  = True
    nk  = False
    er  = False

    adjust = False

    tmp_path = '/projects/NS9853K/DATA/norkyst800/processed/'

    if bw:
        print('barentswatch')
        barentswatch,barentswatch_mean,barentswatch_std = get_barentswatch()
        #barentswatch.to_netcdf(tmp_path+'bw_fig3_tmp_350.nc')
        #barentswatch_std.to_netcdf(tmp_path+'bw_std_fig3_tmp_350.nc')
        #barentswatch_mean.to_netcdf(tmp_path+'bw_mean_fig3_tmp_350.nc')

        #barentswatch_a = ( barentswatch - barentswatch_mean )/barentswatch_std
        #barentswatch_a.to_netcdf(tmp_path+'bw_fig3_tmp.nc')
        #barentswatch_std.to_netcdf(tmp_path+'bw_std_fig3_tmp.nc')

    else:

        #barentswatch_a = xr.open_dataarray(tmp_path+'bw_fig3_tmp.nc')
        #barentswatch_std = xr.open_dataarray(tmp_path+'bw_std_fig3_tmp.nc')

        barentswatch = xr.open_dataarray(tmp_path+'bw_fig3_tmp.nc')
        barentswatch_std = xr.open_dataarray(tmp_path+'bw_std_fig3_tmp.nc')
        barentswatch_mean = xr.open_dataarray(tmp_path+'bw_mean_fig3_tmp.nc')

    if nk:
        print('norkyst')
        norkyst,norkyst_mean,norkyst_std = get_norkyst()
        norkyst.to_netcdf(tmp_path+'nk_fig3_tmp.nc')
        norkyst_std.to_netcdf(tmp_path+'nk_std_fig3_tmp.nc')
        norkyst_mean.to_netcdf(tmp_path+'nk_mean_fig3_tmp.nc')
        #norkyst_a = ( norkyst - norkyst_mean )/norkyst_std
        #norkyst_a.to_netcdf(tmp_path+'nk_fig3_tmp.nc')

    else:
        norkyst = xr.open_dataarray(tmp_path+'nk_fig3_tmp.nc')
        norkyst_std = xr.open_dataarray(tmp_path+'nk_std_fig3_tmp.nc')
        norkyst_mean = xr.open_dataarray(tmp_path+'nk_mean_fig3_tmp.nc')

    if er:
        print('era')
        era,era_mean,era_std = get_era(barentswatch.location)
        era.to_netcdf(tmp_path+'era_fig3_tmp.nc')
        era_mean.to_netcdf(tmp_path+'era_mean_fig3_tmp.nc')
        era_std.to_netcdf(tmp_path+'era_std_fig3_tmp.nc')
        #era_a = ( era - era_mean )/era_std
        #era_a.to_netcdf(tmp_path+'era_a_fig3_tmp.nc')

    else:
        era      = xr.open_dataarray(tmp_path+'era_fig3_tmp.nc')
        era_mean = xr.open_dataarray(tmp_path+'era_mean_fig3_tmp.nc')
        era_std  = xr.open_dataarray(tmp_path+'era_std_fig3_tmp.nc')
        #era_a = xr.open_dataarray(tmp_path+'era_a_fig3_tmp.nc')


    # create anomalies from climate
    norkyst_a      = norkyst.groupby('time.dayofyear')      - norkyst_mean
    era_a          = era.groupby('time.dayofyear')          - era_mean
    barentswatch_a = barentswatch.groupby('time.dayofyear') - barentswatch_mean

    norkyst_a      = norkyst_a.groupby('time.dayofyear')/norkyst_std
    era_a          = era_a.groupby('time.dayofyear')/era_std
    barentswatch_a = barentswatch_a.groupby('time.dayofyear')/barentswatch_std

    if adjust:
        norkyst_a      = norkyst_a - norkyst_a.mean('time')
        era_a          = era_a - era_a.mean('time')
        barentswatch_a = barentswatch_a - barentswatch_a.mean('time')

    norkyst_a,barentswatch_a = xr.align(norkyst_a,barentswatch_a)
    era_a,barentswatch_a     = xr.align(era_a,barentswatch_a)

    # choose only entries where all three data sources are defined
    #finite = np.logical_and(np.isfinite(norkyst_a.values),np.isfinite(barentswatch_a),np.isfinite(era_a))

    bw = barentswatch_a.where(np.isfinite(barentswatch_a),drop=True)
    e5 = era_a.where(np.isfinite(barentswatch_a),drop=True)
    nk = norkyst_a.where(np.isfinite(barentswatch_a),drop=True)

    # acc of era and barentswatch
    var_b  = (bw**2).sum('time')
    var_e  = (e5**2).sum('time')
    cov_eb = (e5 * bw).sum('time')

    acc_eb = cov_eb / (np.sqrt( var_e ) * np.sqrt( var_b ))

    # acc of norkyst and barentswatch
    var_b  = (bw**2).sum('time')
    var_n  = (nk**2).sum('time')
    cov_nb = (nk * bw).sum('time')

    acc_nb = cov_nb / (np.sqrt( var_n ) * np.sqrt( var_b ))

    # MAKE TABLE, SET TO FALSE TO MAKE FIGURE
    if False:

        shore = get_shore()
        shore = np.floor( shore.shore * 10 ) / 10

        shore,acc_eb = xr.align(shore,acc_eb)
        acc_eb = acc_eb.assign_coords(shore=('location',shore.values))
        acc_eb = acc_eb.groupby('shore').mean()

        shore,acc_nb = xr.align(shore,acc_nb)
        acc_nb = acc_nb.assign_coords(shore=('location',shore.values))
        acc_nb = acc_nb.groupby('shore').mean()

        out = {'distance_from_coast_metric':[],'ACC_norkyst_barentswatch':[],'ACC_norkyst_era5':[]}
        for s in np.unique(shore.values):

            out['distance_from_coast_metric'].append(s)
            out['ACC_norkyst_barentswatch'].append(acc_nb.sel(shore=s).values)
            out['ACC_norkyst_era5'].append(acc_eb.sel(shore=s).values)

        pd.DataFrame.from_dict(out).to_csv('/nird/home/heau/figures_paper/acc_table.csv')

        exit()

    # figure settings
    cmap   = cmocean.cm.speed
    #levels = np.sort(np.hstack([np.arange(0.05,1.,0.15),-np.arange(0.05,1.,0.15)]))
    levels = np.arange(0,1.1,0.1)
    norm   = BoundaryNorm(levels,cmap.N,extend='both')

    # plot
    extent = [4.25,6.75,59.3,61.]
    coast = cfeature.NaturalEarthFeature(category='physical', scale='10m',
                                facecolor='none', name='coastline')

    latex.set_style(style='white')
    fig,axes = plt.subplots(1,2,\
        figsize=(4.535076795350768, 2.8028316011177257),\
        subplot_kw=dict(projection=ccrs.NorthPolarStereo())
    )

    ax = axes[0]

    rmse = acc_eb
    rmse = rmse.where(np.isfinite(rmse),drop=True)
    print('ERA5')
    print(rmse.values.mean())
    print(rmse.values.std())
    print('')

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    feature = ax.add_feature(coast, edgecolor='k', linewidth=0.5, zorder=40)

    cs = ax.scatter(
        rmse.lon.values,
        rmse.lat.values,
        c=rmse.values,
        s=4.6,
        norm=norm,
        cmap=cmap,
        #marker='s',
        transform=ccrs.PlateCarree(),
        zorder=55,
        edgecolors='black',
        linewidth=0.3,
    )

    ax.set_title('ACC BarentsWatch ERA5')#, mean ACC: '+str(round(rmse.values.mean(),2)))

    ###

    ax = axes[1]

    rmse = acc_nb
    rmse = rmse.where(np.isfinite(rmse),drop=True)
    print('Norkyst')
    print(rmse.values.mean())
    print(rmse.values.std())
    print('')

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    feature = ax.add_feature(coast, edgecolor='k', linewidth=0.5, zorder=40)

    cs = ax.scatter(
        rmse.lon.values,
        rmse.lat.values,
        c=rmse.values,
        s=4.6,
        norm=norm,
        cmap=cmap,
        #marker='s',
        transform=ccrs.PlateCarree(),
        zorder=55,
        edgecolors='black',
        linewidth=0.3,
    )

    cbar=fig.colorbar(
        cs,
        ax=axes.ravel().tolist(),
        #fraction=0.026,
        shrink=0.8,
        aspect=25,
        orientation='horizontal'
    )
    #levels=np.sort(np.hstack([np.arange(0.05,1.,0.3),-np.arange(0.05,1.,0.3)]))
    #cbar.set_ticks(levels)
    #cbar.set_ticklabels([round(lvl,2) for lvl in levels])
    cbar.ax.tick_params(labelsize=6)
    ax.set_title('ACC BarentsWatch Norkyst')#, mean ACC: '+str(round(rmse.values.mean(),2)))

    #fig.suptitle('ACC of Barentswatch and',fontsize=10)
    latex.save_figure(fig,'/nird/home/heau/figures_paper/Figure4')



    if False:

        bw = bw.isel(location=[20,21,22]).transpose('time','location')
        e5 = e5.sel(location=bw.location).transpose('time','location')
        nk = nk.sel(location=bw.location).transpose('time','location')

        fig,axes = plt.subplots(3,1,\
            figsize=(4.535076795350768, 2.8028316011177257)
        )
        for n,loc in enumerate(bw.location.values):
            ax = axes[n]
            ax.plot(bw.sel(location=loc),'o',markersize=0.5,alpha=0.5,color='k',label='bw')
            ax.plot(e5.sel(location=loc),'o',markersize=0.5,alpha=0.5,color='darkgreen',label='e5')
            ax.plot(nk.sel(location=loc),'o',markersize=0.5,alpha=0.5,color='orange',label='nk')

        ax.legend()
        latex.save_figure(fig,'/nird/home/heau/figures_paper/check_corr'+t_end+t_cen)

    plt.close()
    # figure settings
    cmap   = cmocean.cm.curl
    #levels = np.sort(np.hstack([np.arange(0.05,1.,0.15),-np.arange(0.05,1.,0.15)]))
    levels = np.arange(-0.4,0.45,0.05)
    norm   = BoundaryNorm(levels,cmap.N,extend='both')

    latex.set_style(style='white')
    fig,ax = plt.subplots(1,1,\
        figsize=(4.535076795350768, 2.8028316011177257),\
        subplot_kw=dict(projection=ccrs.NorthPolarStereo())
    )

    rmse = acc_eb - acc_nb
    rmse = rmse.where(np.isfinite(rmse),drop=True)

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    feature = ax.add_feature(coast, edgecolor='k', linewidth=0.5, zorder=40)

    cs = ax.scatter(
        rmse.lon.values,
        rmse.lat.values,
        c=rmse.values,
        s=4.6,
        norm=norm,
        cmap=cmap,
        #marker='s',
        transform=ccrs.PlateCarree(),
        zorder=55,
        edgecolors='black',
        linewidth=0.3,
    )

    cbar=fig.colorbar(
        cs,
        ax=ax,
        #fraction=0.026,
        shrink=0.8,
        aspect=25,
        orientation='vertical'
    )
    #levels=np.sort(np.hstack([np.arange(0.05,1.,0.3),-np.arange(0.05,1.,0.3)]))
    #cbar.set_ticks(levels)
    #cbar.set_ticklabels([round(lvl,2) for lvl in levels])
    cbar.ax.tick_params(labelsize=6)
    ax.set_title('Diff ACC: (ERA,Barentswatch) - (Norkyst,Barentswatch)')#, mean ACC: '+str(round(rmse.values.mean(),2)))

    #fig.suptitle('ACC of Barentswatch and',fontsize=10)
    latex.save_figure(fig,'/nird/home/heau/figures_paper/Figure4_diff')
