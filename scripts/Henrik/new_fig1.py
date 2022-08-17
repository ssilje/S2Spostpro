import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from glob import glob
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm, TwoSlopeNorm
from S2S.data_handler import BarentsWatch
import cmocean
import os

from S2S.graphics import latex

yoi = 2013
moi = 5
doi = 15

tmp_path = '/projects/NS9853K/DATA/norkyst800/processed/'

nk_path = tmp_path + 'FIG1_nk_tmp'+str(yoi)+str(moi)+str(doi)+'.nc'
bw_path = tmp_path + 'FIG1_bw_tmp'+str(yoi)+str(moi)+str(doi)+'.nc'
er_path = tmp_path + 'FIG1_er_tmp'+str(yoi)+str(moi)+str(doi)+'.nc'

def mfmt(i):
    i = str(i)
    return '0'+i if len(i)==1 else i

def get_norkyst():

    if not os.path.exists(nk_path):

        extent = [4.25,6.75,59.3,61.]

        path = '/projects/NS9853K/DATA/norkyst800/processed/norkyst800_sst_'

        clim = []
        for year in np.arange(2012,2021):
            p = glob(path+str(year)+'-'+mfmt(moi-1)+'*_daily_mean.nc')
            p.extend(glob(path+str(year)+'-'+mfmt(moi)+'*_daily_mean.nc'))
            p.extend(glob(path+str(year)+'-'+mfmt(moi+1)+'*_daily_mean.nc'))

            if len(p)>0:
                with xr.open_mfdataset(
                    p,
                    combine='nested',
                    concat_dim='time'
                ) as data:
                    data = data.sortby('time')
                    data = data.sst
                    data = data.rolling(time=7,center=True).mean()
                    data = data.where(data.time.dt.weekday==2,drop=True)
                    data = data.where(data.lat>=extent[2],drop=True)
                    data = data.where(data.lat<=extent[3],drop=True)
                    data = data.where(data.lon>=extent[0],drop=True)
                    data = data.where(data.lon<=extent[1],drop=True)
                    data = data

                    if all(data.time.dt.year==yoi):
                        print(np.unique(data.time.dt.year))
                        toi  = data.sel(time=str(yoi)+'-'+mfmt(moi)+'-'+mfmt(doi)+'T00:00:00.000000000')
                    else:
                        clim.append(data.where(data.time.dt.month==moi,drop=True))

        clim = xr.concat(clim,'time')
        toi = (toi - clim.mean('time'))/clim.std('time')

        toi.to_netcdf(nk_path)

        toi = toi.load()

    else:
        with xr.open_dataarray(nk_path) as toi:
            toi = toi.load()

    return toi

def get_ERA():

    if not os.path.exists(er_path):

        extent = [3.25,7.75,58.3,62.]
        path = '/projects/NS9853K/DATA/SFE/ERA_daily_nc/sea_surface_temperature_'
        clim = []
        for year in np.arange(2012,2021):
            p = glob(path+str(year)+mfmt(moi)+'*.nc')
            p.extend(glob(path+str(year)+mfmt(moi)+'*.nc'))
            p.extend(glob(path+str(year)+mfmt(moi)+'*.nc'))

            with xr.open_mfdataset(
                p,
                combine='nested',
                concat_dim='time'
            ) as data:
                data = data.sst
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

                if all(data.time.dt.year==yoi):
                    print(np.unique(data.time.dt.year))
                    toi  = data.sel(time=str(yoi)+'-'+mfmt(moi)+'-'+mfmt(doi)+'T00:00:00.000000000')
                else:
                    clim.append(data.where(data.time.dt.month==moi,drop=True))

        clim = xr.concat(clim,'time')
        toi = (toi - clim.mean('time'))/clim.std('time')

        toi.to_netcdf(er_path)

        toi = toi.load()

    else:
        with xr.open_dataarray(er_path) as toi:
            toi = toi.load()

    return toi

def get_barentswatch():

    if not os.path.exists(bw_path):

        extent = [4.25,6.75,59.3,61.]
        data = BarentsWatch().load(location='all',no=0,data_label='DATA')

        data = data.where(data.time.dt.month==moi,drop=True)

        data = data.where(data.lat>=extent[2],drop=True)
        data = data.where(data.lat<=extent[3],drop=True)
        data = data.where(data.lon>=extent[0],drop=True)
        data = data.where(data.lon<=extent[1],drop=True)
        data = data.sst

        #for time in data.time.values:
        #    print(time,np.isfinite(data.sel(time=time)).sum('location').values)

        clim = data.where(data.time.dt.year!=yoi,drop=True)
        toi = data.sel(time=str(yoi)+'-'+mfmt(moi)+'-'+mfmt(doi)+'T00:00:00.000000000')
        toi = (toi - clim.mean('time'))/clim.std('time')

        toi.to_netcdf(bw_path)

        toi = toi.load()

    else:
        with xr.open_dataarray(bw_path) as toi:
            toi = toi.load()

    return toi

def _plot_():

    _a_=True
    _b_=True

    coast = cfeature.NaturalEarthFeature(category='physical', scale='10m',
                                facecolor='none', name='coastline')

    extent = [4.25,6.75,59.3,61.]



    # data = data.isel(time=15,step=20).mean('number')
    # data = data.stack(point=['lat','lon'])
    # time = data.valid_time.values

    latex.set_style(style='white')
    fig,axes = plt.subplots(1,3,\
        figsize=(4.535076795350768, 2.8028316011177257),\
        subplot_kw=dict(projection=ccrs.NorthPolarStereo())
    )

    #print(latex.set_size(width=345,subplots=(1,1),fraction=0.95))


    # a)
    print('a')
    cmap   = plt.get_cmap('terrain')
    levels = np.arange(-3,0.75,0.25)
    norm   = BoundaryNorm(levels,cmap.N,extend='both')
    # norm = TwoSlopeNorm(vcenter=0,vmin=-3,vmax=1)

    if _a_:

        data = get_ERA().load()

        ax = axes[0]
        # cs = ax.scatter(
        #     data.lon,
        #     data.lat,
        #     c=data,
        #     s=7,
        #     cmap=cmap,
        #     transform=ccrs.PlateCarree(),
        #     norm=norm,
        #     zorder=30
        # )
        ax.pcolor(
            data.lon,
            data.lat,
            data,
            cmap=cmap,
            transform=ccrs.PlateCarree(),
            norm=norm,
            zorder=30,
            #extend='both'
        )

        ax.set_extent(extent, crs=ccrs.PlateCarree())
        resol = '10m'
        # ax.coastlines(resolution='10m',linewidth=0.1)
        feature = ax.add_feature(coast, edgecolor='k', linewidth=0.5, zorder=40)
        ax.text(0.06, 0.94, 'a)', fontsize=10, horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes, zorder=50)
        # ax.set_title('a) EC')

    if _b_:

        # b)
        print('b')
        data = get_norkyst()
        # data = data.stack(point=['X','Y'])
        # land = xr.where(np.isfinite(data),np.nan,1.).dropna('point')

        ax = axes[1]
        # cs = ax.scatter(
        #     data.lon,
        #     data.lat,
        #     c=data,
        #     s=1.5,
        #     alpha=1.,
        #     marker='.',
        #     cmap=cmap,
        #     transform=ccrs.PlateCarree(),
        #     norm=norm,
        #     edgecolor='none',
        #     linewidths=0.
        # )
        cs = ax.pcolor(
            data.lon,
            data.lat,
            data,
            cmap=cmap,
            transform=ccrs.PlateCarree(),
            norm=norm,
            zorder=30
        )

        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.text(0.06, 0.94, 'b)', fontsize=10, horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes,zorder=50)

    #c
    print('c')
    # data = get_barentswatch()
    data = get_barentswatch()

    no_data = data.where(~np.isfinite(data),drop=True)

    ax = axes[2]
    ax.scatter(
        no_data.lon.values,
        no_data.lat.values,
        c='k',
        s=2.6,
        norm=norm,
        cmap=cmap,
        #marker='s',
        transform=ccrs.PlateCarree(),
        zorder=25,
        edgecolors='black',
        linewidth=0.3,

    )
    data = data.where(np.isfinite(data),drop=True)

    cs = ax.scatter(
        data.lon.values,
        data.lat.values,
        c=data.values,
        s=3.1,
        norm=norm,
        cmap=cmap,
        #marker='s',
        transform=ccrs.PlateCarree(),
        zorder=25,
        edgecolors='black',
        linewidth=0.3,

    )

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    resol = '10m'
    # ax.coastlines(resolution='10m',linewidth=0.1)
    feature = ax.add_feature(coast, edgecolor='k', linewidth=0.5)
    ax.text(0.06, 0.94, 'c)', fontsize=10, horizontalalignment='center',
        verticalalignment='center', transform=ax.transAxes, zorder=50)

    #data = get_bw()

    #for loc,name,col in zip(data.location,data.loc_name.values,['red','green']):
    #    d = data.sel(location=loc,time='2015-10-21')

    #    ax.scatter(
    #        d.lon,
    #        d.lat,
    #        c=d,
    #        s=6,
    #        norm=norm,
    #        cmap=cmap,
    #        marker='s',
    #        edgecolor='yellow',
    #        transform=ccrs.PlateCarree(),
    #        zorder=30,
    #        label=name
    #    )
    # ax.legend(fontsize=10)

    # for loc,name in zip(data.location,data.loc_name.values):
    #     d = data.sel(location=loc)
    #     ax.text(d.lon, d.lat, name, fontsize=8,
    #         horizontalalignment='center',
    #         verticalalignment='top',
    #         color='red',
    #         transform=ccrs.PlateCarree(),
    #         zorder=40
    #     )

    cbar=fig.colorbar(
        cs,
        ax=axes.ravel().tolist(),
        #fraction=0.026,
        shrink=0.8,
        aspect=25,
        orientation='horizontal'
    )
    cbar.ax.set_xlabel('[$^\circ$C]',fontsize=10)

    cbar.ax.tick_params(labelsize=10)

    latex.save_figure(fig,'/nird/home/heau/figures_paper/new062022_fig1')

def plot():

    loc = BarentsWatch().load(location=['Ljonesbjørgene','Aga Ø']).location
    loc = loc.assign_coords(names=('location',['Ljonesbjørgene','Aga Ø']))

    bw = get_barentswatch()

    _a_=True
    _b_=True

    coast = cfeature.NaturalEarthFeature(category='physical', scale='10m',
                                facecolor='none', name='coastline')

    extent = [4.25,6.75,59.3,61.]



    # data = data.isel(time=15,step=20).mean('number')
    # data = data.stack(point=['lat','lon'])
    # time = data.valid_time.values

    latex.set_style(style='white')
    fig,axes = plt.subplots(1,2,\
        figsize=(4.535076795350768, 2.8028316011177257),\
        subplot_kw=dict(projection=ccrs.NorthPolarStereo())
    )

    #print(latex.set_size(width=345,subplots=(1,1),fraction=0.95))


    # a)
    print('a')
    cmap   = cmocean.cm.balance
    levels = np.arange(-2.5,3.,0.5)
    norm   = BoundaryNorm(levels,cmap.N,extend='both')
    # norm = TwoSlopeNorm(vcenter=0,vmin=-3,vmax=1)

    if _a_:

        data = get_ERA().load()

        ax = axes[0]
        # cs = ax.scatter(
        #     data.lon,
        #     data.lat,
        #     c=data,
        #     s=7,
        #     cmap=cmap,
        #     transform=ccrs.PlateCarree(),
        #     norm=norm,
        #     zorder=30
        # )
        ax.pcolor(
            data.lon,
            data.lat,
            data,
            cmap=cmap,
            transform=ccrs.PlateCarree(),
            norm=norm,
            zorder=30,
            #extend='both'
        )

        ax.set_extent(extent, crs=ccrs.PlateCarree())
        resol = '10m'
        # ax.coastlines(resolution='10m',linewidth=0.1)
        feature = ax.add_feature(coast, edgecolor='k', linewidth=0.5, zorder=40)
        ax.text(0.06, 0.94, 'a)', fontsize=10, horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes, zorder=50)
        # ax.set_title('a) EC')

        no_data = bw.where(~np.isfinite(bw),drop=True)

        ax.scatter(
            no_data.lon.values,
            no_data.lat.values,
            c='k',
            s=2.6,
            norm=norm,
            cmap=cmap,
            #marker='s',
            transform=ccrs.PlateCarree(),
            zorder=50,
            edgecolors='black',
            linewidth=0.3,

        )
        data_bw = bw.where(np.isfinite(bw),drop=True)

        cs = ax.scatter(
            data_bw.lon.values,
            data_bw.lat.values,
            c=data_bw.values,
            s=3.1,
            norm=norm,
            cmap=cmap,
            #marker='s',
            transform=ccrs.PlateCarree(),
            zorder=55,
            edgecolors='black',
            linewidth=0.3,

        )

        lo = loc.isel(location=1)
        ax.text(6.5,60.7,lo.names.values,transform=ccrs.PlateCarree(),ha='center',va='bottom',bbox=dict(facecolor='none',linewidth=0.5, edgecolor='k', boxstyle='round'))
        ax.plot([lo.lon,6.48],[lo.lat,60.68],linewidth=0.5,color='k',transform=ccrs.PlateCarree(),zorder=100)

    if _b_:

        # b)
        print('b')
        data = get_norkyst()
        # data = data.stack(point=['X','Y'])
        # land = xr.where(np.isfinite(data),np.nan,1.).dropna('point')

        ax = axes[1]
        # cs = ax.scatter(
        #     data.lon,
        #     data.lat,
        #     c=data,
        #     s=1.5,
        #     alpha=1.,
        #     marker='.',
        #     cmap=cmap,
        #     transform=ccrs.PlateCarree(),
        #     norm=norm,
        #     edgecolor='none',
        #     linewidths=0.
        # )
        cs = ax.pcolor(
            data.lon,
            data.lat,
            data,
            cmap=cmap,
            transform=ccrs.PlateCarree(),
            norm=norm,
            zorder=30
        )

        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.text(0.06, 0.94, 'b)', fontsize=10, horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes,zorder=50)

        no_data = bw.where(~np.isfinite(bw),drop=True)

        ax.scatter(
            no_data.lon.values,
            no_data.lat.values,
            c='k',
            s=3.6,
            norm=norm,
            cmap=cmap,
            #marker='s',
            transform=ccrs.PlateCarree(),
            zorder=50,
            edgecolors='black',
            linewidth=0.3,

        )
        data_bw = bw.where(np.isfinite(bw),drop=True)

        cs = ax.scatter(
            data_bw.lon.values,
            data_bw.lat.values,
            c=data_bw.values,
            s=4.6,
            norm=norm,
            cmap=cmap,
            #marker='s',
            transform=ccrs.PlateCarree(),
            zorder=55,
            edgecolors='black',
            linewidth=0.3,

        )

        # [4.25,6.75,59.3,61.]
        lo = loc.isel(location=0)
        ax.text(6.5,60.7,lo.names.values,transform=ccrs.PlateCarree(),ha='center',va='bottom',bbox=dict(facecolor='none', edgecolor='k', boxstyle='round',linewidth=0.5))
        ax.plot([lo.lon,6.48],[lo.lat,60.68],linewidth=0.5,color='k',transform=ccrs.PlateCarree(),zorder=100)



    cbar=fig.colorbar(
        cs,
        ax=axes.ravel().tolist(),
        #fraction=0.026,
        shrink=0.8,
        aspect=25,
        orientation='horizontal'
    )
    # cbar.ax.set_xlabel('[$^\circ$C]',fontsize=10)
    cbar.ax.set_xlabel('Anomalies from climate',fontsize=10)
    cbar.ax.tick_params(labelsize=10)

    latex.save_figure(fig,'/nird/home/heau/figures_paper/Figure1')

def main():

    plot()
