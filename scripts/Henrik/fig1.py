import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm
from S2S.data_handler import BarentsWatch

from S2S.graphics import latex

def get_ERA():
    extent = [3.25,7.75,58.3,62.]
    path = '/projects/NS9853K/DATA/SFE/ERA_daily_nc/sea_surface_temperature_201510'
    paths = [path+str(n)+'.nc' for n in [18,19,20,21,22,23,24]]

    with xr.open_mfdataset(
        paths,
        combine='nested',
        concat_dim='time'
    ) as data:
        data = data.sst
        data = data.mean('time')
        data = data.rename({'latitude':'lat','longitude':'lon'})
        data = data.where(data.lat>=extent[2],drop=True)
        data = data.where(data.lat<=extent[3],drop=True)
        data = data.where(data.lon>=extent[0],drop=True)
        data = data.where(data.lon<=extent[1],drop=True)
    return data - 273.15

def get_EC():
    extent = [4.25,6.75,59.3,61.]

    path = '/projects/NS9853K/DATA/S2S/MARS/hindcast/ECMWF/sfc/sst/sst_CY47R1_05x05_2020-10-05_pf.grb'
    with xr.open_dataarray(path) as data:
        data = data.rename({'latitude':'lat','longitude':'lon'})
        data = data.sortby(['lat','lon'])
        # data = data.sel(
        #     lat=slice(extent[2],extent[3]),
        #     lon=slice(extent[0],extent[1]),
        #     )
        data = data.rolling(step=7,center=True).mean()
        data = data - 273.15
    return data

def get_norkyst():
    extent = [4.25,6.75,59.3,61.]

    path = '/projects/NS9853K/DATA/norkyst800/processed/norkyst800_sst_2015-10-'
    paths = [path+str(n)+'_daily_mean.nc' for n in [18,19,20,21,22,23,24]]

    with xr.open_mfdataset(
        paths,
        combine='nested',
        concat_dim='time'
    ) as data:
        data = data.sst
        data = data.mean('time')

        data = data.where(data.lat>=extent[2],drop=True)
        data = data.where(data.lat<=extent[3],drop=True)
        data = data.where(data.lon>=extent[0],drop=True)
        data = data.where(data.lon<=extent[1],drop=True)

    return data

def get_barentswatch():

    path = '/projects/NS9853K/DATA/norkyst800/processed/hardanger/norkyst800_sst_daily_mean_hardanger_atBW.nc'

    with xr.open_dataarray(path) as data:
        data = data.squeeze()
    return data

def get_bw():

    extent = [4.25,6.75,59.3,61.]
    data = BarentsWatch().load(location=['Ljonesbjørgene','Aga Ø'],no=380,data_label='DATA')
    data = data.assign_coords(loc_name=('location',['Ljonesbjørgene','Aga Ø']))

    # data = data.where(data.lat>=extent[2],drop=True)
    # data = data.where(data.lat<=extent[3],drop=True)
    # data = data.where(data.lon>=extent[0],drop=True)
    # data = data.where(data.lon<=extent[1],drop=True)

    return data.sst

def get_all_bw():

    extent = [4.25,6.75,59.3,61.]
    data = BarentsWatch().load(location='all',no=0,data_label='DATA')

    data = data.where(data.lat>=extent[2],drop=True)
    data = data.where(data.lat<=extent[3],drop=True)
    data = data.where(data.lon>=extent[0],drop=True)
    data = data.where(data.lon<=extent[1],drop=True)

    return data.sst

def plot():

    coast = cfeature.NaturalEarthFeature(category='physical', scale='10m',
                                facecolor='none', name='coastline')

    extent = [4.25,6.75,59.3,61.]

    data = get_ERA().load()

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
    cmap   = plt.get_cmap('coolwarm')
    levels = np.arange(8,16.5,0.5)
    norm   = BoundaryNorm(levels,cmap.N)

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
    cs = ax.pcolor(
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
    feature = ax.add_feature(coast, edgecolor='gray', zorder=40)
    ax.text(0.06, 0.94, 'a)', fontsize=10, horizontalalignment='center',
        verticalalignment='center', transform=ax.transAxes, zorder=50)
    # ax.set_title('a) EC')

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
    data = get_all_bw()

    data = data.sel(time='2015-07-15')
    no_data = data.where(~np.isfinite(data),drop=True)

    ax = axes[2]
    ax.scatter(
        no_data.lon,
        no_data.lat,
        c='k',
        s=0.9,
        norm=norm,
        cmap=cmap,
        marker='s',
        transform=ccrs.PlateCarree(),
        zorder=25
    )
    data = data.where(np.isfinite(data),drop=True)
    print(data.values,data.values.min(),data.values.max())
    ax.scatter(
        data.lon,
        data.lat,
        c='k',
        s=1.1,
        norm=norm,
        cmap=cmap,
        marker='s',
        transform=ccrs.PlateCarree(),
        zorder=25
    )

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    resol = '10m'
    # ax.coastlines(resolution='10m',linewidth=0.1)
    feature = ax.add_feature(coast, edgecolor='gray')
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

    latex.save_figure(fig,'/nird/home/heau/figures_paper/fig1')
