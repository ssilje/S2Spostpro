import xarray as xr
import pandas as pd
import numpy as np
import json
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import glob

from S2S.local_configuration import config
from S2S.graphics import latex, graphics

from matplotlib.colors import BoundaryNorm

def wash_norkyst(ds,time):

    ds = ds.assign_coords(time=[pd.Timestamp(time)])
    ds = ds.rename({'latitude':'lat','longitude':'lon','temperature':'sst'})

    return ds

def run():
    mparser = {
                '01':'JAN','02':'FEB','03':'MAR',
                '04':'APR','05':'MAY','06':'JUN',
                '07':'JUL','08':'AUG','09':'SEP',
                '10':'OCT','11':'NOV','12':'DEC'
            }
    months = ['01','02','03','04','05','06','07','08','09','10','11','12']

    path         = config['NORKYST']
    arb_filename = 'norkyst800_sst_*_daily_mean.nc'

    d = []
    for filename in glob.glob(path+arb_filename):
        time = filename[-24:-14]
        d.append( wash_norkyst( xr.open_dataset( filename ), time ) )

    ds = xr.concat(d,'time')
    print('done')
    print(ds)

    sst = ds.sst.rolling(time=7,center=True).mean()

    del ds

    for month in months:

        latex.set_style(style='white')
        fig,ax = plt.subplots(1,1,\
            figsize=latex.set_size(width=345,subplots=(1,1),fraction=0.95),\
            subplot_kw=dict(projection=ccrs.NorthPolarStereo()))

        cmap   = latex.cm_rgc(c='yellow')
        levels = np.arange(0,4.5,0.5)
        norm   = BoundaryNorm(levels,cmap.N)

        data = xr.ufuncs.sqrt(sst.where(sst.time.dt.month==int(month),drop=True)\
                                    .var('time',skipna=True)).squeeze()

        cs = ax.contourf(
                            data.lon,
                            data.lat,
                            data,
                            transform=ccrs.PlateCarree(),
                            cmap=cmap,
                            norm=norm,
                            extend='max',
                            levels=levels
                        )

        ax.coastlines(
                            resolution='10m',
                            color='grey',
                            linewidth=0.2
                    )
        ax.set_title(mparser[month] + ' NorKyst-800 SST sqrt(sst*2)[degC]')
        fig.colorbar(cs,ax=ax)
        graphics.save_fig(fig,'std/std_map_norkyst_'+month)

# exit()
#
#
#
# # latex.set_style(style='white')
# # fig,axes = plt.subplots(1,1,\
# #     figsize=latex.set_size(width=345,subplots=(1,1),fraction=0.95),\
# #     subplot_kw=dict(projection=ccrs.NorthPolarStereo()))
#
# # for ax,month in zip(axes.flatten(),months):
#
# for month in months:
#
#     latex.set_style(style='white')
#     fig,ax = plt.subplots(1,1,\
#         figsize=latex.set_size(width=345,subplots=(1,1),fraction=0.95),\
#         subplot_kw=dict(projection=ccrs.NorthPolarStereo()))
#
#     fname = fname1 + '*-' + month + '-*' + fname2
#     print(fname)
#
#     try:
#         # data = xr.open_dataset(path+fn1+month+fn2).squeeze()
#         with xr.open_mfdataset( path + fname, parallel=True ) as data:
#
#
#             # # load to memory
#             data = data.load()
#
#             data = data.temperature.var('time',skipna=True).squeeze()
#
#             lons = data.longitude
#             lats = data.latitude
#
#             cmap   = latex.cm_rgc(c='yellow')
#             levels = np.arange(0,7,0.5)
#             norm   = BoundaryNorm(levels,cmap.N)
#
#             cs = ax.contourf(lons,lats,data,transform=ccrs.PlateCarree(),
#                                 cmap=cmap,norm=norm,extend='max',levels=levels)
#             ax.coastlines(resolution='10m', color='grey',\
#                                     linewidth=0.2)
#             ax.set_title(mparser[month] + ' NorKyst-800 SST variance [degC^2]')
#             fig.colorbar(cs,ax=ax)
#             graphics.save_fig(fig,'variance_map_norkyst_'+month)
#
#     except (FileNotFoundError,MemoryError) as e:
#         print(e)
#         pass
