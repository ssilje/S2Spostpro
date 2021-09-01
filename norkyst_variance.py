import xarray as xr
import pandas as pd
import numpy as np
import json
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from S2S.local_configuration import config
from S2S.graphics import latex, graphics

from matplotlib.colors import BoundaryNorm

from dask.distributed import Client

if __name__ == '__main__':

    client = Client(n_workers=2, threads_per_worker=2, memory_limit='10MB')
    print(client.dashboard_link)

    path   = '/nird/projects/NS9853K/DATA/norkyst800/'
    # fn1    = 'norkyst800_sst_'
    # fn2    = '_var.nc'
    fname1 = 'norkyst800_sst_'
    fname2 = '_daily_mean.nc'
    mparser = {
                '01':'JAN','02':'FEB','03':'MAR',
                '04':'APR','05':'MAY','06':'JUN',
                '07':'JUL','08':'AUG','09':'SEP',
                '10':'OCT','11':'NOV','12':'DEC'
            }
    months = ['01','02','03','04','05','06','07','08','09','10','11','12']

    # latex.set_style(style='white')
    # fig,axes = plt.subplots(1,1,\
    #     figsize=latex.set_size(width=345,subplots=(1,1),fraction=0.95),\
    #     subplot_kw=dict(projection=ccrs.NorthPolarStereo()))

    # for ax,month in zip(axes.flatten(),months):

    # for month in months:
    #
    #     latex.set_style(style='white')
    #     fig,ax = plt.subplots(1,1,\
    #         figsize=latex.set_size(width=345,subplots=(1,1),fraction=0.95),\
    #         subplot_kw=dict(projection=ccrs.NorthPolarStereo()))

    fname = fname1 + '*-' + '10' + '-*' + fname2
    # fname = fname1 + '*' + fname2
    print(fname)

    # data = xr.open_dataset(path+fn1+month+fn2).squeeze()
    with xr.open_mfdataset(
                            path + fname,
                            # parallel=True,
                            chunks={'X':5,'Y':5}
                                                        ) as data:

        print(data)
        exit()
    #     # # # load to memory
    #     # data = data.load()
    #
    #     week_data  = data.temperature.rolling(time=7,center=True).mean()
    #     month_data = data.temperature.rolling(time=30,center=True).mean()
    #
    #     data = ( ( week_data - month_data )**2 )
    #
    #     del week_data
    #     del month_data
    #
    #     cmap   = latex.cm_rgc(c='yellow')
    #     levels = np.arange(0,3,0.5)
    #     norm   = BoundaryNorm(levels,cmap.N)
    #
    #     for month in np.arange(1,13):
    #
    #         monthly_data = data.where(data.time.dt.month==month,drop=True)
    #         monthly_data = xr.ufuncs.sqrt(
    #                     monthly_data.sum('time') / monthly_data.sizes['time']
    #                             )
    #
    #         latex.set_style(style='white')
    #         fig,ax = plt.subplots(1,1,\
    #             figsize=latex.set_size(width=345,subplots=(1,1),fraction=0.95),\
    #             subplot_kw=dict(projection=ccrs.NorthPolarStereo()))
    #
    #         cs = ax.contourf(
    #                             monthly_data.longitude,
    #                             monthly_data.latitude,
    #                             monthly_data,
    #                             transform=ccrs.PlateCarree(),
    #                             cmap=cmap,
    #                             norm=norm,
    #                             extend='max',
    #                             levels=levels
    #                         )
    #
    #         ax.coastlines(resolution='10m', color='grey',\
    #                                 linewidth=0.2)
    #         ax.set_title(mparser[month] + ' NorKyst-800 SST RMSD [degC^2]')
    #         fig.colorbar(cs,ax=ax)
    #         graphics.save_fig(fig,'rmsd_map_norkyst_'+month)
