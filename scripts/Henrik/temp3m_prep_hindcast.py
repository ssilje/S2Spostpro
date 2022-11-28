from S2S import models, xarray_helpers
from datetime import datetime
import numpy  as np
import xarray as xr
import os
from glob import glob
import pandas as pd
from sklearn.neighbors import BallTree
from S2S.process       import Hindcast, Observations
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cmocean

def main():

    obs_path = '/projects/NS9853K/DATA/norkyst800/station_3m/temperature_insitu_3m_norkyst800_barentswatch_closest_20060101-20220920.nc'
    tmp_path = '/projects/NS9853K/DATA/tmp/'
    obs_time1 = datetime(year=2005,month=11,day=1) # noted 22/10/22 for new NK data
    obs_time2 = datetime(year=2021,month=7,day=19)
    hindcast = Hindcast(
            var='sst',
            t_start=(2020,7,16),
            t_end=(2021,7,19),
            bounds=(0,28,55,75),
            high_res=True,
            process=False,
            steps=pd.to_timedelta([4,11,18,25,32,39],'D'),
            download=False,
            split_work=True,
            cross_val=True,
            period=[obs_time1,obs_time2]
        )
    # cap=1.
    # hindcast = hindcast.data.transpose('lat','lon',...).isel(member=0)
    # hindcast = hindcast.where(hindcast.time.dt.year==2006,drop=True)
    # hindcast = hindcast.isel(time=np.arange(0,101,101//12))
    # # hindcast = hindcast.interp(
    # #     {'lon':hindcast.lon.values-0.25,'lat':hindcast.lat.values-0.25}
    # # )
    # hindcast = hindcast.stack(point=['lon','lat'])
    # hindcast = hindcast.where(hindcast>cap)
    #
    # for time in hindcast.time:
    #
    #     fig, axs = plt.subplots(
    #         nrows=3,
    #         ncols=2,
    #         subplot_kw={'projection': ccrs.PlateCarree()},
    #         figsize=(20,14)
    #     )
    #
    #     bins = np.arange(cap,10.5,0.5)
    #     cmap = plt.get_cmap('viridis', len(bins))
    #     norm = mpl.colors.BoundaryNorm(boundaries=bins, ncolors=cmap.N )
    #
    #     for ax,step in zip(axs.flatten(),hindcast.step):
    #
    #         cs = ax.scatter(
    #             x=hindcast.lon,
    #             y=hindcast.lat,
    #             c=hindcast.sel(time=time,step=step),
    #             transform=ccrs.PlateCarree(),
    #             cmap=cmap,
    #             norm=norm,
    #             s=10,
    #             edgecolor='k',
    #             linewidth=0.5
    #         )
    #         ax.coastlines()
    #         ax.set_title('lead time: '+str(step.dt.days.values))
    #
    #     fig.colorbar(cs,ax=axs.ravel().tolist())
    #     plt.savefig(
    #         '/nird/home/heau/fig/hindcast_init_'+\
    #         pd.to_datetime( str( time.values ) ).strftime('%d-%m-%Y')+\
    #         '_capped'+str(int(cap*10))+'.png',
    #         dpi=250
    #     )
    #     plt.close()
    #
    #
    #
    # exit()

    # load mask
    maskpath =\
        '/projects/NS9853K/DATA/S2S/MARS/hindcast/ECMWF/sfc/lsm_CY47R1_05x05_2021-03-18_cf.grb'

    with xr.open_dataarray(maskpath) as landmask:
        landmask=landmask.sortby(['latitude','longitude'])
        landmask=landmask.rename(dict(latitude='lat',longitude='lon'))
        landmask=landmask.sel(lon=slice(0,28),lat=slice(55,75))

    hc = hindcast
    _std_ = hindcast.std
    hindcast = hindcast.data_a
    # hindcast = hindcast.where(
    #     (_std_.max('time')-_std_.min('time')).isel(step=4) > 0.3
    # )
    hindcast = hindcast.where(landmask==0,drop=True)
    print('nans after condition:',np.isnan(hindcast).sum().values)
    hindcast = hindcast.stack(point=['lat','lon'])

    print('number of points before filter:',len(hindcast.point))
    hindcast = hindcast.dropna(dim='point',how='all')
    hindcast = hindcast.dropna(dim='time',how='all')
    hindcast = hindcast.dropna(dim='step',how='all')

    print('number of points after filter:',len(hindcast.point))
    print('nans after filtering:',np.isnan(hindcast).sum().values)
    print('finities after filtering:',np.isfinite(hindcast).sum().values)

    # nans = np.isnan(hindcast).sum(['time'])
    # print(nans.max('point'))
    # exit()
    # time = hindcast.time.isel(time=100)
    # fig, axs = plt.subplots(
    #     nrows=3,
    #     ncols=2,
    #     subplot_kw={'projection': ccrs.PlateCarree()},
    #     figsize=(20,14)
    # )
    #
    # for ax,step in zip(axs.flatten(),hindcast.step):
    #
    #     cs = ax.scatter(
    #         x=hindcast.lon,
    #         y=hindcast.lat,
    #         c=hindcast.sel(time=time,step=step).mean('member'),
    #         transform=ccrs.PlateCarree(),
    #         # cmap=cmap,
    #         # norm=norm,
    #         s=12,
    #         # edgecolor='k',
    #         linewidth=0.5
    #     )
    #     ax.coastlines()
    #     ax.set_title('lead time: '+str(step.dt.days.values))
    #
    # fig.colorbar(cs,ax=axs.ravel().tolist())
    # plt.savefig(
    #     '/nird/home/heau/fig/double_check_masking.png',
    #     dpi=250
    # )
    # plt.close()


    # hindcast = hindcast.dropna(dim='step',how='all')
    # hindcast = hindcast.dropna(dim='point',how='all')

    # print(hindcast)
    # print(np.isfinite(hindcast).sum(['time','step']).values)

    with xr.open_dataarray(obs_path) as obs:
        obs = obs
        location = obs.location

    ycoords = np.deg2rad(
        np.stack([hindcast.lat,hindcast.lon],axis=-1)
    )
    xcoords = np.deg2rad(
        np.stack([location.lat,location.lon],axis=-1)
    )

    tree    = BallTree(ycoords,metric="haversine")
    k_indices = tree.query(xcoords,return_distance=False).squeeze()

    hindcast = hindcast.isel(point=k_indices)
    hindcast = hindcast.assign_coords(point=location.values)
    hindcast = hindcast.rename({'point':'location'})

    hindcast.to_netcdf(tmp_path+'temp3_hindcast_at_point_location_20.nc')

    observations = Observations(
        name='temp3m_norkyst_20',
        observations=obs,
        forecast=hc,
        process=False
    )

    co_path = tmp_path + "temp3_combo_at_point_location_20.nc"

    if not os.path.exists(co_path) or True:

        combo = models.combo(
            init_value      = observations.init_a,
            model           = hindcast.mean('member'),
            observations    = observations.data_a
        )
        combo.to_netcdf(co_path)

    cos_path = tmp_path + "temp3_combo_scaled_at_point_location_20.nc"

    if not os.path.exists(cos_path) or True:

        with xr.open_dataarray(co_path) as combo:

            mean_,std_ = xarray_helpers.o_climatology(combo,window=30,cross_validation=True)
            combo_scaled = ( (combo-mean_)/std_ ) + mean_
            combo_scaled.to_netcdf(cos_path)
