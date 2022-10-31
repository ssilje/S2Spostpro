import os
import numpy as np
import pandas as pd
import xarray as xr

from sklearn.neighbors import BallTree

from S2S         import models, xarray_helpers
from S2S.process import Hindcast, Observations

class HC:
    def __init__(self):

        path   = "/projects/NS9853K/DATA/tmp/"
        h_path = path + "hindcast_at_locations_hardanger.nc"

        if not os.path.exists(h_path):

            nk_path = path + "norkyst_at_locations_hardanger_stationary_kindex.nc"

            with xr.open_dataset(nk_path) as norkyst:
                norkyst = norkyst.location

            bounds = (4.25-1,6.75+1,59.3-1,61.+1)
            var      = 'sst'

            t_start  = (2020,7,1)
            t_end    = (2021,7,3)

            high_res = True
            steps    = pd.to_timedelta([9,16,23,30,37],'D')

            print('Get hindcast')
            hindcast = Hindcast(
                var,
                t_start,
                t_end,
                bounds,
                high_res=high_res,
                steps=steps,
                process=False,
                download=False,
                split_work=True,
                cross_val=True,
                period=[(2012,1,1),(2020,1,1)]
            )

            hc = hindcast.data_a
            hc = hc.stack(point=['lat','lon'])
            hc = hc.where(np.isfinite(hc),drop=True)

            hc_loc = np.deg2rad(np.stack([hc.lat,hc.lon],axis=-1))
            nk_loc = np.deg2rad(np.stack([norkyst.lat,norkyst.lon],axis=-1))

            tree = BallTree(hc_loc,metric="haversine")
            ind = tree.query(nk_loc,return_distance=False)

            hc  = hc.isel( point=ind.flatten() )
            lat = norkyst.lat.values
            lon = norkyst.lon.values
            hc  = hc.assign_coords(point=norkyst.location.values).rename(point='location')
            hc  = hc.assign_coords(lon=('location',lon))
            hc  = hc.assign_coords(lat=('location',lat))

            hc.to_netcdf(h_path)

        else:

            with xr.open_dataarray(h_path) as hc:
                hc = hc

        self.data   = hc
        self.data_a = hc

def main():

    path    = "/projects/NS9853K/DATA/tmp/"
    nk_path = path + "norkyst_at_locations_hardanger_stationary_kindex.nc"

    print("load norkyst data")
    with xr.open_dataarray(nk_path) as norkyst:
        norkyst = norkyst.sortby("time")

    print("running mean of norkyst data")
    norkyst = norkyst.rolling(time=7,center=True).mean()

    print("load hindcast")
    hindcast = HC()

    print('Norkyst: process')
    observations = Observations(
        name='norkyst_at_locations_hardanger_stacked_and_processed',
        observations=norkyst,
        forecast=hindcast,
        process=False
    )

    # COMBO MODEL

    co_path = path + "combo_at_locations_hardanger.nc"

    if not os.path.exists(co_path):

        combo = models.combo(
            init_value      = observations.init_a,
            model           = hindcast.data_a,
            observations    = observations.data_a
        )
        combo.to_netcdf(co_path)

    else:

        with xr.open_dataarray(co_path) as combo:
            combo = combo

    co_adj_path = path + "combo_at_locations_hardanger_adjusted.nc"

    if not os.path.exists(co_adj_path):

        dat = combo

        o_mean,o_std = xarray_helpers.o_climatology(dat,window=30,cross_validation=True)
        dat = dat - o_mean
        dat = dat / o_std
        # dat = dat + o_mean

        dat.to_netcdf(co_adj_path)


    # PERSISTENCE

    pe_path = path + "pers_at_locations_hardanger.nc"

    if not os.path.exists(pe_path):

        pers = models.persistence(observations.init_a,observations.data_a,window=30)
        pers.to_netcdf(pe_path)

    else:

        with xr.open_dataarray(pe_path) as pers:
            pers = pers

    pe_adj_path = path + "pers_at_locations_hardanger_adjusted.nc"

    if not os.path.exists(pe_adj_path):

        dat = pers

        o_mean,o_std = xarray_helpers.o_climatology(dat,window=30,cross_validation=True)
        dat = dat - o_mean
        dat = dat / o_std
        # dat = dat + o_mean

        dat.to_netcdf(pe_adj_path)
