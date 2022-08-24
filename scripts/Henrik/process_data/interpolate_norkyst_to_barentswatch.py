import numpy  as np
import xarray as xr
import os
from glob import glob

from S2S.data_handler import BarentsWatch
from sklearn.neighbors import BallTree

def get_barentswatch_locations():

    path     = "/projects/NS9853K/DATA/tmp/"
    tmp_path = path + "barentswatch_locations_hardanger.nc"

    if not os.path.exists(tmp_path):

        extent = [4.25,6.75,59.3,61.] # only for hardangerfjord

        data = BarentsWatch().load(location='all',no=0,data_label='DATA')

        data = data.location

        data = data.where(data.lat>=extent[2],drop=True)
        data = data.where(data.lat<=extent[3],drop=True)
        data = data.where(data.lon>=extent[0],drop=True)
        data = data.where(data.lon<=extent[1],drop=True)

        data.to_netcdf(tmp_path)

    else:
        with xr.open_dataarray(tmp_path) as data:
            data = data

    return data

def main():

    norkyst_filenames = glob("/projects/NS9853K/DATA/norkyst800/processed/norkyst800_sst_*_daily_mean.nc")

    fname = norkyst_filenames[0]

    bw_loc = get_barentswatch_locations()

    path     = "/projects/NS9853K/DATA/tmp/"
    tmp_path = path + "norkyst_at_locations_hardanger_stationary_kindex.nc"

    if not os.path.exists(tmp_path):

        # create ball tree
        bw_coords = np.deg2rad(np.stack([bw_loc.lat,bw_loc.lon],axis=-1))

        with xr.open_dataarray(fname) as norkyst:

            norkyst = norkyst.sortby(["Y","X"])
            norkyst = norkyst.stack(point=["Y","X"])
            norkyst = norkyst.dropna(dim="point",how="all")

            coords = np.deg2rad(np.stack([norkyst.lat,norkyst.lon],axis=-1))

            tree   = BallTree(coords,metric="haversine")
            k_indices = tree.query(bw_coords,return_distance=False).squeeze()

        norkyst_timesteps = []
        N = len(norkyst_filenames)
        for n,fname in enumerate(norkyst_filenames):

            with xr.open_dataarray(fname) as norkyst:

                norkyst = norkyst.squeeze()
                norkyst = norkyst.sortby(["Y","X"])
                norkyst = norkyst.stack(point=["Y","X"])
                norkyst = norkyst.dropna(dim="point",how="all")

                # uncomment if you wanna look of each grid individually
                coords = np.deg2rad(np.stack([norkyst.lat,norkyst.lon],axis=-1))
                tree   = BallTree(coords,metric="haversine")
                k_indices = tree.query(bw_coords,return_distance=False).squeeze()

                norkyst = norkyst.isel(point=k_indices)
                norkyst = norkyst.drop("lat")
                norkyst = norkyst.drop("lon")
                norkyst = norkyst.assign_coords(point=bw_loc.location.values)
                norkyst = norkyst.assign_coords( loc_name=("point",bw_loc.loc_name.values) )
                norkyst = norkyst.assign_coords( lon=("point",bw_loc.lon.values) )
                norkyst = norkyst.assign_coords( lat=("point",bw_loc.lat.values) )
                norkyst = norkyst.rename(dict(point="location"))

                norkyst_timesteps.append(norkyst)
                print(round(n/N,2))

        norkyst = xr.concat(norkyst_timesteps,dim="time")
        norkyst.to_netcdf(tmp_path)

    else:

        with xr.open_dataarray(tmp_path) as norkyst:
            norkyst = norkyst

    print(norkyst)
