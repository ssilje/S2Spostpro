from glob import glob
import xarray as xr
import numpy as np

for d in glob(
    "/projects/NS9853K/DATA/norkyst800/norkyst800_sst_*_daily_mean.nc"
):
    d2 = "/".join([d[:33],"processed",d[34:]])

    with xr.open_dataset(d)  as data1:
        data1 = data1
    with xr.open_dataset(d2) as data2:
        data2 = data2

    print(
        data1,data2
    )
    exit()
