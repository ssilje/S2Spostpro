import pandas as pd
import xarray as xr


from S2S import models

observations= xr.open_dataset('/nird/projects/NS9001K/share/pers_model/observations.nc')
init_value = xr.open_dataset('/nird/projects/NS9001K/share/pers_model/init_value.nc')


pers  = models.persistence(
                init_value   = observations.sst,
                observations = init_value.sst
                )

print('observations')
print(observations.sst)

print('init_value')
print(init_value.sst)

print('pers')
print(pers)
