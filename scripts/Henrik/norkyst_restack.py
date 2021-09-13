import xarray as xr
import time
import numpy as np
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar

from S2S.local_configuration import config

# NorKyst-800m_ZDEPTHS_avg.an.2013052612.nc

def test():

    path         = config['NORKYST']
    arb_filename = 'norkyst800_sst_*_daily_mean_at-BW.nc'

    n_workers = 8
    n_threads = 2
    processes = True

    cluster = LocalCluster(
                            n_workers=n_workers,
                            threads_per_worker=n_threads,
                            processes=processes,
                            lifetime='1 hour',
                            lifetime_stagger='10 minutes',
                            lifetime_restart=True
                        )

    client = Client(cluster)

    data   = xr.open_mfdataset(
                                path+arb_filename,
                                chunks={'location':9},
                                parallel=True
                            )

    # groups = list(data.groupby('time.month'))
    #
    # for n,g in groups:
    #     print(n,np.isfinite(g.values).sum().compute())
    print(data)
    
    for loc in data.location.values:
        print(loc)
        data.sel(location=loc).to_netcdf(path+'NorKyst800_'+str(loc)+'.nc')

    print(data)

    cluster.close()
    client.close()
