import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import glob
from sklearn.neighbors import BallTree
from S2S.data_handler import BarentsWatch
import cartopy.crs as ccrs
from S2S.graphics import latex
import cmocean
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm
def get_barentswatch():

    extent = [4.25,6.75,59.3,61.]
    data = BarentsWatch().load(location='all',no=0,data_label='DATA')

    data = data.where(data.lat>=extent[2],drop=True)
    data = data.where(data.lat<=extent[3],drop=True)
    data = data.where(data.lon>=extent[0],drop=True)
    data = data.where(data.lon<=extent[1],drop=True)
    return data.sst

def get_norkyst():

    extent = [4.25-1,6.75+1,59.3-1,61.+1]
    path   = glob.glob('/projects/NS9853K/DATA/norkyst800/processed/norkyst800_sst_*_daily_mean.nc')[0]

    with xr.open_dataarray(path) as data:
        data = data.where(data.lat>=extent[2],drop=True)
        data = data.where(data.lat<=extent[3],drop=True)
        data = data.where(data.lon>=extent[0],drop=True)
        data = data.where(data.lon<=extent[1],drop=True)
    return data.stack(point=['X','Y'])

def get_shore():

    barentswatch = get_barentswatch()
    norkyst      = get_norkyst()
    X            = np.stack([norkyst.lat.values,norkyst.lon.values],axis=-1)
    K            = np.stack([barentswatch.lat.values,barentswatch.lon.values],axis=-1)
    X            = np.deg2rad(X)
    K            = np.deg2rad(K)

    tree = BallTree(X,metric='haversine')
    idx  = tree.query(K,k=1000,return_distance=False,sort_results=False)

    count = xr.where(np.isfinite(norkyst),1,0).values.squeeze()
    count = np.array([ count[ii].sum() for ii in idx]) / 1000

    return barentswatch.assign_coords(shore=('location',count))

def main():

    barentswatch = get_shore()

    # figure settings
    cmap   = cmocean.cm.tempo
    levels = np.arange(0,1.1,0.1)
    norm   = BoundaryNorm(levels,cmap.N,extend='both')

    # plot
    extent = [4.25,6.75,59.3,61.]
    coast = cfeature.NaturalEarthFeature(category='physical', scale='10m',
                                facecolor='none', name='coastline')

    latex.set_style(style='white')
    fig,axes = plt.subplots(1,1,\
        figsize=(2.535076795350768, 2.8028316011177257),\
        subplot_kw=dict(projection=ccrs.NorthPolarStereo())
    )

    ax = axes

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    feature = ax.add_feature(coast, edgecolor='k', linewidth=0.5, zorder=40)

    cs = ax.scatter(
        barentswatch.lon.values,
        barentswatch.lat.values,
        c=barentswatch.shore.values,
        s=4.6,
        norm=norm,
        cmap=cmap,
        #marker='s',
        transform=ccrs.PlateCarree(),
        zorder=55,
        edgecolors='black',
        linewidth=0.3,
    )

    cbar=fig.colorbar(
        cs,
        ax=ax,
        #fraction=0.026,
        shrink=0.8,
        aspect=25,
        orientation='horizontal'
    )

    latex.save_figure(fig,'/nird/home/heau/figures_paper/in_to_on_shore_metric')
