import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import seaborn as sns
import xarray as xr

def choose_proj_from_xar(da):
    lon = da.coords['longitude'].values
    lat = da.coords['latitude'].values
    lon_center = lon[int(len(lon)/2)]
    if np.abs(lat.mean()) > 30:
        return ccrs.LambertConformal(central_longitude=lon_center, 
                                     central_latitude=lat.mean())
    else:
        return ccrs.PlateCarree(central_longitude=lon_center)

class Map(object):
    """Configures a spatial map with riverlines and major drainage basins."""
    def __init__(self, projection=None, transform=ccrs.PlateCarree(),
                 figure_kws=dict(), drainage_baisins=True):
        """Set projection, transform and figure keywords for the spatial map.
        
        Parameters
        ----------
        projection : cartopy.crs projection
            if None, use LambertConformal for midlatitudes and PlateCarree for tropics
        transform : cartopy.crs projection
            default is PlateCarree
        figure_kws : dict
            is passed on to plt.figure()
        drainage_baisins : bool
            if True, plots drainage baisins from worldbank.org (Jul 20, 2018)
            
        Usage Example
        ----------
        >>> m = Map(figure_kws=dict(figsize=(15,10)))
        >>> fig, ax = m.plot(xar)        
        """
        self.proj = projection
        self.transform = transform
        self.fig_kws = figure_kws
        self.drainage_baisins = drainage_baisins
        
    def plot(self, xar, **kwargs):
        """Wraps xr.DataArray.plot.pcolormesh and formats the plot as configured
        in the call to Map().
        
        title is xar.long_name
        cbar_label is xar.units
        
        Parameters
        ----------
        xar : xr.DataArray
            two-dimensional data array
        kwargs : dict
            are passed on to xr.DataArray.plot.pcolormesh()
            
        Returns
        ----------
        fig : matplotlib.pyplot.figure 
        ax : matplotlib.pyplot.axis
        """
        for dim in ['longitude', 'latitude']:
            if dim not in xar.coords:
                raise KeyError(dim+' not found in coordinates!')
        
        plt.close()
        fig = plt.figure(**self.fig_kws)
        
        if not self.proj:
            self.proj = choose_proj_from_xar(xar)
        ax = plt.axes(projection=self.proj)

        countries = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_0_boundary_lines_land',
            scale='50m',
            facecolor='none')
        rivers = cfeature.NaturalEarthFeature(scale='50m', category='physical',
                                              name='rivers_lake_centerlines', 
                                              edgecolor='blue', facecolor='none')

        ax.add_feature(countries, edgecolor='grey')
        ax.coastlines('50m')
        ax.add_feature(rivers, edgecolor='blue')

        if self.drainage_baisins:
            sf = Reader("../data/drainage_basins/Major_Basins_of_the_World.shp")
            shape_feature = ShapelyFeature(sf.geometries(),
                                           self.transform, edgecolor='black')
            ax.add_feature(shape_feature, facecolor='none', edgecolor='green')
            
            
        cbar_kwargs = kwargs.pop('cbar_kwargs', dict())
        subplot_kws = kwargs.pop('subplot_kws', dict())
        subplot_kws['projection'] = self.proj
        
        # colorbar preset to match height of plot
        if 'fraction' not in cbar_kwargs: cbar_kwargs['fraction'] = 0.025
        
        im = xar.plot.pcolormesh(ax=ax, transform=self.transform, 
                                 subplot_kws=subplot_kws, 
                                 cbar_kwargs=cbar_kwargs,
                                 **kwargs)
        return fig, ax

    def plot_point(self, ax, lat, lon):
        ax.plot(lon, lat, color='cyan', marker='o', 
                     markersize=20, mew=4, markerfacecolor='none',
                     transform=self.transform)