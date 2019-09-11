import numpy as np
import pandas as pd
import datetime as dt

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
    lat_center = lat[int(len(lat)/2)]
    if np.abs(lat.mean()) > 30:
        return ccrs.Mercator(central_longitude=lon_center, min_latitude=lat.min()-1,
                             max_latitude=lat.max()+1, globe=None,
                             latitude_true_scale=lat_center)

#        return ccrs.LambertConformal(central_longitude=lon_center,
#                                     central_latitude=lat.mean())
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
            if None, use Mercator (old: LambertConformal) for midlatitudes and
            PlateCarree for tropics
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
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
        gl.xlabels_top = False

        if self.drainage_baisins:
            sf = Reader("../data/drainage_basins/Major_Basins_of_the_World.shp")
            shape_feature = ShapelyFeature(sf.geometries(),
                                           self.transform, edgecolor='black')
            ax.add_feature(shape_feature, facecolor='none', edgecolor='green')

        # cbar_kwargs = kwargs.pop('cbar_kwargs', dict())
        subplot_kws = kwargs.pop('subplot_kws', dict())
        subplot_kws['projection'] = self.proj

        # choose which colormap to use: pos and neg values => RdYlGn, else inferno
        if ((xar.max()-xar.min()) > xar.max()):
            cmap = 'RdYlGn'
        else:
            cmap = 'spring_r'

        # colorbar preset to match height of plot
        # if 'fraction' not in cbar_kwargs: cbar_kwargs['fraction'] = 0.015
        xar.plot.pcolormesh(ax=ax, transform=self.transform,
                            subplot_kws=subplot_kws,
                            cmap=cmap,
                            **kwargs)
        return fig, ax

    def plot_point(self, ax, lat, lon):
        ax.plot(lon, lat, color='cyan', marker='o',
                markersize=20, mew=4, markerfacecolor='none',
                transform=self.transform)


def plot_ts(da, key):
    """Plot a times series for a given xarray dataarray.

    Parameters
    ----------
    da : xr.DataArray
        one-dimensional data array
    key : str
        parameter name / ylabel
    """
    p = sns.lineplot(data=da.to_pandas(), linewidth=2)
    p.set_xlabel('time')
    p.set_ylabel(key)

# ########## Model plotting


def plot_recurrent(ax, truth, prediction, each_N=7):
    """Plot predictions of recurrent nets.

    Parameters
    ----------
    ax : matplotlib axes object
    truth : xr.DataArray
        one-dimensional data array (time,)
    prediction : xr.DataArray
        two-dimensional data array of (init_time, forecast_day)
    """
    truth.plot(label='truth', linewidth=2, ax=ax)
    times = prediction.init_time
    for i, init in enumerate(times):
        if not i % each_N == 0:
            continue

        da = prediction.sel(init_time=init)
        time = [pd.Timestamp(da.coords['init_time'].values)
                + dt.timedelta(days=int(i)) for i in da.coords['forecast_day'].values]

        df = pd.Series(da.values, index=time)
        df.plot(ax=ax, label=str(init))
    ax.legend(['truth', 'prediction'])


def feature_importance_plot(xda_features, score_decreases):
    """Bar chart of importances.
    xda_features : xr.DataArray.features
    score_decreases : list of np.arrays
    """
    labels = [e[0] for e in xda_features.values]
    n_iter = len(score_decreases)
    width = 1/n_iter
    plt.subplots(figsize=(20, 5))

    for c in range(n_iter):
        x = np.arange(len(labels))+c/n_iter
        plt.bar(x, score_decreases[c], width=-width, align='edge')

    plt.grid()
    plt.xticks(ticks=x, labels=labels, rotation=45)


def plot_multif_prediction(pred_multif, y_truth, forecast_range=14, title=None):
    """Convenience function for plotting multiforecast shaped prediction and truth.
    Note when using the returned 'ax' variable to plot additional lines outside of the function
    the corresponding objects need to be pd.Series (xr.DataArray objects will not be plotted
    onto the axis)!
    
    Parameters
    ----------
        pred_multif     : xr.DataArray
        y_truth         : xr.DataArray
        forecast_range  : int
        title           : str
    """
    fig, ax = plt.subplots(figsize=(15,5))
    y_truth.sel({'time': pred_multif.time.values.ravel()}).to_pandas().plot(ax=ax, label='truth')

    pdseries = pd.Series(data=pred_multif.sel(num_of_forecast=1).values,
                         index=pred_multif.sel(num_of_forecast=1).time.values)
    pdseries.plot(ax=ax, label='forecast')
    plt.legend()
    for i in pred_multif.num_of_forecast[1:]:
        fcst = pd.Series(data=pred_multif.sel(num_of_forecast=i).values,
                             index=pred_multif.sel(num_of_forecast=i).time.values)
        fcst.plot(ax=ax)

    ax.set_ylabel('river discharge [m$^3$/s]')
        
    y_o = y_truth.loc[{'time': pred_multif.time.values.ravel()}].values
    y_m = pred_multif.values.ravel()

    rmse = np.sqrt(np.nanmean((y_m - y_o)**2))
    nse = 1 - np.sum((y_m - y_o)**2)/(np.sum((y_o - np.nanmean(y_o))**2))

    plt.title(f"{title} | RMSE={round(float(rmse), 2)}; NSE={round(float(nse), 2)} |")
    return fig, ax
