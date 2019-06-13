#!/usr/bin/env python
# coding: utf-8

# # Data Inspection / Analysis
# ### Input: DataArray mit lat, lon, time
# 
#     1) Map erstellen
#     2) m.plot(xar, save_to_path=False)
# 
# 
# ### Output: Plots im jupyter notebook von
# - mean
# - std
# - spatial covariance
# - histogram distribution

# In[1]:


import xarray as xr
import pandas as pd
import numpy as np

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature

import seaborn as sns


# In[2]:


xar = xr.open_mfdataset('../data/danube/*.nc')


# In[3]:


xar


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# method = 'std'
# 
# stats = []
# for var in xar:
#     s = getattr(xar[var], method)(dim='time')
#     stats.append(s)
# stats2 = xr.merge(stats)
# 
# for v in stats2:
#     fig = plt.figure(figsize=(15,10))
#     plt.title('variable: '+v)
#     stats[v].plot.pcolormesh()

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Projections
# 
# - ccrs.LambertConformal(central_longitude=-95, central_latitude=45)
# - ccrs.Orthographic(-110, 35)
# - ccrs.PlateCarree()

# In[4]:


class Map(object):
    def __init__(self, figure_kws=dict(), **kwargs):
        
        self.proj = kwargs.pop('projection',
                               ccrs.LambertConformal(central_longitude=-110, 
                                                     central_latitude=45))
        self.transform = kwargs.pop('transform', 
                                    ccrs.PlateCarree())
        self.kwargs = kwargs
        self.fig_kws = figure_kws
        
    def plot(self, xar):
        plt.close()
        self.fig = plt.figure(**self.fig_kws)
        self.savefig = self.fig.savefig
        ax = plt.axes(projection=self.proj);

        states_provinces = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces_lines',
            scale='50m',
            facecolor='none')
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
        #ax.add_feature(states_provinces, edgecolor='gray')
        ax.add_feature(rivers, edgecolor='blue')

        if self.kwargs.get('drainage_baisins', True):
            sf = Reader("../data/drainage_basins/Major_Basins_of_the_World.shp")
            shape_feature = ShapelyFeature(sf.geometries(),
                                           self.transform, edgecolor='black')
            ax.add_feature(shape_feature, facecolor='none', edgecolor='green')
            
        xar.plot(transform=self.transform, ax=ax,
                 subplot_kws={'projection': self.proj})
        self.ax = ax

    def plot_point(self, lat, lon):
        self.ax.plot(lon, lat, color='cyan', marker='o', 
                     markersize=20, mew=4, markerfacecolor='none',
                     transform=self.transform)


# In[ ]:


anom = xar - xar.mean('time')
field = anom['lsp']
da = field #a = field # /field.std('time')


# In[ ]:


da


# In[ ]:


point = dict(latitude=48, longitude=15)
cov = da.loc[point].dot(da)


# In[ ]:


cov.plot.pcolormesh()


# In[ ]:


m = Map(figure_kws=dict(figsize=(15,10)),
        projection=ccrs.LambertConformal(central_longitude=15, 
                                         central_latitude=48),
        transform=ccrs.PlateCarree())


# In[ ]:


point = dict(latitude=48, longitude=15)
cov = da.loc[point].dot(da)
m.plot(cov)
m.plot_point(lat=point['latitude'], lon=point['longitude'])
m.savefig('1.png')

# In[ ]:


point = dict(latitude=47, longitude=8)
cov = da.loc[point].dot(da)
m.plot(cov)
m.plot_point(lat=point['latitude'], lon=point['longitude'])
m.savefig('2.png')

# In[ ]:


point = dict(latitude=49, longitude=9)
cov = da.loc[point].dot(da)
m.plot(cov)
m.plot_point(lat=point['latitude'], lon=point['longitude'])
m.savefig('3.png')

# In[ ]:





# In[ ]:





# # Dataset Variable's Distribution
# 
# using seaborn? or too many values ("big data")
# -> probably "bokeh" module

# In[ ]:


data = xar['lsp'].values.ravel()
data = data[data > 0.001]

sns.distplot(data)

