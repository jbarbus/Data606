#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install kaggle')


# In[3]:


get_ipython().system('ls')


# In[4]:


get_ipython().system('mkdir .kaggle')


# In[11]:


import json
token = {"username":"barbus1","key":"XXX"}
with open('.kaggle/kaggle.json', 'w') as file:
    json.dump(token, file)


# In[12]:


get_ipython().system('kaggle config set -n path -v{/content}')


# In[13]:


get_ipython().system('chmod 600 /content/.kaggle/kaggle.json')


# In[14]:


get_ipython().system('mkdir ~/.kaggle')
get_ipython().system('cp /content/.kaggle/kaggle.json ~/.kaggle/kaggle.json')


# In[28]:


get_ipython().system('kaggle datasets download -d sohier/crime-in-baltimore -p /content --force --unzip')


# In[17]:


conda install geopandas


# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import sys
import sklearn as svm
import geopandas as gpd
import shapely.geometry 
import fiona
import pyproj 
import six
from shapely.geometry import Point, Polygon


# In[2]:


df = pd.read_csv('C:\content\BPD_Part_1_Victim_Based_Crime_Data.csv')


# In[3]:


df


# In[4]:


df.describe()


# In[5]:


df.info()


# In[6]:


df.dropna(how = 'all')


# In[7]:


df['Weapon'].unique()


# In[8]:


df['CrimeCode'].unique()


# What is Crime code? 
# 

# In[9]:


df['Description'].unique()


# Description vs Crime Code?

# In[10]:


df1 = df.iloc[:, 2:5]


# In[11]:


df1.drop('Location', axis =1, inplace = True)


# In[12]:


df2 = df1.groupby(['Description']).agg(['unique'])


# In[13]:


df2


# In[14]:


pd.set_option('display.max_colwidth', -1)


# In[15]:


df2


# In[16]:


df.drop(['CrimeCode'], axis = 1, inplace = True)


# In[17]:


df.rename(columns = {'Inside/Outside': 'In_Out'}, inplace = True)


# In[18]:


df


# In[19]:


df['In_Out'].unique()


# In[20]:


df['In_Out'].replace('Inside','I', regex = True, inplace = True)


# In[21]:


df['In_Out'].replace('Outside','O', regex = True, inplace = True)


# In[22]:


df['In_Out'].unique()


# In[23]:


df["DateTime"] = df["CrimeDate"] +" "+ df["CrimeTime"]


# In[24]:


df['DateTime']=pd.to_datetime(df['DateTime'], infer_datetime_format=True, errors='coerce')


# In[25]:


df.info()


# In[26]:


df


# In[27]:


df['Total Incidents'].unique()


# In[28]:


df.drop('Total Incidents', axis = 1, inplace = True)


# In[29]:


df.drop(['CrimeDate', 'CrimeTime'], axis = 1, inplace = True)


# In[30]:


df


# In[31]:


df['Weapon'].fillna('None or Unknown', inplace = True)


# In[32]:


df['Weapon'].unique()


# In[33]:


df['District'].unique()


# In[34]:


df.loc[df['Location 1'].isnull()]


# Two thousand rows where location is null, this is less than 1 percent of dataset, can drop without losing too much data

# In[35]:


df1 = df.dropna(how = 'any')
df1.shape


# Dropping all NaN results in less than 5% of data lost... This is acceptable due to amount of data available, will drop all
# 

# In[36]:


del df1


# In[37]:


df = df.dropna(how = 'any')


# In[38]:


df['Premise'] = df['Premise'].str.upper()


# In[39]:


df['Premise'].unique()


# In[40]:


df


# In[41]:


crs = {'init': 'epsg:4326'}


# In[42]:


gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude), crs = crs)


# In[43]:


gdf


# Geometry point created, no need to keep longxlat or location 1

# In[44]:


gdf


# In[45]:


fig, ax = plt.subplots(figsize = (10, 10))
sns.countplot(x = 'Description', palette = 'Blues_d', data = gdf, order = df['Description'].value_counts().index)
sns.despine()
plt.xticks(rotation = 45, ha = 'right')


# In[46]:


fig, ax = plt.subplots(figsize = (10, 10))
sns.countplot(x = 'Weapon', data = gdf, order = df['Weapon'].value_counts().index)
sns.despine()
plt.xticks(rotation = 45, ha = 'right')


# In[47]:


fig, ax = plt.subplots(figsize = (10, 10))
sns.countplot(x = 'District', data = gdf,  palette = 'Blues_d', order = df['District'].value_counts().index)
sns.despine()
plt.xticks(rotation = 45, ha = 'right')


# In[48]:


fig, ax = plt.subplots(figsize = (10, 10))
sns.countplot(x = 'District', data = gdf, palette = 'RdBu_d', hue = 'Weapon', order = df['District'].value_counts().index)
sns.despine()
plt.xticks(rotation = 45, ha = 'right')


# In[49]:


baltimore = gpd.read_file('https://data.baltimorecity.gov/api/views/deus-s85f/files/uvSrbSrNNI6kxdQMdA7Rdp9rx-tLl6m1xGI0baktz2Q?filename=Building.zip')


# In[50]:


conda install descartes


# In[52]:


import descartes
from descartes.patch import PolygonPatch


# In[53]:


fig, ax = plt.subplots(figsize = (15,15))
baltimore.plot(ax = ax)


# In[54]:


fig, ax = plt.subplots( figsize = (15,15))
baltimore.plot(ax = ax, color = 'grey', alpha = .4)
gdf[gdf['Description']==('ROBBERY - STREET')].plot(ax = ax, color = 'red', markersize = 1, alpha = .5 )
gdf[gdf['Description']==('SHOOTING')].plot(ax = ax, color = 'green', markersize = 1, alpha = .5)
gdf[gdf['Description']==('RAPE')].plot(ax = ax, color = 'blue', markersize = 1 )


# In[55]:


crimecam = gpd.read_file('https://data.baltimorecity.gov/api/geospatial/jit3-cud7?method=export&format=Shapefile')


# In[56]:


fig, ax = plt.subplots(figsize = (15,15))
crimecam.plot(ax = ax)


# In[57]:


fig, ax = plt.subplots( figsize = (15,15))
crimecam.plot(ax = ax, color = 'blue')
gdf[gdf['Description']==('ROBBERY - STREET')].plot(ax = ax, color = 'red', markersize = 1)
gdf[gdf['Description']==('SHOOTING')].plot(ax = ax, color = 'green', markersize = 1)
gdf[gdf['Description']==('RAPE')].plot(ax = ax, color = 'purple', markersize = 1 )


# In[58]:


street = gpd.read_file('https://data.baltimorecity.gov/api/geospatial/tau7-6emy?method=export&format=Shapefile')


# In[59]:


fig, ax = plt.subplots( figsize = (15,15))
street.plot(ax = ax, color = 'grey', alpha = .4)
gdf.plot(ax = ax, color = 'red', markersize = 1)


# In[60]:


fig, ax = plt.subplots( figsize = (15,15))
street.plot(ax = ax, color = 'grey', alpha = .4)
gdf[gdf['Description']==('ROBBERY - STREET')].plot(ax = ax, color = 'red', markersize = 1)
gdf[gdf['Description']==('SHOOTING')].plot(ax = ax, color = 'green', markersize = 1)
gdf[gdf['Description']==('RAPE')].plot(ax = ax, color = 'purple', markersize = 1 )


# In[61]:


fig, ax = plt.subplots( figsize = (15,15))
crimecam.plot(ax = ax, color = 'blue')
street.plot(ax = ax, color = 'black', alpha = .2)
gdf[gdf['Description']==('ROBBERY - STREET')].plot(ax = ax, color = 'red', markersize = 1)
gdf[gdf['Description']==('SHOOTING')].plot(ax = ax, color = 'green', markersize = 1)
gdf[gdf['Description']==('RAPE')].plot(ax = ax, color = 'purple', markersize = 1 )


# In[62]:


gdf = gdf.set_index('DateTime') 


# In[63]:


gdf.head


# In[64]:


permitdf = pd.read_csv('https://data.baltimorecity.gov/api/views/cdz5-3y2u/rows.csv?accessType=DOWNLOAD')


# In[65]:


permitdf["intermediate_location_text"]= permitdf["intermediate_location_text"].str.split(":", n = 1, expand = True)


# In[78]:


permitdf


# In[79]:


conda install -c conda-forge geopy


# In[80]:


import geopy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter as rt


# In[81]:


locator = Nominatim(user_agent='myGeocoder')
geocode = rt(locator.geocode, min_delay_seconds=1)


# In[82]:


permitdf["intermediate_location_text"]= permitdf["intermediate_location_text"].str.replace('(Closest Intersection)' ,'', case = False)


# In[83]:


permitdf["intermediate_location_text"]= permitdf["intermediate_location_text"].str.replace('(Closest Street)' ,'', case = False)


# In[84]:


permitdf["intermediate_location_text"]= permitdf['intermediate_location_text'].str.replace(r"\(\)","")


# In[85]:


permitdf[permitdf.intermediate_location_text.isnull()]


# In[86]:


permitdf.drop(permitdf.index[28], inplace = True)


# In[87]:


streetdf = permitdf[permitdf.intermediate_location_text.str.contains('.*[a-zA-Z].*')]


# In[88]:


cordf = permitdf[~permitdf.intermediate_location_text.str.contains('.*[a-zA-Z].*')]


# In[89]:


streetdf['location'] = streetdf['intermediate_location_text'].apply(geocode)


# In[90]:


streetdf


# In[91]:


streetdf['point'] = streetdf['location'].apply(lambda loc: tuple(loc.point) if loc else None)


# In[92]:


streetdf = streetdf[streetdf['point'].notnull()]


# In[93]:


streetdf[['latitude', 'longitude', 'altitude']] = pd.DataFrame(streetdf['point'].tolist(), index=streetdf.index)


# In[94]:


streetdf.drop(['altitude', 'point'], axis = 1, inplace = True)


# In[95]:


cordf.shape


# In[96]:


cordf[['latitude','longitude']] = cordf['intermediate_location_text'].str.split(",",expand=True)


# In[97]:


frames = [cordf , streetdf]
eventdf = pd.concat(frames)


# In[98]:


eventdf


# In[99]:


eventdf.info()


# In[100]:


eventdf['latitude'] = pd.to_numeric(eventdf['latitude'])


# In[101]:


eventdf['longitude'] = pd.to_numeric(eventdf['longitude'])


# In[102]:


eventgdf = gpd.GeoDataFrame(eventdf, geometry=gpd.points_from_xy(eventdf.longitude, eventdf.latitude), crs = crs)


# In[103]:


eventgdf


# In[104]:


fig, ax = plt.subplots(figsize = (15,15))
eventgdf.plot(ax = ax, color = 'red')
street.plot(ax = ax, color = 'black', alpha = .2)
crimecam.plot(ax = ax, color = 'green', alpha = .2)


# In[105]:


eventgdf['start_date']=pd.to_datetime(eventgdf['start_date'], infer_datetime_format=True)


# In[106]:


eventgdf['end_date']=pd.to_datetime(eventgdf['end_date'], infer_datetime_format=True)


# In[107]:


eventgdf.info()


# In[108]:


sns.countplot(x = 'duration', data = eventgdf)


# In[109]:


eventgdf[eventgdf.duration != 1]


# In[110]:


eventgdf1 = eventgdf.loc[eventgdf.index.repeat(eventgdf.duration)].copy()
eventgdf1['day_number'] = eventgdf1.groupby(level=0).cumcount()
eventgdf1 = eventgdf1.reset_index(drop=True)


# Events that last for more than one day are listed only once, the code above replicates the events by the amount of days they're listed and creates a column to tell me how many days out from the beginning the event day is
# 

# In[111]:


eventgdf1


# In[112]:


eventgdf1['start_date'] += pd.to_timedelta(eventgdf1['day_number'], unit='D')


# Increment the days by day_number column to get different start dates for each event
# 

# In[113]:


eventgdf1


# In[114]:


from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from shapely.geometry import MultiPoint


# In[115]:


df = pd.DataFrame(gdf)


# In[116]:


gdf


# In[117]:


df_sample = gdf.sample(n=50000)
coords = df_sample[["Longitude", "Latitude"]].values


# In[118]:


db = DBSCAN(eps = .000029, algorithm='ball_tree', metric= 'haversine').fit(np.radians(coords))
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True


# In[119]:


labels = db.labels_
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)


# In[120]:


clusters = pd.Series([coords[labels == n] for n in range(num_clusters)])


# In[121]:


num_clusters


# In[122]:


fig, ax = plt.subplots(figsize = (15,15))
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = coords[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=10)

    xy = coords[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=1)


# In[123]:


shooting_df = gdf[gdf['Description'] == 'SHOOTING']
rape_df = gdf[gdf['Description'] == 'RAPE']
streetrob_df = gdf[gdf['Description'] == 'ROBBERY - STREET']


# In[124]:


coords = shooting_df[["Longitude", "Latitude"]].values


# In[125]:


db = DBSCAN(eps = .000029, algorithm='ball_tree', metric= 'haversine').fit(np.radians(coords))
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True


# In[126]:


labels = db.labels_
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)


# In[127]:


clusters = pd.Series([coords[labels == n] for n in range(num_clusters)])


# In[128]:


num_clusters


# In[129]:


fig, ax = plt.subplots(figsize = (15,15))
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = coords[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=10)
street.plot(ax = ax, color = 'black', alpha = .2)


# In[130]:


coords = rape_df[["Longitude", "Latitude"]].values


# In[131]:


db = DBSCAN(eps = .000029, algorithm='ball_tree', metric= 'haversine').fit(np.radians(coords))
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True


# In[132]:


labels = db.labels_
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)


# In[133]:


clusters = pd.Series([coords[labels == n] for n in range(num_clusters)])


# In[134]:


num_clusters


# In[135]:


fig, ax = plt.subplots(figsize = (15,15))
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = coords[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=10)
street.plot(ax = ax, color = 'black', alpha = .2)


# In[136]:


coords = streetrob_df[["Longitude", "Latitude"]].values


# In[137]:


db = DBSCAN(eps = .000012, algorithm='ball_tree', metric= 'haversine').fit(np.radians(coords))
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True


# In[138]:


labels = db.labels_
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)


# In[139]:


clusters = pd.Series([coords[labels == n] for n in range(num_clusters)])


# In[140]:


num_clusters


# In[141]:


fig, ax = plt.subplots(figsize = (15,15))
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = coords[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=10)
street.plot(ax = ax, color = 'black', alpha = .2)


# In[143]:


#Data timeseries by day simple graph
daycount = gdf.groupby(pd.Grouper(freq ='D')).agg('count')


# In[144]:


daycount


# In[145]:


daycount = daycount['Location']


# In[147]:


daycount.plot( markersize=.01)


# In[148]:


daycount.idxmax()

#Google search shows that this is when the baltimore riots happened


# In[149]:


daycount.nsmallest(5)


# In[150]:


month = gdf.groupby(pd.Grouper(freq ='M')).agg('count')
month = month['Location']


# In[151]:


month.plot( markersize=1)


# In[152]:


month.nsmallest(10)


# In[153]:


week = gdf.groupby(pd.Grouper(freq ='7d')).agg('count')


# In[154]:


week = week['Location']


# In[155]:


week.plot(markersize=1)


# In[156]:


week.nsmallest()


# In[157]:


eventgdf = eventgdf.set_index('start_date') 


# In[158]:


eventdaycount = eventgdf.groupby(pd.Grouper(freq ='D')).agg('count')


# In[159]:


eventdaycount = eventdaycount['intermediate_location_text']


# In[160]:


eventdaycount.plot(markersize=.01)


# In[161]:


daycount2015 = daycount.loc['2015-01-01':'2015-12-31']


# In[162]:


daycount2015.plot( markersize=.01)


# In[506]:


conda install -c conda-forge fbprophet


# In[163]:


import fbprophet
from fbprophet import Prophet


# In[164]:


df_daycount = daycount.reset_index()


# In[165]:


#preparing daycount for use in FB Prophet
df_daycount.rename(columns={"DateTime": "ds", "Location": "y"}, inplace = True)


# In[166]:


m = Prophet()


# In[167]:


m.fit(df_daycount)


# In[168]:


future = m.make_future_dataframe(periods=365)
future.tail()


# In[169]:


forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[170]:


fig1 = m.plot(forecast)


# In[171]:


fig2 = m.plot_components(forecast)


# In[ ]:




