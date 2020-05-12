#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install kaggle')


# In[1]:


get_ipython().system('ls')


# In[2]:


get_ipython().system('mkdir .kaggle')


# In[3]:


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


baltimore = gpd.read_file('https://jscholarship.library.jhu.edu/bitstream/handle/1774.2/32834/StreetMap2000.shp.zip?sequence=1&isAllowed=y')


# In[50]:


conda install descartes


# In[51]:


import descartes
from descartes.patch import PolygonPatch


# In[52]:


fig, ax = plt.subplots(figsize = (15,15))
baltimore.plot(ax = ax)


# In[53]:


fig, ax = plt.subplots( figsize = (15,15))
baltimore.plot(ax = ax, color = 'grey', alpha = .4)
gdf[gdf['Description']==('ROBBERY - STREET')].plot(ax = ax, color = 'red', markersize = 1, alpha = .5 )
gdf[gdf['Description']==('SHOOTING')].plot(ax = ax, color = 'green', markersize = 1, alpha = .5)
gdf[gdf['Description']==('RAPE')].plot(ax = ax, color = 'blue', markersize = 1 )


# In[54]:


crimecam = gpd.read_file('https://data.baltimorecity.gov/api/geospatial/jit3-cud7?method=export&format=Shapefile')


# In[55]:


fig, ax = plt.subplots(figsize = (15,15))
crimecam.plot(ax = ax)


# In[56]:


fig, ax = plt.subplots( figsize = (15,15))
crimecam.plot(ax = ax, color = 'blue')
gdf[gdf['Description']==('ROBBERY - STREET')].plot(ax = ax, color = 'red', markersize = 1)
gdf[gdf['Description']==('SHOOTING')].plot(ax = ax, color = 'green', markersize = 1)
gdf[gdf['Description']==('RAPE')].plot(ax = ax, color = 'purple', markersize = 1 )


# In[57]:


street = gpd.read_file('https://data.baltimorecity.gov/api/geospatial/tau7-6emy?method=export&format=Shapefile')


# In[58]:


fig, ax = plt.subplots( figsize = (15,15))
street.plot(ax = ax, color = 'grey', alpha = .4)
gdf.plot(ax = ax, color = 'red', markersize = 1)


# In[59]:


fig, ax = plt.subplots( figsize = (15,15))
street.plot(ax = ax, color = 'grey', alpha = .4)
gdf[gdf['Description']==('ROBBERY - STREET')].plot(ax = ax, color = 'red', markersize = 1)
gdf[gdf['Description']==('SHOOTING')].plot(ax = ax, color = 'green', markersize = 1)
gdf[gdf['Description']==('RAPE')].plot(ax = ax, color = 'purple', markersize = 1 )


# In[60]:


fig, ax = plt.subplots( figsize = (15,15))
crimecam.plot(ax = ax, color = 'blue')
street.plot(ax = ax, color = 'black', alpha = .2)
gdf[gdf['Description']==('ROBBERY - STREET')].plot(ax = ax, color = 'red', markersize = 1)
gdf[gdf['Description']==('SHOOTING')].plot(ax = ax, color = 'green', markersize = 1)
gdf[gdf['Description']==('RAPE')].plot(ax = ax, color = 'purple', markersize = 1 )


# In[61]:


gdf = gdf.set_index('DateTime') 


# In[62]:


gdf.head


# In[63]:


permitdf = pd.read_csv('https://data.baltimorecity.gov/api/views/cdz5-3y2u/rows.csv?accessType=DOWNLOAD')


# In[64]:


permitdf["intermediate_location_text"]= permitdf["intermediate_location_text"].str.split(":", n = 1, expand = True)


# In[65]:


permitdf


# In[67]:


import geopy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter as rt


# In[68]:


locator = Nominatim(user_agent='myGeocoder')
geocode = rt(locator.geocode, min_delay_seconds=1)


# In[69]:


permitdf["intermediate_location_text"]= permitdf["intermediate_location_text"].str.replace('(Closest Intersection)' ,'', case = False)


# In[70]:


permitdf["intermediate_location_text"]= permitdf["intermediate_location_text"].str.replace('(Closest Street)' ,'', case = False)


# In[71]:


permitdf["intermediate_location_text"]= permitdf['intermediate_location_text'].str.replace(r"\(\)","")


# In[72]:


permitdf[permitdf.intermediate_location_text.isnull()]


# In[73]:


permitdf.drop(permitdf.index[28], inplace = True)


# In[74]:


streetdf = permitdf[permitdf.intermediate_location_text.str.contains('.*[a-zA-Z].*')]


# In[75]:


cordf = permitdf[~permitdf.intermediate_location_text.str.contains('.*[a-zA-Z].*')]


# In[76]:


streetdf['location'] = streetdf['intermediate_location_text'].apply(geocode)


# In[77]:


streetdf


# In[78]:


streetdf['point'] = streetdf['location'].apply(lambda loc: tuple(loc.point) if loc else None)


# In[79]:


streetdf = streetdf[streetdf['point'].notnull()]


# In[80]:


streetdf[['latitude', 'longitude', 'altitude']] = pd.DataFrame(streetdf['point'].tolist(), index=streetdf.index)


# In[81]:


streetdf.drop(['altitude', 'point'], axis = 1, inplace = True)


# In[82]:


cordf.shape


# In[83]:


cordf[['latitude','longitude']] = cordf['intermediate_location_text'].str.split(",",expand=True)


# In[84]:


frames = [cordf , streetdf]
eventdf = pd.concat(frames)


# In[85]:


eventdf


# In[86]:


eventdf.info()


# In[87]:


eventdf['latitude'] = pd.to_numeric(eventdf['latitude'])


# In[88]:


eventdf['longitude'] = pd.to_numeric(eventdf['longitude'])


# In[89]:


eventgdf = gpd.GeoDataFrame(eventdf, geometry=gpd.points_from_xy(eventdf.longitude, eventdf.latitude), crs = crs)


# In[90]:


eventgdf


# In[91]:


fig, ax = plt.subplots(figsize = (15,15))
eventgdf.plot(ax = ax, color = 'red')
street.plot(ax = ax, color = 'black', alpha = .2)
crimecam.plot(ax = ax, color = 'green', alpha = .2)


# In[92]:


eventgdf['start_date']=pd.to_datetime(eventgdf['start_date'], infer_datetime_format=True)


# In[93]:


eventgdf['end_date']=pd.to_datetime(eventgdf['end_date'], infer_datetime_format=True)


# In[94]:


eventgdf.info()


# In[95]:


sns.countplot(x = 'duration', data = eventgdf)


# In[96]:


eventgdf[eventgdf.duration != 1]


# In[97]:


eventgdf1 = eventgdf.loc[eventgdf.index.repeat(eventgdf.duration)].copy()
eventgdf1['day_number'] = eventgdf1.groupby(level=0).cumcount()
eventgdf1 = eventgdf1.reset_index(drop=True)


# Events that last for more than one day are listed only once, the code above replicates the events by the amount of days they're listed and creates a column to tell me how many days out from the beginning the event day is
# 

# In[98]:


eventgdf1


# In[99]:


eventgdf1['start_date'] += pd.to_timedelta(eventgdf1['day_number'], unit='D')


# Increment the days by day_number column to get different start dates for each event
# 

# In[100]:


eventgdf1


# In[101]:


from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from shapely.geometry import MultiPoint


# In[102]:


df = pd.DataFrame(gdf)


# In[103]:


gdf


# In[104]:


df_sample = gdf.sample(n=50000)
coords = df_sample[["Longitude", "Latitude"]].values


# In[105]:


db = DBSCAN(eps = .000029, algorithm='ball_tree', metric= 'haversine').fit(np.radians(coords))
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True


# In[106]:


labels = db.labels_
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)


# In[107]:


clusters = pd.Series([coords[labels == n] for n in range(num_clusters)])


# In[108]:


num_clusters


# In[109]:


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


# In[110]:


shooting_df = gdf[gdf['Description'] == 'SHOOTING']
rape_df = gdf[gdf['Description'] == 'RAPE']
streetrob_df = gdf[gdf['Description'] == 'ROBBERY - STREET']


# In[111]:


coords = shooting_df[["Longitude", "Latitude"]].values


# In[112]:


db = DBSCAN(eps = .000029, algorithm='ball_tree', metric= 'haversine').fit(np.radians(coords))
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True


# In[113]:


labels = db.labels_
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)


# In[114]:


clusters = pd.Series([coords[labels == n] for n in range(num_clusters)])


# In[115]:


num_clusters


# In[116]:


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


# In[117]:


coords = rape_df[["Longitude", "Latitude"]].values


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
street.plot(ax = ax, color = 'black', alpha = .2)


# In[123]:


coords = streetrob_df[["Longitude", "Latitude"]].values


# In[124]:


db = DBSCAN(eps = .000012, algorithm='ball_tree', metric= 'haversine').fit(np.radians(coords))
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True


# In[125]:


labels = db.labels_
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)


# In[126]:


clusters = pd.Series([coords[labels == n] for n in range(num_clusters)])


# In[127]:


num_clusters


# In[128]:


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


# In[129]:


#Data timeseries by day simple graph
daycount = gdf.groupby(pd.Grouper(freq ='D')).agg('count')


# In[130]:


daycount


# In[131]:


daycount = daycount['Location']


# In[132]:


daycount.plot( markersize=.01)


# In[133]:


daycount.idxmax()

#Google search shows that this is when the baltimore riots happened


# In[134]:


daycount.nsmallest(5)


# In[135]:


month = gdf.groupby(pd.Grouper(freq ='M')).agg('count')
month = month['Location']


# In[136]:


month.plot( markersize=1)


# In[137]:


month.nsmallest(10)


# In[138]:


week = gdf.groupby(pd.Grouper(freq ='7d')).agg('count')


# In[139]:


week = week['Location']


# In[140]:


week.plot(markersize=1)


# In[141]:


week.nsmallest()


# In[261]:


eventgdf = eventgdf.set_index('start_date') 


# In[143]:


eventdaycount = eventgdf.groupby(pd.Grouper(freq ='D')).agg('count')


# In[144]:


eventdaycount = eventdaycount['intermediate_location_text']


# In[145]:


eventdaycount.plot(markersize=.01)


# In[146]:


daycount2015 = daycount.loc['2015-01-01':'2015-12-31']


# In[147]:


daycount2015.plot( markersize=.01)


# In[280]:


import fbprophet
from fbprophet import Prophet
import utils


# In[149]:


df_daycount = daycount.reset_index()


# In[150]:


#preparing daycount for use in FB Prophet
df_daycount.rename(columns={"DateTime": "ds", "Location": "y"}, inplace = True)


# In[167]:


p = Prophet()


# In[168]:


p.fit(df_daycount)


# In[203]:


future = p.make_future_dataframe(periods=365)
future.tail()


# In[205]:


forecast = p.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[206]:


fig1 = p.plot(forecast)


# In[207]:


fig2 = p.plot_components(forecast)


# In[159]:


gdf.Neighborhood.unique()


# In[160]:


sns.countplot(x = 'Neighborhood', data = gdf)


# In[163]:


gdf.Neighborhood.value_counts()


# In[164]:


#shooting_df = gdf[gdf['Description'] == 'SHOOTING']
#rape_df = gdf[gdf['Description'] == 'RAPE']
#streetrob_df = gdf[gdf['Description'] == 'ROBBERY - STREET']

shooting_df.Neighborhood.value_counts()


# In[165]:


rape_df.Neighborhood.value_counts()


# In[166]:


streetrob_df.Neighborhood.value_counts()


# In[177]:


from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics


# In[229]:


df_cv = cross_validation(p, initial='1095 days', horizon = '180 days')
df_cv.head()


# In[230]:


df_p = performance_metrics(df_cv)
df_p.head()


# In[217]:


eventdaycount
eventdaycount.rename(columns={"DateTime": "ds", "Location": "y"}, inplace = True)


# In[220]:


m = Prophet()
model_w_holidays= m.add_country_holidays(country_name = 'US')


# In[221]:


m.fit(df_daycount)


# In[222]:


mfuture = m.make_future_dataframe(periods=365)


# In[223]:


mforecast = m.predict(future)
mforecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[231]:


mdf_cv = cross_validation(m, initial='1095 days', horizon = '180 days')


# In[232]:


mdf_cv.head()


# In[233]:


mdf_p = performance_metrics(mdf_cv)
mdf_p.head()


# In[237]:


eventdaycount


# In[252]:


df2k15 = df_daycount[df_daycount.ds.between('2015-01-01', '2015-12-30')]


# In[256]:


df2k15 = df2k15.reset_index()


# In[266]:


evendf = eventdaycount.to_frame()


# In[321]:


evendf =evendf.reset_index()


# In[322]:


future_even_df = evendf


# In[323]:


future_even_df['start_date'] = evendf['start_date'].apply(lambda x: x + pd.DateOffset(years=1))


# In[324]:


evendf = evendf.set_index('start_date')


# In[325]:


future_even_df


# In[284]:


t15 = df2k15.set_index('ds')


# In[290]:


t15 = t15.drop('index', 1)


# In[294]:


regressordf = t15.join(evendf)


# In[295]:


regressordf


# In[296]:


regressordf['intermediate_location_text'] = regressordf['intermediate_location_text'].fillna(0.0)


# In[298]:


reg_df = regressordf.rename(columns={'intermediate_location_text': 'event'})


# In[301]:


reg_df = reg_df.reset_index()


# In[302]:


reg_df


# In[303]:


x = Prophet()
x.add_regressor('event')
x.fit(reg_df)


# In[329]:


f = eventdaycount.to_frame()


# In[331]:


f = f.reset_index()


# In[333]:


f = f.intermediate_location_text


# In[336]:


xfuture = x.make_future_dataframe(periods=365)
xfuture['event']=f


# In[344]:


xfuture['event'] = xfuture['event'].fillna(0.0)


# In[345]:


xforecast = x.predict(xfuture)
xforecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[347]:


xdf_cv = cross_validation(x, horizon = '30 days')


# In[348]:


xdf_cv.head()


# In[349]:


xdf_p = performance_metrics(mdf_cv)
xdf_p.head()


# In[355]:


fig1 = m.plot(mforecast)


# In[353]:


fig1 = x.plot(xforecast)


# In[354]:


fig1 = p.plot(forecast)


# In[356]:


xy = Prophet()
model_w_holidays= xy.add_country_holidays(country_name = 'US')


# In[359]:


t15 = t15.reset_index()


# In[360]:


xy.fit(t15)


# In[362]:


xyfuture = xy.make_future_dataframe(periods=)


# In[364]:


xyforecast = xy.predict(xyfuture)
xyforecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[365]:


xydf_cv = cross_validation(xy, horizon = '30 days')


# In[366]:


fig1 = xy.plot(xyforecast)


# In[367]:


xydf_p = performance_metrics(xydf_cv)
xydf_p.head()


# In[ ]:




