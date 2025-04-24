#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


df = pd.read_csv('cleaned_data_nprices.csv')
print(df.dtypes)
coords = df[['latitude', 'longitude']]

kmeans = KMeans(n_clusters=20, random_state=42)
df['cluster_id']=kmeans.fit_predict(coords)


# In[6]:


#just for visualization

plt.scatter(df['longitude'], df['latitude'], c=df['cluster_id'], cmap='tab20')
plt.xlabel('Longitude')
plt.ylabel('Latitutde')
plt.title('Airbnb Location Clusters')
plt.show()


# In[7]:


# create and return a list of Centroids

def get_centroids(kmeans_model):
    centroids = kmeans_model.cluster_centers_
    centroids_list = [(float(lat), float(lon)) for lat, lon in centroids]
    return centroids_list

centroids_list = get_centroids(kmeans)
print(centroids_list)



#you can save this in a .csv or .txt file


# In[8]:


df['centroid_latitude'] = df['cluster_id'].apply(lambda x: centroids_list[x][0])
df['centroid_longitude'] = df['cluster_id'].apply(lambda x: centroids_list[x][1])


# In[9]:


df.columns


# In[10]:


df.to_csv('final_airbnb_dataset.csv', index=False)

