#!/usr/bin/env python
# coding: utf-8

# In[3]:


# import required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
os.environ["OMP_NUM_THREADS"] = '2'


# In[4]:


# import our dataset
dataset = pd.read_csv("cluster_list.csv")


# In[5]:


dataset.head()


# In[6]:


dataset.info()


# In[7]:


dataset.describe()


# In[8]:


sns.pairplot(dataset.iloc[:,[2,3,4]])


# In[9]:


from sklearn.preprocessing import StandardScaler
x = dataset.iloc[:,[3,4]].values
sc_x = StandardScaler()
x = sc_x.fit_transform(x)


# In[10]:


# find optimal number of clustering
from sklearn.cluster import KMeans


# In[11]:


wcss = []
for cluster in range(2,20):
    kmeans= KMeans(n_clusters=cluster, init='k-means++', random_state=38)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(2,20),wcss)
plt.title('The Enblow Method')
plt.xlabel('Number Of Cluster')
plt.ylabel('WCSS')
plt.show()


# In[12]:


# Fitting K-Means to the dataset
kmeans = KMeans (n_clusters = 5, init = 'k-means++', random_state=38)
y_kmeans=kmeans.fit_predict(x)


# In[13]:


plt.figure(figsize=(7,7))
plt.scatter(x[y_kmeans== 0,0], x[y_kmeans==0,1], s=100, c='red', label='Cluster 1')
plt.scatter(x[y_kmeans== 1,0], x[y_kmeans==1,1], s=100, c='blue', label='Cluster 2')
plt.scatter(x[y_kmeans== 2,0], x[y_kmeans==2,1], s=100, c='green', label='Cluster 3')
plt.scatter(x[y_kmeans== 3,0], x[y_kmeans==3,1], s=100, c='yellow', label='Cluster 4')
plt.scatter(x[y_kmeans== 4,0], x[y_kmeans==4,1], s=100, c='brown', label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s=200,c='magenta', label="centroids")
plt.title('Clusters of customers')
plt.xlabel('Buying Rate')
plt.ylabel('Image Price')
plt.legend()
plt.show()


# In[14]:


#Hierarchical Clustering
import scipy.cluster.hierarchy as sch


# In[18]:


plt.figure(figsize=(10,5))
dendrogram = sch.dendrogram(sch.linkage(x,method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()


# In[19]:


from sklearn.cluster import AgglomerativeClustering
hc= AgglomerativeClustering(n_clusters=3, affinity = 'euclidean', linkage='ward')
y_hc=hc.fit_predict(x)


# In[20]:


plt.figure(figsize=(7,7))
plt.scatter(x[y_hc== 0,0], x[y_hc==0,1], s=100, c='red', label='Cluster 1')
plt.scatter(x[y_hc== 1,0], x[y_hc==1,1], s=100, c='blue', label='Cluster 2')
plt.scatter(x[y_hc== 2,0], x[y_hc==2,1], s=100, c='green', label='Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s=200,c='magenta', label="centroids")
plt.title('Clusters of customers')
plt.xlabel('Buying Rate')
plt.ylabel('Image Price')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




