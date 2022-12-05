#!/usr/bin/env python
# coding: utf-8

# # The Sparks Foundation: Data Science and Business Analytics Intership
# ##  Prediction Using Unsupervised MLProblem 
# ### Statement:From the given iris dataset predict the optimum number of clusters and represent it visually
# ### Author: Akshata Nagnath Nichal

# ## Step 1:Importing The Dataset

# In[1]:


# importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


# In[2]:


# Load the dataset
data=pd.read_csv("C:/Users/91986/Downloads/Iris (1).csv")
data.head() #To show the first five coulmns of data set


# In[3]:


data.drop("Id",axis=1,inplace=True)


# ## Step2: Data Wrangling

# In[4]:


data.describe()


# In[5]:


data.info()


# In[6]:


data.Species.value_counts()


# ## Step 3:Using the Elow Method to find the optimal number of clusters

# In[12]:


# Find the optimal number of clusters for k means clustering
x=data.iloc[:,:-1].values
from sklearn.cluster import KMeans
WCSS=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',
                 max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    WCSS.append(kmeans.inertia_)
plt.plot(range(1,11), WCSS)
plt.title('The Elow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# ### We choose the number of cluster as '3'

# ## Step 4: Trainig the kmeans Model on the dataset

# In[24]:


# Apply kmeans to the dataset
kmeans=KMeans(n_clusters=3,
            max_iter=300,n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(x)
y_kmeans


# ## Step5: Visualize the test set result

# In[25]:


plt.scatter(x[y_kmeans==0,0], x[y_kmeans==0,1],
           s=100,c='red',label='Iris-setosa')
plt.scatter(x[y_kmeans==1,0], x[y_kmeans==1,1],
           s=100,c='blue',label='Iris-versicolour')
plt.scatter(x[y_kmeans==2,0], x[y_kmeans==2,1],
           s=100,c='green',label='Iris-virginica')
    
# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
           s=100,c='yellow',label='Centroids')
plt.legend()


# ## Data Visualisation

# In[26]:


data.corr()


# In[27]:


plt.figure(figsize=(12,5))
sns.heatmap(data.corr(),annot=True,cmap="BuPu")


# In[28]:


plt.figure(figsize=(8,6))
sns.boxplot(x="Species",y="SepalLengthCm",data=data)


# In[29]:


sns.pairplot(data.corr())


# ## Thank you!!!
