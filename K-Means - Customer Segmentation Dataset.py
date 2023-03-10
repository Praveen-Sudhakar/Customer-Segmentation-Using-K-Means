#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Importing necessary packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#Reading the data

df = pd.read_csv("D:\AIML\Dataset\Cust_Segmentation.csv")
df


# In[4]:


df.info()


# In[5]:


#Dropping the 'Object' column

df.drop(['Address'],axis=1,inplace=True)


# In[6]:


df


# In[7]:


#Standard scaling the data & handling null values

from sklearn.preprocessing import StandardScaler

x = df.values[:,1:] #Select all rows from 1st column #IV
x = np.nan_to_num(x) #Filling the null values to zero

clus_data = StandardScaler().fit_transform(x) #Standard scaling the IV

clus_data.shape


# In[8]:


#Modeling

from sklearn.cluster import KMeans

km = KMeans(init = "k-means++",n_clusters=4,n_init=100)

km.fit(x)


# In[9]:


#Generating labels/DV

labels = km.labels_ #DV

print(labels[0:5])
print("No. of classes in labels=",set(labels))


# In[10]:


#Inserting the generated labels/DV into the dataset

df['Class'] = labels

df


# In[11]:


#Plotting the graph of Age vs Income using labels created by K-Means

plt.scatter(df['Age'],df['Income'],c=df['Class'],alpha=0.5)
plt.xlabel("Age")
plt.ylabel("Income")
plt.show()


# In[ ]:





# # Using another dataset

# In[12]:


#Import necessary packages

import numpy as np
import pandas as pd


# In[20]:


#Reading the dataset

daf = pd.read_csv("D:\AIML\Dataset\ChurnData.csv")

daf


# In[21]:


daf.info()


# In[25]:


daf['age'].astype('int')


# In[30]:


x = df.values[:,1:]

x = np.nan_to_num(x)


# In[31]:


#Standard scaling the data

from sklearn.preprocessing import StandardScaler

sca_x = StandardScaler()

sca_x.fit_transform(x)


# In[32]:


#Modeling

from sklearn.cluster import KMeans
k_means = KMeans(init='k-means++',n_clusters=4,n_init=100)
k_means.fit(x)


# In[36]:


#Generating labels

labels = k_means.labels_
print(labels.shape)
print("No. of classes = ",set(labels))


# In[37]:


#Inserting labels into dataset

daf['Class'] = labels

daf.head()


# In[ ]:




