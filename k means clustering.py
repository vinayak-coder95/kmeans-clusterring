#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import datasets


# In[2]:


iris = datasets.load_iris()
iris


# In[3]:


iris1 = pd.DataFrame(iris.data,columns=iris.feature_names)


# In[4]:


iris1.head()


# In[5]:


plt.scatter(iris1['sepal length (cm)'],iris1['petal width (cm)'],color=('red'))


# In[6]:


iris1.drop(['sepal width (cm)','petal length (cm)'],axis=1,inplace=True)


# In[7]:


iris1


# In[71]:


from sklearn.cluster import KMeans
for k in range(1,4):
    km = KMeans(k)
    prediction = km.fit_predict(iris1[['sepal length (cm)','petal width (cm)']])
prediction


# In[72]:


iris1['prediction']=prediction


# In[73]:


iris1['prediction'].values


# In[74]:


km.cluster_centers_


# In[75]:


df = iris1


# In[76]:


df


# In[77]:


df1 = df[df.prediction==0]
df2 = df[df.prediction==1]
df3 = df[df.prediction==2]
df4 = df[df.prediction==3]
df5 = df[df.prediction==4]
df6 = df[df.prediction==5]
df7 = df[df.prediction==6]
df8 = df[df.prediction==7]
df9 = df[df.prediction==8]
plt.scatter(df1['sepal length (cm)'],df1['petal width (cm)'],color='green')
plt.scatter(df2['sepal length (cm)'],df2['petal width (cm)'],color='red')
plt.scatter(df3['sepal length (cm)'],df3['petal width (cm)'],color='black')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.legend()


# In[63]:


sse = []
k_rng = range(1,10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['sepal length (cm)','petal width (cm)']])
    sse.append(km.inertia_)

plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)


# In[ ]:





# In[ ]:




