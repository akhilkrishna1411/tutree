#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import preprocessing


# so i have taken a sample dataset from the below given link
# r"F:\Data_Science\Week_13\09-06-2019_PCA & PROJECT 5 TEXT ANALYSIS MODELS\AG\iris.csv
# 

# In[14]:


iris=pd.read_csv(r"F:\Data_Science\Week_13\09-06-2019_PCA & PROJECT 5 TEXT ANALYSIS MODELS\AG\iris.csv")


# In[15]:


iris.head()


# Here am dropping the dependent feature
# 

# In[16]:


iris=iris.drop(["Species"],axis=1)


# In[17]:


iris.head()


# In[18]:


# Create the Scaler object
scaler = preprocessing.StandardScaler()


# why exactly we need standardization??
# answer:standardizing the features will result in with a mean value of 0 and with a standard deviatin of 1 across all features
# 

# In[19]:


# Fit your data on the scaler object
x_std = scaler.fit_transform(iris)


# In[23]:


x_std.shape


# this the shape of the dataset before applying dimensional reduction technique

# In[24]:


x_std.min()


# In[25]:


x_std.max()


# In[26]:


n=x_std.shape[0]


# Calculating covariance gives us the raltionship between 2 indepenedent features
# 

# In[27]:


n


# In[28]:


x_bar=np.mean(x_std)


# In[29]:


x_bar


# Method # 1

# In[47]:


cov=(((x_std-x_bar).T)).dot(x_std-x_bar)/(n-1)


# In[48]:


cov


# Method #2

# In[49]:


cov1=(((x_std.T-x_bar.T))).dot(x_std-x_bar)/(n-1)


# In[50]:


cov1


# Computing eigen values and eigen vectors

# eigen values:
# *it tells us about how much variance or spread of the data obtained by each of the principal component
# *if we are 4 fetaures we will be getting 4 eigen values
# *interseting thing is the eigen value of first feature will be greater than the other and so follows
# *the first eigen value will have maximaum variance
# 
# 
#     

# In[54]:


lamda,w=np.linalg.eig(cov)


# In[56]:


lamda


# eigen vector:
# *it tells us the directions of the vectors 
# *ecah vector is perpendicular to the other

# In[57]:


w


# Reducing the dimensions

# In[59]:


w_trim=w[:,:2]


# In[61]:


w_trim.shape


# Transposed space(T)

# In[62]:


T=x_std.dot(w_trim)


# In[63]:


T.shape


# *till now we have done from scratch but now we dive into this Using pre-defined function
# 

# In[68]:


from sklearn.decomposition import PCA


# In[73]:


pca_data=PCA().fit(x_std)


# In[74]:


(pca_data.explained_variance_ratio_)*100


# In[76]:


pca_data.components_


# In[79]:


pca_data.explained_variance_


# *SVD:- it consists in finding components that are linear combinations of the original coordinates
# The components are ordered in such a way that principal component_1 explains the largest percentage of variance, and principal component-2 has the second largest (that has not been explained by p_1) and so on.

# In[80]:


np.linalg.svd(x_std)


# In[81]:


u,sigma,v=np.linalg.svd(x_std)


# In[82]:


u


# In[83]:


sigma


# In[84]:


##V is nothing but w
v


# In[85]:


u.shape


# In[86]:


sigma.shape


# In[87]:


v.shape


# finally we transformed a dataset which has a shape of (150,4) to (150,2)

# In[ ]:




