#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[2]:


df=pd.read_csv('World Happiness Report.csv')
df.head()


# In[3]:


df.shape


# In[4]:


df.columns


# In[5]:


df.dtypes


# In[6]:


#dropping two columns 
df.drop(['Country','Region','Happiness Rank'],axis=1,inplace=True)


# In[7]:


df


# Visualization

# In[8]:


#lets check the correlation between the columns
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(),annot=True,vmin=-1.0,vmax=1.0)
plt.show()


# In[9]:


df.describe()


# In[10]:


df.boxplot(column="Standard Error")


# In[11]:


#plotting to check outliers for data cleaning
df.boxplot(figsize=(20,8))
plt.subplots_adjust(bottom=0.25)
plt.show()


# In[12]:


df.skew()


# In[13]:


from scipy.stats import zscore
z=np.abs(zscore(df))
threshold=3
print(np.where(z>3))


# In[14]:


z[40][1]


# In[15]:


z[157][3]


# In[19]:


###lets check the presence of outliers


dfnew=df[(z<3).all(axis=1)]


# In[20]:


#shape after removing outliers
dfnew.shape


# In[21]:


df.shape


# In[22]:


df=dfnew


# In[23]:


df


# In[24]:


#spliting the data into x and y
y=df['Happiness Score']
x=df.drop('Happiness Score',axis=1)


# In[25]:


x


# In[26]:


y


# In[27]:


x.skew()


# In[29]:


#lets remove the skewness
from sklearn.preprocessing import power_transform
x=power_transform(x,method='yeo-johnson')


# In[30]:


x


# In[33]:


#Scaling of data
sc=StandardScaler()
x=sc.fit_transform(x)
x


# In[42]:


#train_test_split to perform further operations
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.40,random_state=50)


# In[43]:


lm=LinearRegression()
lm.fit(x_train,y_train)


# In[44]:


lm.coef_


# In[45]:


lm.intercept_


# In[41]:


lm.score(x_train,y_train)


# In[46]:


pred=lm.predict(x_test)


# In[50]:


print("predicted Happiness Score:",pred)


# In[51]:


from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score


# In[53]:


print('error:')
print('mean absolute error:',mean_absolute_error(y_test,pred))
print('Mean squared error:',mean_squared_error(y_test,pred))
print(r2_score(y_test,pred))


# In[ ]:




