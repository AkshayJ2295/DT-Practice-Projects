#!/usr/bin/env python
# coding: utf-8

# to find the type of Mushroom as edible{0} or poisonous{1}

# In[17]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv('Mushroom dataset.csv')


# In[3]:


df


# In[7]:


df=pd.DataFrame(data=df)
df.head()


# In[8]:


df.keys()


# In[11]:


df.info()


# In[13]:


df.describe()


# In[14]:


df.shape


# In[15]:


df.isnull().sum()


# In[18]:


sns.countplot(df['class'])


# In[28]:


plt.figure(figsize=(14,6))
g=sns.catplot(x="cap-shape",hue="cap-color",col="class",
             data=df,kind="count")
plt.show()


# 1. white color and bell shaped mushrooms are highly recommended for eating
# 2. Red coloured knob shaped mushrooms are poisonous.

# In[29]:


##Data differentiation between odor and bruises
plt.figure(figsize=(14,6))
g=sns.catplot(x="odor",hue="bruises",col="class",
             data=df,kind="count")
plt.show()


# In[30]:


plt.figure(figsize=(14,6))
g=sns.catplot(x="gill-size",hue="gill-color",col="class",
             data=df,kind="count")
plt.show()


# In[34]:


#converting all columns of dataset into integer type to perform furthet operations
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df=df.apply(LabelEncoder().fit_transform)
df.head()


# In[35]:


#split the data into x and y to perform train_test_split
x=df.drop(['class'],axis=1)
y=df['class']


# In[38]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


# In[39]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3,random_state=42)


# In[42]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[43]:


dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
preddt=dt.predict(x_test)
print('accuracy score:',)
print(accuracy_score(y_test,preddt))
print(confusion_matrix(y_test,preddt))
print(classification_report(y_test,preddt))


# In[44]:


gnb=GaussianNB()
gnb.fit(x_train,y_train)
predgnb=gnb.predict(x_test)
print('accuracy score:',)
print(accuracy_score(y_test,predgnb))
print(confusion_matrix(y_test,predgnb))
print(classification_report(y_test,predgnb))


# In[45]:


rf=RandomForestClassifier()
rf.fit(x_train,y_train)
predrf=rf.predict(x_test)
print('accuracy score:',)
print(accuracy_score(y_test,predrf))
print(confusion_matrix(y_test,predrf))
print(classification_report(y_test,predrf))


# In[46]:


#for better understanding, lets do cross validation
from sklearn.model_selection import cross_val_score


# In[48]:


score=cross_val_score(dt,x,y,cv=5)
print(score)
print(score.mean())


# In[49]:


score=cross_val_score(gnb,x,y,cv=5)
print(score)
print(score.mean())


# In[50]:


score=cross_val_score(rf,x,y,cv=5)
print(score)
print(score.mean())


# Minimum difference in cross validation and accuracy score is for DecisiontreeClassifier so this is our best model.

# In[51]:


#saving the best results.

import joblib


# In[52]:


joblib.dump(dt,"dtmodel.obj")


# In[53]:


dtfile=joblib.load("dtmodel.obj")
dtfile.predict(x_train)


# In[ ]:




