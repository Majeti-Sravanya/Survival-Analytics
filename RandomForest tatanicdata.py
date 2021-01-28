#!/usr/bin/env python
# coding: utf-8

# In[163]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report 


# In[ ]:





# In[ ]:





# In[135]:


data=pd.read_csv("C:/Users/DELL/Desktop/svn/train.csv")


# In[136]:


data.head(5)


# In[137]:


data.isna().sum()


# In[138]:


data = data.fillna({"Embarked": "S"})
data = data.fillna({"Cabin": "C123"})


# In[139]:


data.fillna(data.mean(), inplace=True)


# In[140]:


data.isna().sum()


# In[48]:


data.dtypes


# In[143]:


le = preprocessing.LabelEncoder()
data['Cabin']=le.fit_transform(data['Cabin'])
data['Sex']=le.fit_transform(data['Sex'])
data['Embarked']=le.fit_transform(data['Embarked'])


# In[144]:


data.head(5)


# In[154]:


data_cols=['PassengerId','Pclass','Age','Sex','SibSp','Cabin','Embarked','Parch']
target=['Survived']


# In[155]:


X=data[data_cols]
y=data[target]


# In[156]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[160]:


model = RandomForestClassifier()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
accuracy_score(y_test.values, y_predict)


# In[161]:


confusion_matrix(y_test.values, y_predict)


# In[165]:


print(classification_report(y_test.values, y_predict))


# In[166]:


model1 = RandomForestClassifier(n_estimators=8,max_depth=2,min_samples_split=4)
model1.fit(X_train, y_train)
y_predict1 = model.predict(X_test)
accuracy_score(y_test.values, y_predict1)


# In[ ]:




