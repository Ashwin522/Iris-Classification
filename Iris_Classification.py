#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt
iris = pd.read_csv('iris_csv.csv')


# In[7]:


iris.head(15)


# In[9]:


print(len(iris['class']))
for col in iris.columns:
    print(col)
    print(iris.groupby('class').size())


# In[13]:


plt.figure(figsize = (15,10))
plt.subplot(2,2,1)
sns.boxplot(x='class',y='sepallength',data=iris)
plt.subplot(2,2,2)
sns.boxplot(x='class',y='sepalwidth',data=iris)
plt.subplot(2,2,3)
sns.boxplot(x='class',y='petallength',data=iris)
plt.subplot(2,2,4)
sns.boxplot(x='class',y='petalwidth',data=iris)


# In[15]:


iris.isnull().values.any()


# In[16]:


iris.info()


# In[17]:


from sklearn.model_selection import train_test_split
array = iris.values
X = array[:,0:4]
Y = array[:,4]
x_train , x_test , y_train , y_test = train_test_split(X,Y,test_size = 0.3 , random_state = 0)


# In[19]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
svc = SVC(max_iter = 1000 , gamma = 'auto')
svc.fit(x_train , y_train)
y_pred = svc.predict(x_test)
acc_svc = round(accuracy_score(y_pred , y_test),2) * 100
print("Accuracy : ",acc_svc)


# In[24]:


from sklearn.tree import DecisionTreeClassifier
decisiontree = DecisionTreeClassifier(random_state = 0)
decisiontree.fit(x_train , y_train)
y_pred = decisiontree.predict(x_test)
acc_decisiontree = round(accuracy_score(y_pred,y_test),2)*100
print("Accuracy : " , acc_decisiontree)


# In[28]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(max_iter = 1000)
logreg.fit(x_train , y_train)
y_pred = logreg.predict(x_test)
acc_logreg = round(accuracy_score(y_pred , y_test) , 2)*100
print("Accuracy : ",acc_logreg)


# In[ ]:




