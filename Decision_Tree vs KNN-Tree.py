#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd


# In[8]:


df = pd.read_csv('Car.csv')


# In[9]:


df


# In[10]:


df.info()


# In[11]:


import numpy as np
import sklearn as sk 
import matplotlib.pyplot as plt
import seaborn as sns


# In[12]:


df.shape


# In[13]:


df.head()


# In[14]:


df.describe()


# In[15]:


print("Classification of Decision-Tree Algorithm")


# In[27]:


from sklearn.model_selection import train_test_split


# In[28]:


xc=['Price','Speed','Doors','Luggage_boot']
y=['Good','Bad']
all_inputs = df[xc]
all_classes = df['Quality']


# In[42]:


(x_train,x_test,y_train,y_test)=train_test_split(all_inputs,all_classes,train_size=0.7, random_state=0)


# In[104]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train_s=sc.fit_transform(x_train)
x_test_s=sc.fit_transform(x_test)


# In[43]:


from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import seaborn as sns


# In[44]:


classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(x_train,y_train)


# In[90]:


plt.figure(figsize=(15,7.5))
plot_tree(classifier, filled=True,rounded=True,class_names=y,feature_names=xc);


# In[92]:


y_pred=classifier.predict(x_test_s)
print(y_pred)


# In[93]:


print(y_test)


# In[96]:


from sklearn import metrics
score=metrics.accuracy_score(y_test,y_pred)
print('accuracy score:%.2f\n\n'%(score))


# In[97]:


confusion_metrics=metrics.confusion_matrix(y_test,y_pred)
print('Confusion Matrix:')
print(confusion_metrics,'\n\n')
print('--------------------------------------------------------------------------')
result=metrics.classification_report(y_test,y_pred)
print('Classification Report:\n')
print(result)


# In[103]:


visual_matrix=sns.heatmap(confusion_metrics, cmap='BrBG_r', annot=True, fmt='g')
plt.xlabel("predicted class",fontsize=12)
plt.ylabel("Actual class",fontsize=12)
plt.title("Confusion Matrix",fontsize=12)
visual_matrix.xaxis.set_ticklabels(['Good','Bad'])
visual_matrix.yaxis.set_ticklabels(['Good','Bad'])


# In[105]:


print("Classification of KNN Algorithm")


# In[106]:


from sklearn.neighbors import KNeighborsClassifier


# In[131]:


classifier = KNeighborsClassifier(n_neighbors=10,metric='minkowski', p=2)
classifier.fit(x_train_s,y_train)


# In[132]:


y_pred=classifier.predict(x_test_s)
print(y_pred)


# In[133]:


print(y_test)


# In[134]:


score=metrics.accuracy_score(y_test,y_pred)
print('accuracy score:%.2f\n\n'%(score))


# In[135]:


confusion_metrics=metrics.confusion_matrix(y_test,y_pred)
print('Confusion Matrix:')
print(confusion_metrics,'\n\n')
print('--------------------------------------------------------------------------')
result=metrics.classification_report(y_test,y_pred)
print('Classification Report:\n')
print(result)


# In[137]:


visual_matrix=sns.heatmap(confusion_metrics, cmap='flare', annot=True, fmt='d')
plt.xlabel("predicted class",fontsize=12)
plt.ylabel("Actual class",fontsize=12)
plt.title("Confusion Matrix",fontsize=12)
visual_matrix.xaxis.set_ticklabels(['Good','Bad'])
visual_matrix.yaxis.set_ticklabels(['Good','Bad'])


# In[ ]:




