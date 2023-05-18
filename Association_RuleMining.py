#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install apyori')
get_ipython().system('pip install mlxtend')


# In[2]:


import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori
import utils
from mlxtend.frequent_patterns import association_rules
get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


data = pd.read_csv('Heart_disease_rulesmining.csv')


# In[11]:


data.head()


# In[12]:


data.tail()


# In[13]:


data.info()


# In[14]:


data.describe()


# In[15]:


yes=(data=='Yes').sum()
no=(data=='No').sum()
disease = pd.concat([yes,no],axis=1, keys=['yes','no'])
ax=disease.plot.bar(stacked=True)
plt.show()


# In[21]:


transaction= utils.data_prepare(data)


# In[25]:



rule_mining = list(apriori(transaction, min_support=0.02,min_confidence=0.2))
association_rules = utils.extract(rule_mining)


# In[26]:


rules_df=pd.DataFrame(association_rules,columns=['LHS','RHS','Support','Confidence','Lift'])
len(rules_df)


# In[27]:


rules_df.nlargest(10,"Lift")


# In[29]:


rules_df.nlargest(10,"Support")


# In[31]:


rules_df.nlargest(10,"Confidence")


# In[33]:


rules_df[rules_df['LHS'].apply(lambda x: len(x)>0)].nlargest(10,"Support")


# In[34]:


rule_mining = list(apriori(transaction, min_support=0.02,min_confidence=0.2,max_length=3))
association_rules = utils.extract(rule_mining)
rules_df=pd.DataFrame(association_rules,columns=['LHS','RHS','Support','Confidence','Lift'])
len(rules_df)


# In[35]:


rules_df.nlargest(10,"Lift")


# In[36]:


rules_df.nlargest(10,"Support")


# In[37]:


rules_df.nlargest(10,"Confidence")


# In[38]:


ax=disease.plot.bar()
plt.show()


# In[91]:


rule_mining = list(apriori(transaction, min_support=0.1,min_confidence=0.95))
association_rules = utils.extract(rule_mining,'Chest Pain Type',2)
utils.inspect(association_rules)


# In[93]:


rule_mining = list(apriori(transaction, min_support=0.1,min_confidence=0.05))
association_rules = utils.extract(rule_mining,'Chest Pain Type',2)
utils.inspect(association_rules)


# In[ ]:




