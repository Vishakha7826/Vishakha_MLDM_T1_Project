#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sns
from wordcloud import WordCloud
import nltk


# In[2]:


data = pd.read_csv('tourist_accommodation_reviews.csv',encoding='windows-1252')


# In[3]:


data.head()


# In[4]:


data['Location'].unique()


# In[5]:


data_count = data.groupby(['Location']).Review.nunique()
data_count


# In[6]:


data.sort_values("Location",inplace=True)
edit = data["Location"]==' Phuket Town'
data.where(edit, inplace=True)
data


# In[7]:


visitor_place = data.dropna()


# In[8]:


visitor_place


# In[9]:


orderHotelByLocation= visitor_place.groupby(['Hotel/Restaurant name']).Review.nunique()
orderHotelByLocation


# In[10]:


selectedHotels1 = orderHotelByLocation.nlargest(30)
selectedHotels1


# In[11]:


finalList =list(selectedHotels1.index)
finalList


# In[12]:


chosenHotels = data[data['Hotel/Restaurant name'].isin(finalList)]
chosenHotels


# In[13]:


nltk.download(['stopwords','punkt','wordnet','omw-1.4','vader_lexicon'])
get_ipython().run_line_magic('matplotlib', 'inline')


# In[14]:


stop_words = nltk.corpus.stopwords.words('english')
print(stop_words)


# In[15]:


def preprocess_text(text):
    tokenized_document = nltk.tokenize.RegexpTokenizer('[a-zA-Z0-9\']+').tokenize(text)
    cleaned_tokens = [word.lower() for word in tokenized_document if word.lower() not in stop_words]
    stemmed_text = [nltk.stem.PorterStemmer().stem(word) for word in cleaned_tokens]
    return stemmed_text


# In[18]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
sent=SentimentIntensityAnalyzer()


# In[20]:


chosenHotels['compound'] = [sent.polarity_scores(review)['compound'] for review in chosenHotels['Review']]
chosenHotels['neg'] = [sent.polarity_scores(review)['neg'] for review in chosenHotels['Review']]
chosenHotels['neu'] = [sent.polarity_scores(review)['neu'] for review in chosenHotels['Review']]
chosenHotels['pos'] = [sent.polarity_scores(review)['pos'] for review in chosenHotels['Review']]


# In[21]:


chosenHotels[['compound','neg','neu','pos']].describe()


# In[22]:


sns.histplot(chosenHotels['compound'])


# In[23]:


sns.histplot(chosenHotels['pos'])


# In[24]:


sns.histplot(chosenHotels['neg'])


# In[25]:


sns.histplot(chosenHotels['neu'])


# In[26]:


(chosenHotels['compound']<=0).groupby(chosenHotels['Hotel/Restaurant name']).sum()


# In[27]:


percent_negative = pd.DataFrame((chosenHotels['compound']<=0).groupby(chosenHotels['Hotel/Restaurant name']).sum()
                               /chosenHotels['Hotel/Restaurant name'].groupby(chosenHotels['Hotel/Restaurant name']).count()*100,
                               columns=['% negative reviews']).sort_values(by='% negative reviews')
percent_negative


# In[28]:


sns.barplot(data=percent_negative, x='% negative reviews', y=percent_negative.index, color='b')


# In[29]:


chosenHotels['processed_review'] = chosenHotels['Review'].apply(preprocess_text)
positive_reviews = chosenHotels.loc[(chosenHotels['Hotel/Restaurant name'] == 'Beach Bar') & (chosenHotels['compound']>0),:]
negative_reviews = chosenHotels.loc[(chosenHotels['Hotel/Restaurant name'] == 'Beach Bar' ) & (chosenHotels['compound']<=0),:]
negative_reviews.head()


# In[30]:


neg_tokens = [word for review in negative_reviews['processed_review'] for word in review]
wordcloud= WordCloud(background_color='white').generate_from_text(' '.join(neg_tokens))
plt.figure(figsize=(12,12))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[31]:


pos_tokens = [word for review in positive_reviews['processed_review'] for word in review]
wordcloud= WordCloud(background_color='yellow').generate_from_text(' '.join(pos_tokens))
plt.figure(figsize=(12,12))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[ ]:




