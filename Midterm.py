#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib as plt
import re
import math
from wordcloud import WordCloud
import pyLDAvis

from pprint import pprint

#SKYLEARN
from sklearn.datasets import fetch_20newsgroups 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

#NTLK
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('words', quiet=True)
from nltk.tokenize import regexp_tokenize, wordpunct_tokenize, RegexpTokenizer
from nltk.tokenize import sent_tokenize, PunktSentenceTokenizer, WhitespaceTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn 
from nltk.corpus import stopwords

#CORPORA
import gensim
from gensim.corpora import Dictionary
from gensim.models import TfidfModel

print('All libraries imported successfully')


# In[2]:


def preprocess( data ):
    newdata = []
    for idx, x in enumerate(data): #x -- document in data 
        newdoc = ""
        stop_words = stopwords.words('english')
        words = set(nltk.corpus.words.words())

        for token in sent_tokenize(x): #token -- sentence in doc
            seperator = " " 
            token = re.sub('[^A-Za-z0-9]+', ' ', token) #remove special characters
            token = re.sub(r'\d+', '', token) #remove numbers
            token = re.sub(r'\b\w{1,2}\b', '', token) #remove words with <= 2 characters
            #         if w.lower() in words or not w.isalpha())
            whitespace_token = WhitespaceTokenizer().tokenize( token )
            wo_stopwords_token = [x for x in whitespace_token
                                  if not x in stop_words]
            newdoc += seperator.join( (wo_stopwords_token) ).lower()
            newdoc += " " 
            
        # Word Tokens for Each Document
        word_tokens = RegexpTokenizer('\s+', gaps=True).tokenize(newdoc)
        data[idx] = word_tokens 
        newdata.append( data[idx] )
        
    return newdata


# In[3]:


# Import and clean data
chosen_category = 'sci.space'
print("Loading 20 newsgroups dataset for category " + chosen_category)
newsgroups_train = fetch_20newsgroups(subset='train', categories=[chosen_category],
                                      remove=('headers', 'footers', 'quotes'))
newdata = preprocess(newsgroups_train.data)
pprint("Data Cleaned")


# In[4]:


# Actual Bag of Words and LDA
dictionary = gensim.corpora.Dictionary(newdata)
bow_corpus = [dictionary.doc2bow(doc) for doc in newdata]
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics = 5, id2word = dictionary, passes = 10, workers = 2)
pprint(lda_model.print_topics())


# In[179]:


type(lda_model)


# In[ ]:




