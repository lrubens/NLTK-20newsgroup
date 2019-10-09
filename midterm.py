import time
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import json
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import re
import unicodedata
import string
import gensim
import contractions
import pyLDAvis
import matplotlib
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.datasets import fetch_20newsgroups
from nltk.probability import FreqDist
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words


def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words


def remove_emails(words):
    new_words = []
    for word in words:
        new_word = re.sub(r'\S*@\S*\s?', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    stop_words = set(stopwords.words('english'))
    new_words = [i for i in words if not i in stop_words]
    return new_words

def lemmatize_stemming(text):
    # stemmer = SnowballStemmer('english')
    stemmer = PorterStemmer()
    words = ' '.join(text)
    stems = stemmer.stem(WordNetLemmatizer().lemmatize(words, pos='v'))
    return gensim.utils.simple_preprocess(stems)


def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = PorterStemmer()
    new_words = [stemmer.stem(i) for i in words]
    return new_words


def remove_smallwords(words):
    new_words = [i for i in words if len(i) > 3]
    return new_words


def remove_nonalphabet(words):
    return re.sub('[^A-Za-z]', ' ', words)


def remove_nonenglish(words):
    english_words = set(nltk.corpus.words.words())
    return [words[i] for i in range(len(words)) if words[i] in english_words or not words[i].isalpha()]


def preprocess(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_stopwords(words)
    words = remove_smallwords(words)
    words = remove_emails(words)
    # words = remove_nonenglish(words)
    words = lemmatize_stemming(words)
    return words


def get_data():
    raw_dataset = fetch_20newsgroups(subset='train', categories=["talk.politics.guns", "sci.crypt"], shuffle=True, random_state=42)
    # raw_dataset = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
    dataset = raw_dataset.data
    return dataset


def main():
    raw_data = get_data()
    corpus = []
    for line in raw_data:
        line = remove_nonalphabet(line)
        tokens = gensim.utils.simple_preprocess(line)
        data_chunks = preprocess(tokens)
        corpus.append(data_chunks)
    dictionary = gensim.corpora.Dictionary(corpus)
    bow_corpus = [dictionary.doc2bow(doc) for doc in corpus]
    vocab = list(dictionary.values()) #list of terms in the dictionary
    vocab_tf = [dict(i) for i in bow_corpus]
    vocab_tf = list(pd.DataFrame(vocab_tf).sum(axis=0)) #list of term frequencies
    pairs = [(vocab[i], vocab_tf[i]) for i in range(len(vocab_tf)) if vocab_tf[i] > 2]
    cnt = Counter()
    for word, frequency in pairs:
        cnt[word] = frequency
    wordcloud = WordCloud(width=800, height=600, relative_scaling=.8).generate_from_frequencies(cnt)
    wordcloud.to_file("wordcloud.png")


if __name__ == "__main__":
    main()
