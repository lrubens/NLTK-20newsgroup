import time
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


def remove_tabs(words):
    tabs = str.maketrans('', '', '\t');
    return [word.translate(tabs) for word in words]


def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words


def replace_contractions(words):
    new_words = []
    for word in words:
        new_words.append(contractions.fix(word))
    return new_words


def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words


def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    tokenizer = RegexpTokenizer(r'\w+')
    for word in words:
        new_word = tokenizer.tokenize(word)
        new_word = ' '.join(new_word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def remove_emails(words):
    new_words = []
    for word in words:
        new_word = re.sub(r'\S*@\S*\s?', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def remove_newline(words):
    new_words = []
    for word in words:
        new_word = re.sub('\s+', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def remove_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    new_words = []
    for word in words:
        new_word = re.sub(r'\d+', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    stop_words = set(stopwords.words('english'))
    new_words = [i for i in words if not i in stop_words]
    return new_words


def lemmatize(words):
    lemmatizer = WordNetLemmatizer()
    new_words = []
    for word in words:
        new_words.append(lemmatizer.lemmatize(word))
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = PorterStemmer()
    new_words = [stemmer.stem(i) for i in words]
    return new_words


def remove_smallwords(words):
    new_words = [i for i in words if len(i) > 2]
    return new_words


def remove_nonalphabet(words):
    return re.sub('[^A-Za-z]', ' ', words)


def preprocess(words):
    words = remove_non_ascii(words)
    # words = remove_numbers(words)
    words = to_lowercase(words)
    words = replace_contractions(words)
    words = remove_punctuation(words)
    words = remove_stopwords(words)
    words = remove_tabs(words)
    words = remove_emails(words)
    words = lemmatize(words)
    # words = stem_words(words)
    words = remove_smallwords(words)
    return words


def get_data():
    categories = ["comp.windows.x", "comp.os.ms-windows.misc", "talk.politics.misc", "comp.sys.ibm.pc.hardware",
                  "talk.religion.misc", "rec.autos",
                  "sci.space", "talk.politics.guns", "alt.atheism", "misc.forsale", "comp.graphics", "sci.electronics",
                  "sci.crypt", "soc.religion.christian",
                  "rec.sport.hockey", "sci.med", "rec.motorcycles", "comp.sys.mac.hardware", "talk.politics.mideast",
                  "rec.sport.baseball"];
    description = ["Alt - Atheism", "Computer - Graphics", "Computer - Microsoft/Windows", "Computer - IBM/PC",
                   "Computer - Mac",
                   "Computer - Windows", "Misc. - For Sale", "Rec. - Autos", "Rec - Motorcycles", "Sports - Baseball",
                   "Sports - Hockey",
                   "Science - Cryptography", "Science - Electronics", "Science - Medicine", "Science - Space",
                   "Religion - Christianity",
                   "Politics - Guns", "Politics - Middle East", "Politics - Misc.", "Religion - Misc."]
    raw_dataset = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
    dataset = raw_dataset.data
    dataset = ' '.join(dataset)
    dataset = remove_nonalphabet(dataset)
    tokens = word_tokenize(dataset)
    return tokens


def main():
    raw_data = get_data()
    corpus = preprocess(raw_data)
    count_vect = CountVectorizer(max_df=0.85, max_features=1000)
    word_count_vect = count_vect.fit_transform(corpus)
    print(count_vect.vocabulary_)
    print(word_count_vect.shape)
    # print(list(count_vect.vocabulary_.keys())[:10])
    # tfidf_transformer = TfidfTransformer()
    # vectors = tfidf_transformer.fit(word_count_vect)
    # feature_names = vectors
    # X_train_tfidf.shape
    # print(corpus)
    # print(preprocessed_text)

if __name__ == "__main__":
    main()
