import warnings
import time
import pprint
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import json
import pandas as pd
import numpy as np
import re
import unicodedata
import string
import gensim
import pyLDAvis
import pyLDAvis.gensim
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
import spacy
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", DeprecationWarning)


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
    print(words)
    for word in words:
        new_word = re.sub(r'\S*@\S*\s?', '', word)
        new_words.append(new_word)
    return new_words


def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
    new_words = [i for i in words if not i in stop_words]
    return new_words

def lemmatize(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for idx, sent in enumerate(texts):
        # if (idx) % 500 == 0:
        #     print(str(idx) + ' documents lemmatised')
        doc = nlp(" ".join(sent))
        # print("lemma", doc)
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

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


def get_stats(data):
    count_vec = CountVectorizer(strip_accents='unicode')
    count_vec.fit(data)
    term_mat = count_vec.fit_transform(data)
    features = count_vec.get_feature_names()
    tfidfTrans = TfidfTransformer(smooth_idf = False)
    tfidfMatrix = tfidfTrans.fit_transform(term_mat)
    print(len(data), bow.shape, type(bow))
    # print(count_vec.vocabulary_)
    print(bow.toarray().sum(axis=0))


def preprocess(words):
    words = remove_stopwords(words)
    # words = to_lowercase(words)
    # words = remove_non_ascii(words)
    words = remove_smallwords(words)
    # words = lemmatize(words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    # words = remove_nonenglish(words)
    # print('lemma', words)
    return words


def build_lda_model(corpus, id2word):
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=20,
                                           random_state=100,
                                           chunksize=10,
                                           passes=2,
                                           workers=4)
    # pprint(lda_model.print_topics())
    for idx, topic in lda_model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    pyLDAvis.save_html(vis, 'vis.html')

def wordcloud_bow(corpus, id2word):
    vocab = list(id2word.values()) #list of terms in the dictionary
    vocab_tf = [dict(i) for i in bow_corpus]
    vocab_tf = list(pd.DataFrame(vocab_tf).sum(axis=0)) #list of term frequencies
    pairs = [(vocab[i], vocab_tf[i]) for i in range(len(vocab_tf)) if vocab_tf[i] > 2]
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(pairs)
    cnt = Counter()
    for word, frequency in pairs:
        cnt[word] = frequency
    wordcloud = WordCloud(width=800, height=600, relative_scaling=.8).generate_from_frequencies(cnt)
    wordcloud.to_file("wordcloud.png")


def get_data():
    raw_dataset = fetch_20newsgroups(subset='train', categories=["talk.politics.guns", "sci.crypt"], shuffle=True, random_state=42)
    # raw_dataset = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
    dataset = raw_dataset.data
    return dataset

def doc_to_words(sentences):
    new_words = []
    for sentence in sentences:
        word = gensim.utils.simple_preprocess(sentence)
        print("word", word)
        new_words.append(word)
    return new_words

def main():
    raw_data = get_data()
    corpus = []
    original_format_corpus = []
    for doc in raw_data:
        # doc = remove_nonalphabet(doc)
        words = gensim.utils.simple_preprocess(doc)
        # data_chunks = preprocess(tokens)
        # words = doc_to_words(doc)
        data_chunks = preprocess(words)
        # og_format_chunks = ' '.join(data_chunks)
        # original_format_corpus.append(og_format_chunks)
        corpus.append(data_chunks)
    # fewfwef
    # get_stats(original_format_corpus)
    id2word = gensim.corpora.Dictionary(corpus)
    bow_corpus = [id2word.doc2bow(doc) for doc in corpus]
    print(bow_corpus)
    print("Building LDA Model")
    build_lda_model(bow_corpus, id2word)


if __name__ == "__main__":
    main()
