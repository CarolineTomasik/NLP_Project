#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 15:18:24 2020

@author: caroline
"""

import nltk
nltk.download('stopwords')
import pandas as pd
import numpy as np
import re
import codecs
from nltk.tokenize import RegexpTokenizer
import sqlite3
import spacy
nlp_en = spacy.blank("en")
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import RegexpTokenizer
nltk.download('maxent_ne_chunker')
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('wordnet')
from stanfordcorenlp import StanfordCoreNLP
from Stanford_lemmatizer import lemmatize_corenlp
from itertools import chain

# select articles text by site
def get_site_articles(site):
    cur = conn.cursor()
    sql = "SELECT * FROM articles WHERE site = '{}' AND length(article) > 0;".format(site)
    cur.execute(sql)
    articles_list = []
    #rows = cur.fetchmany(31)
    rows = cur.fetchall()
    for row in rows:
        if len(row[4]) > 0:
            articles_list.append(row[4])
    return articles_list


def configure_stopwords():
	stop_words = set(stopwords.words('english'))
	extra_stopwords = ['this', 'also', 'fox', 'cnn', 'click', 'caption']
	for extra in extra_stopwords:
		stop_words.add(extra)
	return stop_words

 
def cleaner(stop_words, text):
	#clean text: tokenize, remove newlines, remove punctuation, remove stop words
    lowered = str(text).lower().replace('\\n\\n', ' ').replace("\\\'","")
    tokens = [str(token) for token in nlp_en(lowered)]
    words = [word for word in tokens if word.isalpha()]
    words = [w for w in words if not w in stop_words]
    return words


def preprocess_text(site):
		articles_list = get_site_articles(site) 
		clean_list = []
		for article in articles_list:
			clean = cleaner(stop_words, article)
			clean = lemmatize_corenlp(conn_nlp=nlp, sentence=' '.join(clean))
			clean_list.append(clean)
		return clean_list


def main():
	#connect to sqlite database
	try:
		conn = sqlite3.connect('../data/articles.db')
		print ("Opened database successfully");
	except Exception as e:
		print("Error during connection:",str(e))

	# connect to Standford lemmatizer
	nlp = StanfordCoreNLP('http://localhost', port=9000, timeout=30000)

	stop_words = configure_stopwords()

	fox_clean = preprocess_text('www.foxnews.com')
	cnn_clean = preprocess_text('www.cnn.com')

	print("Found {0} FOXnews and {1} CNN articles".format(len(fox_clean), len(cnn_clean)))

	print(Counter(list(chain(*fox_clean))).most_common(10))
	print(Counter(list(chain(*cnn_clean))).most_common(10))


if __name__ == '__main__':
    main()



#Count frequencies and graph
freq = nltk.FreqDist(clean_fox)
#print(freq)
for key,val in freq.items():
    print(str(key) + ':' + str(val))    
freq.plot(20, cumulative=False)    

freq = nltk.FreqDist(clean_cnn)
#print(freq)
for key,val in freq.items():
    print(str(key) + ':' + str(val))    
freq.plot(20, cumulative=False) 






def cleaning_text(text):
    lowered = str(text).lower().replace('\\n\\n', ' ').replace("\\\'","")
    #lemmatized = lemmatize_corenlp(conn_nlp=nlp, sentence=' '.join(lowered))
    tokens = [str(token) for token in nlp_en(lowered)]
    words = [word for word in tokens if word.isalpha()]
    words = [w for w in words if not w in stop_words]
    return words


fox_text = ' '.join(lemmatizer(fox_text))    
fox_clean = cleaning_text((lemmatize_corenlp(conn_nlp=nlp, sentence=fox_text)))

cnn_text =  ' '.join(lemmatizer(cnn_text))    
cnn_clean = cleaning_text((lemmatize_corenlp(conn_nlp=nlp, sentence=cnn_text)))
print(cnn_clean)



clean_fox = cleaning_text(fox_text)
print(clean_fox)

clean_cnn = cleaning_text(cnn_text)
#print(clean_cnn)
lemma_test = ['coronavirus', 'coronaviruses', 'is', 'was', 'be']
#print(cleaning_text(lemma_test))
lemma1 = lemmatize_corenlp(conn_nlp=nlp, sentence=' '.join(lemma_test))
#print(lemma1)


#POS tagging
def preprocess(text):
    text = nltk.pos_tag(text)
    return text

fox_text_POS = preprocess(clean_fox)
print(fox_text_POS)
cnn_text_POS = preprocess(clean_cnn)
print(cnn_text_POS)
"""
"""

#entity tagging
# import sys
# !{sys.executable} -m pip install spacy
# !{sys.executable} -m spacy download en
# spacy.load('en')
# pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz
# #after downloading, restart kernel and then run the following:  
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()

doc = nlp(str(fox_text))
#pprint([(X.text, X.label_) for X in doc.ents])

#BILUO tagging
#pprint([(X, X.ent_iob_, X.ent_type_) for X in doc])


#Counting
from bs4 import BeautifulSoup
import requests
import re

# def most_common(text):
#     article = nlp(str(text))
#     len(article.ents)
#     labels = [x.label_ for x in article.ents]
#     Counter(labels)
#     items = [x.text for x in article.ents]
#     return Counter(items).most_common(5)

# most_common(clean_fox)
# most_common(clean_cnn)


def highlighted_sents(text):
    article = nlp(str(text))
    sentences = [x for x in article.sents]
    print(sentences[20])
    #displacy.render(nlp(str(sentences[20])), jupyter=True, style='ent')

highlighted_sents(clean_fox)
highlighted_sents(clean_cnn)

Counter(clean_fox).most_common(5)
Counter(clean_cnn).most_common(5)

#classification

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

fox_df = pd.DataFrame(clean_fox)
fox_df


























'''
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(clean_fox).toarray()
y = clean_fox.iloc[:, 1].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#ATTEMPTS
#TFIDF
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(clean_fox)
analyze = vectorizer.build_analyzer()
print(clean_fox, analyze(clean_fox))
print(clean_fox,list(clean_fox).toarray())

print(vectorizer)

X = vectorizer.fit_transform(list(clean_fox))
print(vectorizer.get_feature_names())


DF = {}
for i in range(len(clean_fox)):
    tokens = processed_text[i]
    for w in tokens:
        try:
            DF[w].add(i)
        except:
            DF[w] = {i}
            
sklearn.feature_extraction.text.TfidfVectorizer(fox_text)            

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(fox_text)
print(vectorizer.get_feature_names())
print(X.shape)



# from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer = TfidfVectorizer()  
# analyze = vectorizer.build_analyzer() 
# print(‘Fox’,analyze(fox_text))
# print(‘CNN’,analyze(cnn_text))
# print(‘Document transform’,X.toarray())      

# X = vectorizer.fit_transform(fox_text)
# print(vectorizer.get_feature_names()) 





bagOfWordsA = fox_text
bagOfWordsB = cnn_text

uniqueWords = set(bagOfWordsA).union(set(bagOfWordsB))

numOfWordsA = dict.fromkeys(uniqueWords, 0)
for word in bagOfWordsA:
    numOfWordsA[word] += 1
numOfWordsB = dict.fromkeys(uniqueWords, 0)
for word in bagOfWordsB:
    numOfWordsB[word] += 1
    
    
def computeTF(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict    
    
tfA = computeTF(numOfWordsA, bagOfWordsA)
#print(tfA)
tfB = computeTF(numOfWordsB, bagOfWordsB)    
#print(tfB)   

def computeIDF(documents):
    import math
    N = len(documents)
    
    idfDict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1
    
    for word, val in idfDict.items():
        idfDict[word] = math.log(N / float(val))
    return idfDict    
   
 
idfs = computeIDF([numOfWordsA, numOfWordsB])    
    
def computeTFIDF(tfBagOfWords, idfs):
    tfidf = {}
    for word, val in tfBagOfWords.items():
        tfidf[word] = val * idfs[word]
    return tfidf    
    
tfidfA = computeTFIDF(tfA, idfs)
tfidfB = computeTFIDF(tfB, idfs)
df = pd.DataFrame([tfidfA, tfidfB])    

print(df)    
    
    
print(str(clean_fox))



tokens = [t for t in fox_text.split()]
print(tokens) 

from nltk.corpus import stopwords
sr= stopwords.words('english')
clean_tokens = tokens[:]
for token in tokens:
    if token in stopwords.words('english'):
        
        clean_tokens.remove(token)
        
freq = nltk.FreqDist(clean_tokens)
for key,val in freq.items():
    print(str(key) + ':' + str(val))
freq.plot(20, cumulative=False)



print(fox_text)

print(clean_fox)




list1 = ['material', 'may', 'published']

def wordListToFreqDict(wordlist):
    wordfreq = [wordlist.count(p) for p in wordlist]
    return dict(list(zip(wordlist,wordfreq)))

wordListToFreqDict(list1)
    
def sortFreqDict(freqdict):
    aux = [(freqdict[key], key) for key in freqdict]
    aux.sort()
    aux.reverse()
    return aux

sortFreqDict(clean_fox)



Counter(clean_fox)
'''

