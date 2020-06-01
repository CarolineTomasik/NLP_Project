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

#download sqlite database
try:
    conn = sqlite3.connect('../data/articles.db')
    print ("Opened database successfully");
    
except Exception as e:
    print("Error during connection:",str(e))


# separate fox text from cnn text 
def get_site_articles(site):
    cur = conn.cursor()
    sql = "SELECT * FROM articles WHERE site = " + "'" + site + "';"
    cur.execute(sql)
    articles_list = []
    #rows = cur.fetchmany(2)
    rows = cur.fetchall()
    for row in rows:
        articles_list.append(row[4])
    return articles_list

fox_text = get_site_articles('www.foxnews.com') 
#print(fox_text)  
cnn_text = get_site_articles('www.cnn.com')
#print(cnn_text)

#clean text: tokenize, remove enters, remove punctuation, remove stop words, and lemmatize
stop_words = set(stopwords.words('english'))
print(stop_words)
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
print(lemmatizer)
extra_stopwords = ['this', 'also', 'fox', 'cnn', 'click', 'caption']

for extra in extra_stopwords:
    stop_words.add(extra)
    

def cleaning_text(text):
    lowered = str(text).lower().replace('\\n\\n', ' ').replace("\\\'","")
    lemmatized = lemmatizer.lemmatize(lowered)
    tokens = [str(token) for token in nlp_en(lemmatized)]
    words = [word for word in tokens if word.isalpha()]
    words = [w for w in words if not w in stop_words]
    return words
    

clean_fox = cleaning_text(fox_text)
#print(clean_fox)
clean_cnn = cleaning_text(cnn_text)
#print(clean_cnn)
clean_list = ['coronavirus', 'coronaviruses', 'is', 'was', 'be']
print(clean_list)


#POS tagging
def preprocess(text):
    text = nltk.pos_tag(text)
    return text

fox_text_POS = preprocess(clean_fox)
print(fox_text_POS)
cnn_text_POS = preprocess(clean_cnn)
print(cnn_text_POS)


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

def most_common(text):
    article = nlp(str(text))
    len(article.ents)
    labels = [x.label_ for x in article.ents]
    Counter(labels)
    items = [x.text for x in article.ents]
    return Counter(items).most_common(5)

most_common(clean_fox)
most_common(clean_cnn)


def highlighted_sents(text):
    article = nlp(str(text))
    sentences = [x for x in article.sents]
    print(sentences[20])
    #displacy.render(nlp(str(sentences[20])), jupyter=True, style='ent')

highlighted_sents(clean_fox)
highlighted_sents(clean_cnn)

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



from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()  
analyze = vectorizer.build_analyzer() 
print(‘Fox’,analyze(fox_text))
print(‘CNN’,analyze(cnn_text))
print(‘Document transform’,X.toarray())      

X = vectorizer.fit_transform(fox_text)
print(vectorizer.get_feature_names()) 





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