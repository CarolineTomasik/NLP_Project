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
stop_words = set(stopwords.words('english'))

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


#clean text: tokenize, remove enters, remove punctuation
def cleaning_text(text):
    tokens = [str(token) for token in nlp_en(str(text).replace('\\n\\n', ' ').replace("\\\'",""))]
    words = [word for word in tokens if word.isalpha()]
    #print(len(words))
    words = [w for w in words if not w in stop_words]
    #print(len(words))
    return words

clean_fox = cleaning_text(fox_text)
#print(clean_fox)
clean_cnn = cleaning_text(cnn_text)
#print(clean_cnn)


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


