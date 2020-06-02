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
from collections import Counter
from itertools import chain
import pickle
import pprint

# select articles text by site
def get_site_articles(conn, site):
	cur = conn.cursor()
	sql = "SELECT * FROM articles WHERE site = '{}' AND length(article) > 0;".format(site)
	cur.execute(sql)
	articles_list = []
	rows = cur.fetchall()
	#rows = cur.fetchmany(31)
	for row in rows:
		if len(row[4]) > 0:
			articles_list.append(row[4])
	return articles_list


def configure_stopwords():
	stop_words = set(stopwords.words('english'))
	extra_stopwords = ['this', 'also', 'fox', 'cnn', 'click', 'caption', 'photo', 'hide', 'latest', 'inbox', 'app','']
	for extra in extra_stopwords:
		stop_words.add(extra)
	return stop_words

#clean text: tokenize, remove newlines, remove punctuation, remove stop words 
def cleaner(stop_words, text):
	lowered = str(text).lower().replace('\\n\\n', ' ').replace("\\\'","")
	tokens = [str(token) for token in nlp_en(lowered)]
	words = [word for word in tokens if word.isalpha()]
	words = [w for w in words if not w in stop_words]
	return words


def preprocess_text(conn, site, stop_words, nlp):
		articles_list = get_site_articles(conn, site) 
		clean_list = []
		for article in articles_list:
			clean = cleaner(stop_words, article)
			clean = lemmatize_corenlp(conn_nlp=nlp, sentence=' '.join(clean))
			clean_list.append(clean)
		return clean_list


def freq_counter(tokens, title):
	#Count frequencies and graph
	freq = nltk.FreqDist(chain(*tokens))
	for key,val in freq.items():
		print(str(key) + ':' + str(val))	
	freq.plot(20, title=title, cumulative=False)	


def bigram_counter(tokens, title):
	#bigrams
	bigrams = [(x, i[j + 1]) for i in tokens
	       for j, x in enumerate(i) if j < len(i) - 1] 
	bigram_freq = nltk.FreqDist(bigrams)
	for key,val in bigram_freq.items():
		print(str(key) + ':' + str(val))
	bigram_freq.plot(20, title = title, cumulative =False)
	Counter(bigrams).most_common(10)


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

	clean_fox = preprocess_text(conn, 'www.foxnews.com', stop_words, nlp)
	clean_cnn = preprocess_text(conn, 'www.cnn.com', stop_words, nlp)
	
	dbfile = open(r'../data/fox_clean.pickle', 'wb')
	pickle.dump(clean_fox, dbfile)
	dbfile.close()
	
	dbfile = open(r'../data/cnn_clean.pickle', 'wb')
	pickle.dump(clean_cnn, dbfile)
	dbfile.close()
	
	print("Found {0} FOXnews and {1} CNN articles".format(len(clean_fox), len(clean_cnn)))

	freq_counter(clean_fox, 'Fox News Word Frequency')
	freq_counter(clean_cnn, 'CNN Word Frequency')
	
	bigram_counter(clean_fox, "Fox News Bigram Frequency")
	bigram_counter(clean_cnn, "CNN Bigram Frequency")
	
	pprint(Counter(list(chain(*clean_fox))).most_common(10))
	pprint(Counter(list(chain(*clean_cnn))).most_common(10))


if __name__ == '__main__':
	main()
	
