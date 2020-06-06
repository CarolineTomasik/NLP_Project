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
import gensim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


def get_site_articles(conn, site):
	# select articles text by site
	cur = conn.cursor()
	sql = "SELECT * FROM articles WHERE site = '{}' AND length(article) > 0;".format(site)
	cur.execute(sql)
	articles_list = []
	rows = cur.fetchall()
	for row in rows:
		if len(row[4]) > 0:
			articles_list.append(row[4])
	return articles_list


def configure_stopwords():
	#add extra stopwords that are irrelevant for our purposes
	stop_words = set(stopwords.words('english'))
	extra_stopwords = ['this', 'also', 'fox', 'cnn', 'click', 'caption', 'photo', 'hide', 'latest', 'inbox', 'app','']
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


def preprocess_text(conn, site, stop_words, nlp):
	#Use Stanford CoreNLP to lemmatize
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


def tfidf(text):
	#tfidf for both sites
	dictionary = gensim.corpora.Dictionary(text)
	print(dictionary.token2id)
	corpus = [dictionary.doc2bow(token) for token in text]
	print(corpus)
	tfidf = gensim.models.TfidfModel(corpus, smartirs='ntc')
	for text in tfidf[corpus]:
		   print([[dictionary[id], np.around(freq, decimals=2)] for id, freq in text])
	return tfidf


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
	
	#create list of all articles and all sites for classification
	all_articles =  clean_fox + clean_cnn
	#print(all_articles)
	all_sites = list([0] * 149 + [1] * 115)
	#print(all_sites)
	X, y = [' '.join(x) for x in all_articles], all_sites
	print(X)
	vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7)
	X = vectorizer.fit_transform(X) #.toarray()
	#y = vectorizer.fit_transform(all_sites)#.toarray()
	tfidfconverter = TfidfTransformer()
	X = tfidfconverter.fit_transform(X).toarray()
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
	
	#random forest classifier
	classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
	classifier.fit(X_train, y_train) 
	y_pred = classifier.predict(X_test)
	
	#confusion matrix
	print(confusion_matrix(y_test,y_pred))
	cm = confusion_matrix(y_test,y_pred)
	ax= plt.subplot()
	sns.heatmap(cm, annot=True, ax = ax, cmap='Greens'); 
	ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
	ax.set_title('Confusion Matrix'); 
	ax.xaxis.set_ticklabels(['Fox News', 'CNN']); ax.yaxis.set_ticklabels(['Fox News', 'CNN']);
	
	print(classification_report(y_test,y_pred))
	print(accuracy_score(y_test, y_pred))
	
	#feature importance
# 	importances = classifier.feature_importances_
# 	std = np.std([tree.feature_importances_ for tree in classifier.estimators_], axis=0)
# 	indices = np.argsort(importances)[::-1]
	# Print the feature ranking
	#print("Feature ranking:")
	#for f in range(X.shape[1]):
		#print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

	# Plot the impurity-based feature importances of the forest
# 	plt.figure()
# 	plt.title("Feature importances")
# 	X_subset = range(X.shape[1])
# 	X_subset = X_subset[:20]
# 	plt.bar(X_subset, importances[indices[:20]],
# 	        color="r", yerr=std[indices[:20]], align="center")
# 	plt.xticks(X_subset, indices[:20])
# 	plt.xlim([-1, X_subset])
# 	plt.show()
	

	

	#dbfile = open(r'../data/fox_clean.pickle', 'wb')
	#pickle.dump(clean_fox, dbfile)
	#dbfile.close()
	
	#dbfile = open(r'../data/cnn_clean.pickle', 'wb')
	#pickle.dump(clean_cnn, dbfile)
	#dbfile.close()
	
	print("Found {0} FOXnews and {1} CNN articles".format(len(clean_fox), len(clean_cnn)))

	#freq_counter(clean_fox, 'Fox News Word Frequency')
	#freq_counter(clean_cnn, 'CNN Word Frequency')
	
	#bigram_counter(clean_fox, "Fox News Bigram Frequency")
	#bigram_counter(clean_cnn, "CNN Bigram Frequency")
	
	#pprint(Counter(list(chain(*clean_fox))).most_common(10))
	#pprint(Counter(list(chain(*clean_cnn))).most_common(10))
	
	#fox_tfidf = tfidf(clean_fox)
	#print(fox_tfidf)
	#cnn_tfidf = tfidf(clean_cnn)
	#print(cnn_tfidf)
	



if __name__ == '__main__':
	main()
	
	
	'''
	#Naive bayes classifier
	count_vect = CountVectorizer()
	x_train_counts = count_vect.fit_transform(all_articles)
	tfidf_transformer = TfidfTransformer()
	x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
	 
	train_x, test_x, train_y, test_y = train_test_split(x_train_tfidf, all_sites, test_size=0.3)
	
	clf = MultinomialNB().fit(train_x, train_y)
	y_score = clf.predict(test_x)
	
	n_right = 0	
	for i in range(len(y_score)):
	    if y_score[i] == test_y[i]:
	        n_right += 1
	
	print("Accuracy: %.2f%%" % ((n_right/float(len(test_y)) * 100)))
	'''