# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 15:05:03 2020

@author: caroline
"""

"""
!pip install google
!pip install newspaper3k
!pip install beautifulsoup4
"""

# to search
from googlesearch import search
from newspaper import Article
from newspaper import ArticleException
from newspaper import fulltext
import requests
from bs4 import BeautifulSoup
from collections import defaultdict
import time

def get_foxnews_date(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'lxml')

    for tag in soup.find_all('meta'):
        prop = tag.get('data-hid', None)
        if prop == 'dc.date':
            return tag.get('content')


def read_article(url):
    publish_date = None
    article = Article(url)
    article.download()
    article.parse()
    print('======================================================')
    print(url)
    authors = ', '.join(article.authors)
    print(authors)
    if article.publish_date:
        publish_date = article.publish_date
    else:
        publish_date = get_foxnews_date(url)
    print(publish_date)
    return (url, publish_date, authors, article.text)
    #print(article.text)

    #article.nlp()
    #article.keywords
    #article.summary


# not used
def read_full_text(url):
    html = requests.get(url).text
    text = fulltext(html)
    text

urls = defaultdict(list)
for q in ['covid-19', 'coronavirus', 'pandemic', 'wuhan', 'lockdown']:
    for s in ['site:foxnews.com', 'site:cnn.com']:
        query = q + ' ' + s
        print(query)
        for url in search(query, tld="com", num=10, stop=None, pause=5):
            urls[url].append(q)
        print(len(urls))
        time.sleep(10)


'''

import pickle

dbfile = open(r'..\data\article_urls.pickle', 'wb')
pickle.dump(urls, dbfile)
dbfile.close()

dbfile = open(r'..\data\article_urls.pickle', 'rb')
urls = pickle.load(dbfile)
dbfile.close()
'''

import sqlite3
from sqlite3 import IntegrityError
from sqlite3 import Error
from create_db import create_connection

conn = create_connection(r"..\data\articles.db")
sql_ins_search = ''' INSERT INTO search(url, search_key)
                    VALUES(?,?) '''

for url in urls:
    for search_key in urls[url]:
        print(url + "\t" + search_key)
        try:
            conn.execute(sql_ins_search, (url, search_key))
        except IntegrityError as e:
            print(e)

conn.commit()

sql_ins_article = '''INSERT INTO articles(url, date, authors, article)
                    VALUES(?,?,?,?)'''

for url in urls:
    try:
        conn.execute(sql_ins_article, read_article(url))
        print(url)
        conn.commit()
    except Error as e:
        print(e)
    except ArticleException as e:
        print(e)


'''
dc.date gives date only
dcterms.created gives full datetime

<meta content="2020-04-17" data-hid="dc.date" data-n-head="true" name="dc.date"/>
<meta content="2020-04-17T02:00:37-04:00" data-hid="dcterms.created" data-n-head="true"

print(soup.head.prettify())
    == 'dc.date':
        print(tag.get('content', None)

for tag in soup.find_all("meta"):
    if tag.get("property", None) == "og:title":
        print tag.get("content", None)
    elif tag.get("property", None) == "og:url":
        print tag.get("content", None)

# <meta data-n-head="true" data-hid="dc.date" name="dc.date" content="2020-04-23">
'''