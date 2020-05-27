#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 16:24:07 2020

@author: caroline
"""

import sqlite3
import spacy
nlp_en = spacy.blank("en")

try:
    conn = sqlite3.connect('../data/articles.db')
    print ("Opened database successfully");
    
except Exception as e:
    print("Error during connection: ",str(e))

cur = conn.cursor()
cur.execute("SELECT * FROM articles")
#rows = cur.fetchmany(5)
rows = cur.fetchall()

for row in rows:
    text = row[4]
    print([str(token) for token in nlp_en(text.lower())])



#conn.close()