#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 16:24:07 2020

@author: caroline
"""

import sqlite3

try:
    conn = sqlite3.connect('../data/articles.db')
    print ("Opened database successfully");
    
except Exception as e:
    print("Error during connection: ",str(e))

#conn.close()

cur = conn.cursor()
cur.execute("SELECT * FROM articles")
rows = cur.fetchmany(5)

for row in rows:
    print(row)

