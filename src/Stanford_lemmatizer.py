#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 17:09:06 2020

@author: caroline
"""
from stanfordcorenlp import StanfordCoreNLP
import json, string

lemma_test = ['coronavirus', 'coronaviruses', 'is', 'was', 'be']

def lemmatize_corenlp(conn_nlp, sentence):
    props = {
        'annotators': 'pos,lemma',
        'pipelineLanguage': 'en',
        'outputFormat': 'json'
    }
        # tokenize into words
    sents = conn_nlp.word_tokenize(sentence)

    # remove punctuations from tokenised list
    sents_no_punct = [s for s in sents if s not in string.punctuation]

    # form sentence
    sentence2 = " ".join(sents_no_punct)

    # annotate to get lemma
    parsed_str = conn_nlp.annotate(sentence2, properties=props)
    parsed_dict = json.loads(parsed_str)

    # extract the lemma for each word
    lemma_list = [v for d in parsed_dict['sentences'][0]['tokens'] for k,v in d.items() if k == 'lemma']

    # form sentence and return it
    #return " ".join(lemma_list)
    return lemma_list

nlp = StanfordCoreNLP('http://localhost', port=9000, timeout=30000)
print(lemmatize_corenlp(conn_nlp=nlp, sentence=' '.join(lemma_test)))