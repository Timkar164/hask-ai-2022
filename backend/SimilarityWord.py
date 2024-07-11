# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 19:57:52 2022

@author: timka
"""
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from string import punctuation
russian_stopwords = stopwords.words("russian")
import spacy
import pandas as pd
nlp = spacy.load("ru_core_news_lg") #или "ru_core_news_sm" но она без векторов зато легкая https://proglib.io/p/fun-nlp
df = pd.read_pickle('models/dataframe10.pkl')

def preprocess(line):
        support_chars = {33: ' ', 34: ' ', 35: ' ', 36: ' ', 37: ' ', 38: ' ', 39: ' ', 40: ' ', 41: ' ', 42: ' ', 43: ' ', 44: ' ', 45: ' ', 46: ' ', 47: ' ', 58: ' ', 59: ' ', 60: ' ', 61: ' ', 62: ' ', 63: ' ', 64: ' ', 91: ' ', 92: ' ', 93: ' ', 94: ' ', 95: ' ', 96: ' ', 123: ' ', 124: ' ', 125: ' ', 126: ' '}
        line = line.translate(support_chars).lower().split(' ')
        t = [token for token in line if token not in russian_stopwords and token != " " and token.strip() not in punctuation]
        return ' '.join(t)
def get10code(text,TNVED):
    global nlp
    global df
    data = df[df['TNVED4']==TNVED]
    OPISANIE_SPR = data['OPISANIE_SPR'].tolist()
    TNVED6=data['TNVED6'].tolist()
    maxDist = 0
    maxRes = 0
    doc1 = nlp(text)
    for i in range(len(OPISANIE_SPR)):
        line = preprocess(OPISANIE_SPR[i])
        doc2 = nlp(line)
        res = doc1.similarity(doc2)
        if res>maxDist:
            maxDist=res
            maxRes=i
    return TNVED6[maxRes]

    