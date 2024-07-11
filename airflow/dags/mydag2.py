import os
import datetime as dt

import requests
import pandas as pd
from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator
import sklearn
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import BallTree
from sklearn.base import BaseEstimator
import json
import pickle
import pandas as pd
import time
# Получение текстовой строки из списка слов
def str_corpus(corpus):
    str_corpus = ''
    for i in corpus:
        str_corpus += ' ' + i
    str_corpus = str_corpus.strip()
    return str_corpus
# Получение списка всех слов в корпусе
def get_corpus(data):
    corpus = []
    for phrase in data:
        for word in phrase.split():
            corpus.append(word)
    return corpus
# Получение облака слов
def get_wordCloud(corpus):
    wordCloud = WordCloud(background_color='white',
                              stopwords=STOPWORDS,
                              width=3000,
                              height=2500,
                              max_words=200,
                              random_state=42
                         ).generate(str_corpus(corpus))
    return wordCloud

def softmax(x):
  #создание вероятностного распределения
  proba = np.exp(-x)
  return proba / sum(proba)

class NeighborSampler(BaseEstimator):
  def __init__(self, k=5, temperature=10.0):
    self.k=k
    self.temperature = temperature
  def fit(self, X, y):
    self.tree_ = BallTree(X)
    self.y_ = np.array(y)
  def predict(self, X, random_state=None):
    distances, indices = self.tree_.query(X, return_distance=True, k=self.k)
    result = []
    resultDist = []
    for distance, index in zip(distances, indices):
      result.append(np.random.choice(index, p=softmax(distance * self.temperature)))
      resultDist.append(np.random.choice(distance, p=softmax(distance * self.temperature)))
    return self.y_[result] , resultDist
def build_data():
 data = pd.read_csv('dataset.csv',sep=';')
 import nltk
 nltk.download("stopwords")
 from nltk.corpus import stopwords
 from string import punctuation
 russian_stopwords = stopwords.words("russian")
 data=data[:10000]

 def remove_punct(text):
    table = {33: ' ', 34: ' ', 35: ' ', 36: ' ', 37: ' ', 38: ' ', 39: ' ', 40: ' ', 41: ' ', 42: ' ', 43: ' ', 44: ' ', 45: ' ', 46: ' ', 47: ' ', 58: ' ', 59: ' ', 60: ' ', 61: ' ', 62: ' ', 63: ' ', 64: ' ', 91: ' ', 92: ' ', 93: ' ', 94: ' ', 95: ' ', 96: ' ', 123: ' ', 124: ' ', 125: ' ', 126: ' '}
    return text.translate(table)

 data['Post_clean'] = data['OPISANIE'].map(lambda x: x.lower())
 data['Post_clean'] = data['Post_clean'].map(lambda x: remove_punct(x))
 data['Post_clean'] = data['Post_clean'].map(lambda x: x.split(' '))
 data['Post_clean'] = data['Post_clean'].map(lambda x: [token for token in x if token not in russian_stopwords\
                                                                  and token != " " \
                                                                  and token.strip() not in punctuation])
 data['Post_clean'] = data['Post_clean'].map(lambda x: ' '.join(x))
 data['TNVED2']=data['TNVED'].map(lambda x: int(str(x)[:-2]))
 data['TNVED4']=data['TNVED'].map(lambda x: int(str(x)[2:]))
 TwoClasses = data['TNVED2'].unique()
 import pymorphy2
 Errors = []
 morph = pymorphy2.MorphAnalyzer(lang='ru')
 for TNVED in TwoClasses:
   try:
    df = data[data['TNVED2']==TNVED]
    questions_response = df['Post_clean'].tolist()
    answer_response = df['TNVED4'].tolist()
    questions = []
    answer = []
    transform=0
    for i in range(len(questions_response)):
          if questions_response[i]>"":
           phrases=questions_response[i]
   
            # разбираем вопрос на слова
           words=phrases.split(' ')
           phrase=""
           for word in words:
                word = morph.parse(word)[0].normal_form  
                phrase = phrase + word + " "
                # Если длинна полученной фразы больше 0 добавляем ей в массив вопросов и массив кодов ответов
           if (len(phrase)>0):
             questions.append(phrase.strip())
             transform=transform+1
             answer.append(answer_response[i])
    vectorizer_q = TfidfVectorizer()
    vectorizer_q.fit(questions)
    matrix_big_q = vectorizer_q.transform(questions)
    if transform>200:
              transform=200
    svd_q = TruncatedSVD(n_components=transform)
    svd_q.fit(matrix_big_q)
    matrix_small_q = svd_q.transform(matrix_big_q)
    ns_q = NeighborSampler()
    ns_q.fit(matrix_small_q, answer) 
    pipe_q = make_pipeline(vectorizer_q, svd_q, ns_q)
    pickle.dump(pipe_q, open('models/'+str(int(time.time())+'TNVED'+str(TNVED)+'.pkl', 'wb'),protocol=4)
   except:
    Errors.append(TNVED)

args = {
    'owner': 'airflow',
    'start_date': dt.datetime(2022, 6, 5),
    'schedule_interval': dt.timedelta(days=1),
    'retries': 1,
    'retry_delay': dt.timedelta(minutes=30),
    'depends_on_past': False,
}


with DAG(dag_id='update_caskade_models_2', default_args=args, schedule_interval=dt.timedelta(minutes=5)) as dag:
    create_model_data = PythonOperator(
        task_id='build_data',
        python_callable=build_data,
        dag=dag
    )
    create_model_data
