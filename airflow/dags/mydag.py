import os
import datetime as dt

import requests
import pandas as pd
from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator
import pandas as pd
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import os 
import time
import pickle
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from string import punctuation
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

def build_model():
 russian_stopwords = stopwords.words("russian")
 data = pd.read_csv('dataset.csv',encoding='utf-8',sep=';')
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
 data['TNVED']=data['TNVED'].map(lambda x: int(str(x)[:2]))
 from sklearn.model_selection import train_test_split
 X_train, X_valid, y_train, y_valid = train_test_split(data['Post_clean'], data['TNVED'], test_size=0.1, random_state=42)
 X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
 from sklearn.pipeline import Pipeline
 # pipeline позволяет объединить в один блок трансформер и модель, что упрощает написание кода и улучшает его читаемость
 from sklearn.feature_extraction.text import TfidfVectorizer
 # TfidfVectorizer преобразует тексты в числовые вектора, отражающие важность использования каждого слова из некоторого набора слов (количество слов набора определяет размерность вектора) в каждом тексте
 from sklearn.linear_model import SGDClassifier
 from sklearn.neighbors import KNeighborsClassifier
 # линейный классификатор и классификатор методом ближайших соседей
 from sklearn import metrics
 # набор метрик для оценки качества модели
 from sklearn.model_selection import GridSearchCV
 # модуль поиска по сетке параметров
 sgd_ppl_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('sgd_clf', SGDClassifier(random_state=42))])
 sgd_ppl_clf.fit(X_train, y_train)
 pickle.dump(sgd_ppl_clf, open('sgd_ppl_clf'+str(int(time.time())+'.pkl', 'wb'))


args = {
    'owner': 'airflow',
    'start_date': dt.datetime(2022, 6, 5),
    'schedule_interval': dt.timedelta(days=1),
    'retries': 1,
    'retry_delay': dt.timedelta(minutes=30),
    'depends_on_past': False,
}


with DAG(dag_id='update_caskade_models', default_args=args, schedule_interval=dt.timedelta(minutes=5)) as dag:
    create_model = PythonOperator(
        task_id='build_model',
        python_callable=build_model,
        dag=dag
    )
    create_model
