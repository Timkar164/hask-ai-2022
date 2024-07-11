import os
import numpy as np
import torch
import transformers
import torch.nn as nn
from transformers import AutoModel, BertTokenizer
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from string import punctuation
russian_stopwords = stopwords.words("russian")



'''import keras
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Embedding, LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import tokenizer_from_json
import json
class ClassifierLSTM:
    def __init__(self, weights = 'lstm'):
        self.model = keras.models.load_model('lstm')
        
        with open('lstm_tokenizer.json') as f:
            data = json.load(f)
            self.tokenizer = tokenizer_from_json(data)
        
        self.encoder = LabelEncoder()
        self.encoder.classes_ = np.load('lstm_encoder_classes.npy')

    def preprocess(self, line):
        support_chars = {33: ' ', 34: ' ', 35: ' ', 36: ' ', 37: ' ', 38: ' ', 39: ' ', 40: ' ', 41: ' ', 42: ' ', 43: ' ', 44: ' ', 45: ' ', 46: ' ', 47: ' ', 58: ' ', 59: ' ', 60: ' ', 61: ' ', 62: ' ', 63: ' ', 64: ' ', 91: ' ', 92: ' ', 93: ' ', 94: ' ', 95: ' ', 96: ' ', 123: ' ', 124: ' ', 125: ' ', 126: ' '}
        line = line.translate(support_chars).lower().split(' ')
        t = [token for token in line if token not in russian_stopwords and token != " " and token.strip() not in punctuation]
        return ' '.join(t)        
        
    def predict(self, line):
        line = self.preprocess(line)
        
        text_sec = self.tokenizer.texts_to_sequences([input_line])
        text_sec = pad_sequences(text_sec, maxlen=69)
        pred = self.model.predict(text_sec, batch_size=1, verbose=1)
        pred = np.argmax(pred,axis=1)
        pred = self.encoder.inverse_transform(pred) - 1
        if pred < 0:
            pred = 0
        
        return pred
        

input_line = 'изделия прочие пластмасс изделия прочих материалов товарных позиций 3901 3914 прочие прочие прочие прочие'      
myLSTMmodel = ClassifierLSTM()
def LSTM(input_line):
    global myLSTMmodel
    res = myLSTMmodel.predict(input_line)
    return res[0]

'''

class BERT_Arch(nn.Module):
    def __init__(self, bert, num_classes = 96):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768,512)
        self.fc2 = nn.Linear(512,num_classes)
        self.softmax = nn.LogSoftmax(dim = 1)
    
    def forward(self, sent_id, mask):
        _, cls_hs = self.bert(sent_id, attention_mask = mask, return_dict = False)
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

class TansformerRuBERT:
    def __init__(self, weights = 'models/bert.pt', dev='cpu'):
        # dev='cuda'
        self.device = torch.device(dev)
        self.bert = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased-sentence")
        self.tokenizer = BertTokenizer.from_pretrained("DeepPavlov/rubert-base-cased-sentence")
        for param in self.bert.parameters():
            param.requires_grad = False
        self.model = BERT_Arch(self.bert)
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load(weights, map_location=torch.device(self.device)))
        self.model.eval()
        
    def preprocess(self, line):
        support_chars = {33: ' ', 34: ' ', 35: ' ', 36: ' ', 37: ' ', 38: ' ', 39: ' ', 40: ' ', 41: ' ', 42: ' ', 43: ' ', 44: ' ', 45: ' ', 46: ' ', 47: ' ', 58: ' ', 59: ' ', 60: ' ', 61: ' ', 62: ' ', 63: ' ', 64: ' ', 91: ' ', 92: ' ', 93: ' ', 94: ' ', 95: ' ', 96: ' ', 123: ' ', 124: ' ', 125: ' ', 126: ' '}
        line = line.translate(support_chars).lower().split(' ')
        t = [token for token in line if token not in russian_stopwords and token != " " and token.strip() not in punctuation]
        return ' '.join(t)
        
        
    def predict(self, line):
        line = self.preprocess(line)
        
        sequence = self.tokenizer.encode(line, 
                                        max_length = 15, 
                                        padding = 'max_length',
                                        truncation = True)
        mask = torch.tensor([1]*len(sequence)).to(self.device)
        sequence = torch.tensor(sequence).to(self.device)
        mask = torch.unsqueeze(mask, 0)
        sequence = torch.unsqueeze(sequence, 0)
        res = self.model(sequence, mask)
        res = int(res.argmax(dim=1).cpu().numpy())
        if res> 77:
            return res+2
        else:
            return res+1
bertModel = TansformerRuBERT()   
def bert(input_line):
   try:
    global bertModel
    res = bertModel.predict(input_line)
    return res
   except:
    return 00

def preprocess(text):
    global bertModel
    return bertModel.preprocess(text)

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
import sklearn
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import BallTree
from sklearn.base import BaseEstimator
import pickle

modelSGD2 = pickle.load(open('models/sgd_ppl_clf_big.pkl', 'rb'))
def SGD2(input_line):
   try:
    global modelSGD2
    res = modelSGD2.predict([input_line])
    return res[0]
   except:
    return 00

modelSGD4 = pickle.load(open('models/sgd_ppl_clf_4.pkl', 'rb'))
def SGD4(input_line):
   try:
    global modelSGD4
    res = modelSGD4.predict([input_line])
    return res[0]
   except:
    return 0000

modelSGD2PipLines = pickle.load(open('models/sgd_ppl_clf_pipline_big.pkl','rb'))
def SGD2Piplines(input_line):
   try:
    global modelSGD2PipLines
    res = modelSGD2PipLines.predict([input_line])
    return res[0]
   except:
    return 00

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
def SGD2_2(class_name,input_line):
    model = pickle.load(open('models/TNVED'+str(class_name)+'.pkl', 'rb'))
    res = model.predict([input_line])
    return res[0][0]
