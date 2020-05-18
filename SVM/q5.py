
import pickle
import string 
import nltk
from unidecode import unidecode
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.svm import SVC
from sklearn import svm
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pandas as pd
import seaborn as sns
from pprint import pprint
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, r2_score
from heapq import heappush, heappop
import sys
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

class AuthorClassifier:
    
    X_train = Y_train = X_test = Y_test = ""
    vectorizer_x = TfidfVectorizer()
    
    def __init__(self):
        name = "name"
        
    def Read(self,filename):
        TrainData = pd.read_csv(filename)
        return TrainData
    
    def PreProcessing(self,Data,a):
        #print('shape - ', Data.shape[0])
        #print(Data.iloc[0,:])
        for i in range(Data.shape[0]):
            before = Data
            #remove numbers
            feature = Data.iloc[i,:]

            feature = ''.join(i for i in feature if not i.isdigit())
            #remove punctuation
            exc = set(string.punctuation)
            feature = ''.join(ch for ch in feature if ch not in exc)
            #print(feature)
            #encoding to ascii
            feature = unidecode(feature)
            feature = feature.lower()
            #tokenize
            tokenizer = RegexpTokenizer("\w+|\d\.]+|\S+")
            before = Data
            tokens = tokenizer.tokenize(feature)
            stopWords = set(stopwords.words('english'))
            val = 10
            New_Data = [w for w in tokens if not w in stopWords]
            Data.iloc[i,:] = ' '.join([str(w) for w in New_Data])
        return Data
    
    def results(self,y_test,y_pred):
        f1 = f1_score(y_test, y_pred, average="macro")
        ps = precision_score(y_test, y_pred, average="macro")
        rs = recall_score(y_test, y_pred, average="macro")
        acs = accuracy_score(y_test, y_pred)
        return np.array([acs, ps, rs, f1])
    
    
    def train(self,filename):
        Data = self.Read(filename)
        Data.columns = ['a','b','c']
        self.X_train  = Data.drop(['a','c'],axis = 1)
        self.X_train  = self.PreProcessing(self.X_train,"a")
        self.Y_train  = Data['c']
        self.X_train = self.X_train.values.flatten()
        self.X_train = self.vectorizer_x.fit_transform(self.X_train)
    
    
    def PredictionUsingSvm(self):
        SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
        SVM.fit(self.X_train,self.Y_train)
        predictions_SVM = SVM.predict(self.X_test)
        return predictions_SVM
    
    def predict(self,filename):
        Data = self.Read(filename)
        Data.columns = ['a','b']
        self.X_test  = Data.drop(['a'],axis = 1)
        self.X_test  = self.PreProcessing(self.X_test,"a")
        self.X_test = self.X_test.values.flatten()
        self.X_test = self.vectorizer_x.transform(self.X_test)
        return self.PredictionUsingSvm().flatten()
    
    def results(self,y_test,y_pred):
        f1 = f1_score(y_test, y_pred, average="macro")
        ps = precision_score(y_test, y_pred, average="macro")
        rs = recall_score(y_test, y_pred, average="macro")
        acs = accuracy_score(y_test, y_pred)
        return np.array([acs, ps, rs, f1])