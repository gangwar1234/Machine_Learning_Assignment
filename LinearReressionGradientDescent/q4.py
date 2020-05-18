
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pandas as pd
import seaborn as sns
from pprint import pprint
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, r2_score
from heapq import heappush, heappop
import sys

class Weather:
    theta = ""
    def standardize(self,X):
        X_standardized = (X - X.mean()) /X.std()
        #print(X_standardized)
        return X_standardized
    def Normalise(self,df):
        normalized_df=(df-df.min())/(df.max()-df.min())
        return normalized_df
    def preprocessing(self,X,type = "std"):
        if(type == "std"):
            return self.standardize(X)
        else:
            return   self.Normalise(X)
        
    def GradientDescent(self,train_X, train_Y):
        iteration = 1000
        alpha = 0.008
        m = train_X.shape
        #print(m)
        for i in range(iteration):
            c = m
#           print(c)
            b =  np.sum(train_X * (train_X @ self.theta.T - train_Y))
            self.theta = self.theta - (alpha / len(train_X)) * np.sum(train_X * (train_X @ self.theta.T - train_Y), axis=0)
    def train(self,filename):
        train_XX = pd.read_csv(filename)
        train_X = train_XX.drop(['Formatted Date','Precip Type','Summary','Apparent Temperature (C)','Daily Summary'],axis = 1)
        train_Y = train_XX[['Apparent Temperature (C)']]
        train_Y.columns = ['Apparent Temperature (C)']
        train_X = self.preprocessing(train_X,"std")
        train_X = train_X.values
        ones    = np.ones([train_X.shape[0],1])
        train_X = np.concatenate((ones,train_X),axis=1)
        train_Y = train_Y.values
        self.theta   = np.zeros([1,train_X.shape[1]])
        self.GradientDescent(train_X, train_Y)
    
    def predict(self,filename):
        test_X = pd.read_csv(filename)
        test_X = test_X.drop(['Formatted Date','Precip Type','Summary','Apparent Temperature (C)','Daily Summary'],axis = 1)
        #test_X = WeatherDataTest_X
        test_X = self.preprocessing(test_X,"std")
        test_X = test_X.values
        ones   = np.ones([test_X.shape[0],1])
        test_X = np.concatenate((ones, test_X), axis =1)
        Res = (test_X @ self.theta.T).flatten()
        Y =  pd.read_csv('./Datasets/q4/test.csv')
        Y = Y['Apparent Temperature (C)']
        print("r 2 score :")
        print(r2_score(Y,Res))
        print(' mse ',mean_squared_error(Y,Res))
        return Res
