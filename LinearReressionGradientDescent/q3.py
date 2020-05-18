

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pandas as pd
import seaborn as sns
from pprint import pprint
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, r2_score
from heapq import heappush, heappop
import sys



class Airfoil:
    theta = np.zeros([1,20])

    def standardize(self,X):
        X_standardized = (X - X.mean()) /X.std()
        return X_standardized
    def Normalise(self,df):
        normalized_df=(df-df.min())/(df.max()-df.min())
        return normalized_df
    def GradientDescent(self,train_X, train_Y):
        iteration = 10000
        alpha = 0.009
        m = train_X.shape
        for i in range(iteration):
            self.theta = self.theta - (alpha / len(train_X)) * np.sum(train_X * (train_X @ self.theta.T - train_Y), axis=0)
        
    def train(self,filename):
        NasaData = pd.read_csv(filename)
        NasaData.columns = ['A', 'B', 'C', 'D','E','F']
        train_X = NasaData.drop(['F'],axis = 1)
        train_Y = NasaData[['F']]
        train_Y.columns = ['F']
        train_X = self.preprocessing(train_X,"std")
        train_X = train_X.values
        ones    = np.ones([train_X.shape[0],1])
        train_X = np.concatenate((ones,train_X),axis=1)
        train_Y = train_Y.values
        self.theta   = np.zeros([1,train_X.shape[1]])
        #print("in train before gradient= ",type(self.theta))
        self.GradientDescent(train_X, train_Y)
        #print("in train = ",type(self.theta))
        return self.theta
    def predict(self,filename):
        #test_X = NasaDataTest_X
        test_X = pd.read_csv(filename)
        test_X = self.preprocessing(test_X,"std")
        test_X = test_X.values
        test_X = test_X[0:test_X.shape[0],0:2]
        ones   = np.ones([test_X.shape[0],1])
        test_X = np.concatenate((ones, test_X), axis =1)
#         A = self.theta.T
        #print("type ",type(self.theta))
        Res = (test_X @ self.theta.T).flatten()
        Y = pd.read_csv("./Datasets/q3/test.csv")
        Y = Y.values
        Y = Y[0:Y.shape[0],Y.shape[1]-1:Y.shape[1]]
        print(Y.shape)
        from sklearn.metrics import mean_squared_error
        print('R2 Score ',r2_score(Y,Res ))
        print(' MSE ',mean_squared_error(Y,Res))
        return Res
    def preprocessing(self,X,type = "std"):
        if(type == "Normalise"):
            return self.Normalise(X)
        else:
            return self.standardize(X)
    
