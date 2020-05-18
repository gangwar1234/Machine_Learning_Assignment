import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from sklearn.metrics import confusion_matrix , accuracy_score , f1_score, precision_score , recall_score, r2_score
from sklearn.neighbors import KNeighborsClassifier
import sys
import statistics 
from statistics import mode
class DecisionTree :
    
    def __init__(self):
        name=""
        
    sub_tree =  dict()
    
    def FillWithUniqueContinuous(self,train_data,col):
        # print(" in FillWithUniqueContinuous ")
        df  = train_data[col]
        lis = df.values
        lis = list(set(lis))
        val = 0
        lis1=[]
        if(len(lis) == 1):
            return lis
        for i in range(len(lis)-1):
            lis1.append((lis[i]+lis[i+1])/2)
        return  lis1
    
    def FillWithUniqueCat(self,train_data,Col):
    # print( " in FillWithUniqueCat ")
        df  = train_data[Col]
        lis = df.values
        mylist = list(set(lis))
        return mylist
    
    
    def FillMissingValues(self,train_data_frm):
    
        fill_col=['YrSold','MoSold','MiscVal','PoolArea','ScreenPorch',
                  '3SsnPorch','EnclosedPorch','OpenPorchSF','WoodDeckSF',
                  'GarageArea','GarageCars','GarageYrBlt','Fireplaces',
                  'TotRmsAbvGrd','Kitchen','Bedroom','LotFrontage','LotArea',
                  'YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2',
                  'BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF',
                  'GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath']
        for val in train_data_frm.columns.values:
            if(val in fill_col):
                train_data_frm[val] = train_data_frm[val].fillna((train_data_frm[val].mean()))
            else:
                train_data_frm[val] = train_data_frm[val].fillna((train_data_frm[val].mode()[0]))      
        return train_data_frm

    def PrepareData(self,train_data_frm):
        #print(train_data_frm.columns.values)
        train_data_frm = self.DropColumns(train_data_frm)
        train_data_frm = self.FillMissingValues(train_data_frm)
        df = train_data_frm
        nan_rows = df[df.isnull().T.any().T]
        return train_data_frm

#train_data= pd.read_csv("train.csv")
#PrepareData(train_data)
    
    def DropColumns(self,train_data_frm):
        train_data_frm = train_data_frm.drop('Alley', axis=1)
        train_data_frm = train_data_frm.drop('PoolQC', axis=1)
        train_data_frm = train_data_frm.drop('Fence', axis=1)
        train_data_frm = train_data_frm.drop('MiscFeature', axis=1)
        train_data_frm = train_data_frm.drop('Id', axis=1)

        return train_data_frm

    def check_categorical(self,column):
            fill_col=['YrSold','MoSold','MiscVal','PoolArea','ScreenPorch',
                  '3SsnPorch','EnclosedPorch','OpenPorchSF','WoodDeckSF',
                  'GarageArea','GarageCars','GarageYrBlt','Fireplaces',
                  'TotRmsAbvGrd','Kitchen','Bedroom','LotFrontage','LotArea',
                  'YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2',
                  'BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF',
                  'GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath']
            if(column in fill_col):
                return False
            return True 

    
    
    def train(self,filename):
        train_data = pd.read_csv(filename)
        train_data = self.PrepareData(train_data)
        self.sub_tree = self.DecisionTrees(train_data,0,2,5)
        
        
    def predict(self,filename):
        test_data = pd.read_csv(filename)
        test_data = self.PrepareData(test_data)
        return [self.classifyMean(row, self.sub_tree) for i, row in test_data.iterrows()]
    
    def make_dictionary(self,train_data,NewDict):
        for col in train_data.columns.values:
            if self.check_categorical(col):
                NewDict[col] = self.FillWithUniqueCat(train_data,col)
            else :
                NewDict[col] = self.FillWithUniqueContinuous(train_data,col)
        return NewDict
        
    
    
    def FindMeanSquareError(self,train_data ,  attribute,col):
        left  = train_data[train_data[col] == attribute]
        right = train_data[train_data[col] != attribute]
        mleft = left['SalePrice'].mean()
        mright= right['SalePrice'].mean()
        summ  = 0
        summ1 = 0
        
        for val in left['SalePrice']:
            summ=summ+(val-mleft)*(val-mleft)
        
        for val in right['SalePrice']:
            summ1=summ1+(val-mright)*(val-mright)
        countUp = left.shape[1]
        countDown = right.shape[1]
        over_all_mean =  ((countUp*summ)/(countUp + countDown))  +  ((countDown*summ1)/(countUp + countDown))
        return [over_all_mean , col, attribute] 
            
    
    def FindUpAndDown(self,train_data,lis,col):
        #print(" in FindUpAndDown ")
        MSE = []
        i=0
        for attribute in lis:
                MSE.append(self.FindMeanSquareError(train_data,attribute,col))
        # MSE = [self.FindMeanSquareError(train_data, attribute, col) for attribute in lis].sort()[0]
        #print(MSE)    
        MSE.sort()
        #print(MSE[0])
        return MSE[0]
    

    def FindMeanSquareErrorNUm(self,train_data, left, right,col,x):
        meanl = left['SalePrice'].mean()
        meanr = left['SalePrice'].mean()
        summ =0
        for i in left['SalePrice']:
            summ = summ + (i -meanl)*(i - meanl)   
        summ1=0
        for i in right['SalePrice']:
            summ1 = summ1 + (i -meanr)*(i - meanr)
        total = len(left) + len(right)
        overAllMean = ((summ*len(left))/total) + ((summ1*len(right))/total)
        #print(" mean num = ")
        #print(overAllMean)
        
        return [overAllMean,col, x] 
            
    
    def FindUpAndDownForNUm(self,train_data, lis, col):
        #print(" in FindUpAndDownForNUm ")

        MSE = []
        i=0
       # print(" lis = ")
        #print(len(lis))
        for i in range(len(lis)):
            left = train_data[train_data[col] <= lis[i]]
            right= train_data[train_data[col] > lis[i]]
            MSE.append(self.FindMeanSquareErrorNUm(train_data, left, right,col,lis[i]))
            i=i+1
            
        MSE.sort()
        #print(len(MSE))
        #print(MSE[0])
        #print(MSE)
        return MSE[0]
    ## alright*****************************        
    def check_purity(self,data):
        label_column = data['SalePrice'].values
        unique_classes = np.unique(label_column)
        if len(unique_classes) == 1:
            return True
        return False
        
    def make_split(self,train_data):
       # print("in make split")
        val = []
        NewDict = dict()
        NewDict = self.make_dictionary(train_data,NewDict )
        #print(" dict size = ")
        #print(len(NewDict))
        column = train_data.columns.values
        #print(column)
        for col in column:
            if col == 'SalePrice':
                continue
            if self.check_categorical(col):
                lis = NewDict[col]
                val.append(self.FindUpAndDown(train_data , lis, col))
            else :
                lis = NewDict[col]
                val.append(self.FindUpAndDownForNUm(train_data , lis, col))
        val.sort()
        return val[0]
    
    
    def split_data(self,data , col , attribute):
        if self.check_categorical(col):
            data_below = data[data[col] == attribute]
            data_above = data[data[col] != attribute]
        else:
            data_below = data[data[col] <= attribute]
            data_above = data[data[col] > attribute]
        return data_below, data_above
    
    #*********
    def classify_data(self,data):
        label_column = data['SalePrice']
        unique_classes, counts_unique_classes = np.unique(
            label_column,return_counts = True)
        index = counts_unique_classes.argmax()
        classification = unique_classes[index]
        return classification
    
    
    def MakeQuestion(self,feature_name,attribute):
        question = "{} <= {}".format(feature_name,attribute)
        return question
    
    def CheckTerminatingCondition(self,data,min_samples,counter,max_depth):
        if (self.check_purity(data)) or  (data.shape[1] < min_samples) or (counter == max_depth) :
            return True
        return False
    
    def DecisionTrees(self,train_data, counter,min_samples, max_depth):
        #print(" IN decision tree ")
        if counter == 0:
            global col 
            col = train_data.columns
        data = train_data
            
        if (self.CheckTerminatingCondition( data,min_samples,counter,max_depth)):
            classification = self.classify_data(data)
            return classification
        counter+=1
            #print(" data ")
            #print(data.shape)
        mean, c , attribute = self.make_split(data)
            #print(" mean c attribute ")
            #print(mean)
            #print(c)
            #print(attribute)
        data_below, data_above = self.split_data(data, c, attribute)
        feature_name   = c
        question       = self.MakeQuestion(feature_name,attribute)
        sub_tree       = {question : []}
        BelowAnswer    = self.DecisionTrees(data_below, counter,2,5)
        AboveAnswer    = self.DecisionTrees(data_above, counter,2,5)
        if BelowAnswer == AboveAnswer:
            self.sub_tree = BelowAnswer
        else :
            sub_tree[question].append(BelowAnswer)
            sub_tree[question].append(AboveAnswer)

        return sub_tree
    
    def SplitQuestion(self,question):
        return question.split(" ")
    
    def classifyMean(self,Row, tree):
        question = list(tree.keys())[0]
        feature_name, comparison_operator, value = self.SplitQuestion(question)
        if(self.check_categorical(feature_name)):
            if Row[feature_name] == value:
                answer = tree[question][0]
            else:
                answer = tree[question][1]

        else:
            if Row[feature_name] <=float(value):
                answer = tree[question][0]
            else :
                answer = tree[question][1]

        if not isinstance(answer, dict):
            return answer

        else:
            restree = answer
            return self.classifyMean(Row, restree)
    

   