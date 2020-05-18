import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from sklearn.metrics import confusion_matrix , accuracy_score , f1_score, precision_score , recall_score, r2_score
from sklearn.neighbors import KNeighborsClassifier
import sys
import statistics 
from statistics import mode


class KNNClassifier:
    TrainData = pd.DataFrame()
    TrainLabel=pd.DataFrame()
    
    def evaluate_result( self,y_test, y_pred):
        acs = accuracy_score(y_test, y_pred)
    
    def FindDistance(self,test_row, train_row):
        dis = np.sqrt(np.sum([(x - y)**2 for x, y in zip(test_row, train_row)]))
        return dis
    
    
    def Encoding(self , Set ):
        df=Set
        lis = []
        for i in range(0,1):
            a = ['b','c','x','f','k','s']
            lis.append(a)

        for i in range(0,1):
            a = ['f', 'g' , 'y', 's']
            lis.append(a)

        for i in range(0,1):
            a = ['n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w', 'y']
            lis.append(a)

        for i in range(0,1):
            a = ['t', 'f']
            lis.append(a)
               
        for i in range(0,1):
            a = ['a', 'l', 'c', 'y', 'f', 'm', 'n', 'p', 's']
            lis.append(a)

        
        for i in range(0,1):
            a = ['a', 'f', 'd', 'n']
            lis.append(a)

        


        for i in range(0,1):
            a = ['c', 'w' , 'd']
            lis.append(a)
                 


        for i in range(0,1):
            a = ['b', 'n']
            lis.append(a)


        for i in range(0,1):
            a = ['k', 'n' , 'b', 'h', 'g', 'r', 'o', 'p', 'u', 'e', 'w', 'y']
            lis.append(a)

        
        for i in range(0,1):
            a = ['e', 't']
            lis.append(a)         
                 
        

        for i in range(0,1):
            a = ['b','c','u','e','z','r']
            lis.append(a)   

        
        for i in range(0,1):
            a = ['f', 'y' , 'k', 's']
            lis.append(a) 
        
        for i in range(0,1):
            a = ['f', 'y' , 'k', 's']
            lis.append(a) 

        for i in range(0,1):
            a = ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'] 
            lis.append(a) 
        
        for i in range(0,1):
            a = ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'] 
            lis.append(a) 

        for i in range(0,1):
            a = ['p', 'u']
            lis.append(a) 



        for i in range(0,1):
            a = ['n', 'o', 'w', 'y']
            lis.append(a) 
                
        
        for i in range(0,1):
            a = ['n', 'o', 't']
            lis.append(a)        
        



        for i in range(0,1):
            a = ['c', 'e', 'f', 'l', 'n', 'p', 's', 'z']
            lis.append(a)    
        
        
        for i in range(0,1):
            a = ['k', 'n', 'b', 'h', 'r', 'o', 'u', 'w', 'y']
            lis.append(a)    
        
        for i in range(0,1):
            a = ['a', 'c', 'n', 's', 'v', 'y']
            lis.append(a)    


        for i in range(0,1):
            a = ['g', 'l', 'm', 'p', 'u', 'w', 'd']
            lis.append(a)  

        #print(df.shape)
        #print(Set.columns)
        col = Set.columns.values 
        i=0
        NewDf=pd.DataFrame()
        for column in col:
            dummies = pd.get_dummies(data=df[column],columns=lis[i])
            dummies=dummies.T.reindex(lis[i]).T.fillna(0)
            #print(dummies)
            NewDf=pd.concat([NewDf,dummies],axis=1,sort=False)
            i=i+1
        #print(NewDf)

        return NewDf

    
    
    def train(self,filename):
        TrainSet   = pd.read_csv(filename ,header = None)
        self.TrainData  = TrainSet
        self.TrainLabel = TrainSet[0]
        del self.TrainData[0]

    
    
    def predict(self,filename):
        TestSet = pd.read_csv(filename ,header = None)
        self.TrainData,TestSet = self.ModifyData(self.TrainData, TestSet)
        Solution = self.KnnForAll(self.TrainData, TestSet, self.TrainLabel,3)
        return Solution
    
        
    def KnnForAll(self,TrainSet,TestSet,TrainLabel,k):
        solution=pd.DataFrame()
        solution['predicted'] = [
            self.KNNForOne(TrainSet, row, TrainLabel,k) for i , row in TestSet.iterrows()
        ]
        return solution.values
    
    
    
    
    def KNNForOne(self,TrainSet, TestPoint,TrainLabel,k):
        distance=[]

        for (index , TrainPoint) in TrainSet.iterrows():
            dist=self.FindDistance(TrainPoint , TestPoint)
            distance.append(dist)
        distanceLabelPair = [list(x) for x in zip(distance, TrainLabel)]
        distanceLabelPair.sort()
        labelList = zip(*distanceLabelPair)[1]
        labelList = labelList[:k]
        most_frequent = max(set(labelList), key=labelList.count)
        return most_frequent
    
    def ModifyData(self,TrainSet, TestSet ):
        mTrain   = TrainSet.mode()[11][0]
        mTest    = TestSet.mode()[10][0]
        ColumnsNamesArr = TrainSet.columns.values
        TrainSet.replace(to_replace='?', value = mTrain , inplace = True)
        TestSet.replace(to_replace = '?',value = mTest , inplace  = True)
        return self.Encoding(TrainSet), self.Encoding(TestSet)



        

        
    
