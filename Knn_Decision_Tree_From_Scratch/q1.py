import pandas as pd
import numpy as np
import math
import operator
import sys
import matplotlib.pyplot as plt
import statistics 
from statistics import mode
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, r2_score
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics import classification_report

class KNNClassifier:
    
    def __init__(self):
        name=""    
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()
    
    def train(self,filename):
        self.train_data = (pd.read_csv(filename,header = None)).values
    
    
    def Read(self,filename):
        self.test_data = (pd.read_csv(filename, header = None)).values
        return self.test_data

    
    def find_distance_label(self,train_data , test , k , DistanceFlag):
        distance = []
        label = []
        for point in self.train_data:
            point1 = point[1:]
            point2 = test
            if(DistanceFlag==0):
                dist = np.linalg.norm( point2 - point1)
            else :
                 dist = np.sum([abs(x - y) for x, y in zip(point1, point2)])
            distance.append(dist)
            label.append(point[0])
        distanceLabelPair = [list(x) for x in zip(distance, label)]
        distanceLabelPair.sort()
        labelList = zip(*distanceLabelPair)[1]
        labelList = labelList[:k]
        most_frequent = max(set(labelList), key=labelList.count)
        return most_frequent
    
      
    def evaluate_result( self, y_test, y_pred):
        matrix = confusion_matrix(y_test, y_pred)
        df = pd.DataFrame(matrix)
        df = df.transpose()
        print(df.head())
        f1 = f1_score(y_test, y_pred, average="macro")
        ps = precision_score(y_test, y_pred, average="macro")
        rs = recall_score(y_test, y_pred, average="macro")
        acs = accuracy_score(y_test, y_pred)
        print("accuracy_score\t : ", acs)
        return acs
        
    
    def evaluate_result1( self, y_test, y_pred):
        matrix = confusion_matrix(y_test, y_pred)
        df = pd.DataFrame(matrix)
        #df = df.transpose()
        f1 = f1_score(y_test, y_pred, average="macro")
        ps = precision_score(y_test, y_pred, average="macro")
        rs = recall_score(y_test, y_pred, average="macro")
        acs = accuracy_score(y_test, y_pred)
        return f1


    
    def Predict_for_validation(self,TestData, TrainData, TestLabel, TrainLabel):
        solution=[]
        k=3
        DistanceFlag =0
        for test in self.test_data:
            solution.append(find_distance_label(self.train_data,test,k,DistanceFlag))
        return solution
        
    def predict(self,filename):
        self.test_data=self.Read(filename)
        solution=[]
        k=3
        DistanceFlag =0
        for test in self.test_data:
            solution.append(self.find_distance_label(self.train_data,test,k,DistanceFlag))
        return solution
     
    def validate(self,TestSet,TrainSet,TestLabel):
        solution=[]
        k=1
        DistanceFlag =0
        i=0
        num = []

        for test in TestSet:
            solution.append(find_distance_label(TrainSet,test,k,DistanceFlag))
        return solution
    
        def validate_f1(self,TestSet,TrainSet,TestLabel):
            solution=[]
            #k=1
            DistanceFlag =0
            i=0
            num = []
            for k in range(1,6):
                solution = []
                for test in TestSet:
                    solution.append(find_distance_label(TrainSet,test,k,DistanceFlag))
                num.append(self.evaluate_result1(TestLabel , solution))
            return num
    
        

    

    

    
