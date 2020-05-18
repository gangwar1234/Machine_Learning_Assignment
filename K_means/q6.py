
import pickle
import nltk
import random 
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
import string
import nltk
from unidecode import unidecode
from nltk.tokenize import RegexpTokenizer
from statistics import mode
from sklearn.svm import LinearSVC
from os import listdir
from os.path import isfile, join
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

class Cluster:
    
    X_data = ""
    Y_data = ""
    X_train = ""
    centroids =[]
    classes  = {}
    onlyfiles = []
    LabelClasses = {}
    def __init__(self, k = 5, tolerance = 0.0001, max_iterations = 200, start =0,end =1725  ):
        self.k = k
        self.tolerance = tolerance 
        self.max_iterations = max_iterations
        self.start = start
        self.end = end

    def PreProcessing(self,Data):
        #print('shape - ', Data.shape[0])
        #print(Data.iloc[0,:])
        for i in range(Data.shape[0]):
            #remove numbers
            feature = Data[i]

            feature = ''.join(i for i in feature if not i.isdigit())
            #remove punctuation
            exclude = set(string.punctuation)
            feature = ''.join(ch for ch in feature if ch not in exclude)
            #print(feature)
            #encoding to ascii
            feature = unidecode(feature)
            feature = feature.lower()
            #tokenize
            tokenizer = RegexpTokenizer("\w+|\d\.]+|\S+")
            tokens = tokenizer.tokenize(feature)
            stopWords = set(stopwords.words('english'))
            New_Data = [w for w in tokens if not w in stopWords]
            Data[i] = ' '.join([str(w) for w in New_Data])
        return Data
    
    def Read1(self,Path):
        self.onlyfiles = [f for f in listdir(Path) if isfile(join(Path, f))]
        self.X_train = np.empty(shape = (0,0))
        for filename in self.onlyfiles:
            filePath = Path + filename
            with open(filePath,encoding="utf8", errors='ignore') as f:
                lis_content = f.read()
            f.close()
            #print(lis_content)
            self.X_train = np.append(self.X_train,lis_content)
            #print(self.X_train.shape)
            #print(self.X_train)
        self.X_train = self.PreProcessing(self.X_train)
        vectorizer_x = TfidfVectorizer()
        self.X_train = vectorizer_x.fit_transform(self.X_train)
        self.X_train =self.X_train.toarray()
    def FileCorrespondsToCluster(self):
        files = {}
        h = 0
        for list1 in self.LabelClasses:
            for point in self.LabelClasses[list1]:
                files[self.onlyfiles[point]] = h
            h=h+1
        return files
        
    
            
    def cluster(self,path):
        self.Read1(path)
        for i in range(self.k):
            l = random.randint(self.start, self.end)
            self.centroids.append(self.X_train[l])
        #print(len(self.centroid))
        for i in range(self.max_iterations):
            self.classes = {}
            self.LabelClasses = {}
            for i in range(self.k):
                self.classes[i] = []
                self.LabelClasses[i] = [] 
            Data = self.X_train
            for features in range(Data.shape[0]):
                distances = [np.linalg.norm(Data[features] - centroid) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classes[classification].append(Data[features])
                self.LabelClasses[classification].append(features)
            previous = self.centroids
            for classification in self.classes:
                self.centroids[classification] = np.average(self.classes[classification],axis =0) 
        ReturnFile = self.FileCorrespondsToCluster()
        return ReturnFile
                
            
    def Read(self,PathOf):
        self.X_data = np.empty(shape = (0,0))
        self.Y_data = np.empty(shape = (0,0))
        print(len(onlyfiles))
        for filename in onlyfiles:
            path = filename
            file = filename.split('.',1)[0]
            label = file.split('_')[1]
            filePath = './dataset/'+path
            with open(filePath,encoding="utf8", errors='ignore') as f:
                lis_content = f.read()
            f.close()
            self.Y_data = np.append(self.Y_data,label)
            self.X_data = np.append(self.X_data,lis_content) 
        vectorizer_x = TfidfVectorizer()
        self.X_data = vectorizer_x.fit_transform(self.X_data)
        self.X_data =self.X_data.toarray()
        print(self.Y_data.shape)
        

    def K_means(self):
        for i in range(self.k):
            l = random.randint(self.start, self.end)
            self.centroids.append(self.X_data[l])
        #print(len(self.centroid))
        for i in range(self.max_iterations):
            self.classes = {}
            self.LabelClasses = {}
            for i in range(self.k):
                self.classes[i] = []
                self.LabelClasses[i] = [] 
            Data = self.X_data
            for features in range(Data.shape[0]):
                distances = [np.linalg.norm(Data[features] - centroid) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classes[classification].append(Data[features])
                self.LabelClasses[classification].append(features)
            previous = self.centroids
            for classification in self.classes:
                self.centroids[classification] = np.average(self.classes[classification],axis =0) 
            flag = True
#             for centroid in range(len(self.centroids)):
#                 original_centroid = previous[centroid]
#                 curr = self.centroids[centroid]
#                 print(' original_centroid ',original_centroid)
#                 print(' current_centroid ', curr)
#                 if np.sum((curr - original_centroid)) > self.tolerance:
#                     flag = False
#             if(flag):
#                 break
        return self.classes,self.LabelClasses
    