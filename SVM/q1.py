#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pickle
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pandas as pd
import seaborn as sns
import scikitplot as skplt
from sklearn.decomposition import PCA
from pprint import pprint
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, r2_score
from heapq import heappush, heappop
import sys


# # Split Data 

# In[3]:


def SplitTestTrain(X, Y,percent = 0.8):
    mask = np.random.rand(len(X))<percent
    X_train = X[mask].dropna()
    X_test  = X[~mask].dropna()
    Y_train = Y[mask].dropna()
    Y_test  = Y[~mask].dropna()
    #print(X_train.shape, X_test.shape)
    X_train = X_train.reset_index(drop = True)
    X_test  = X_test.reset_index(drop = True)
    Y_train = Y_train.reset_index(drop = True)
    Y_test  = Y_test.reset_index(drop = True)
    return X_train.values, X_test.values, Y_train.values ,Y_test.values


# # Standardize Data

# In[4]:




def standardize(X):
    X_standardized = (X - X.mean()) /X.std()
    return X_standardized


# # PCA 
# # for preprocessing standardization and PCA has been used so that operation can be more accurate and efficient

# In[5]:




def pcaProcessing(X_data):
    pca = PCA(n_components = 50)
    pca.fit(X_data)
    X_data = pca.transform(X_data)
    return X_data


# # Unpickle File

# In[6]:



def Unpickle(data_batch):
    with open(data_batch,'rb') as fo:
        dic = pickle.load(fo,encoding = 'bytes')
    labels = np.asarray(dic[b'labels'])
    labels = np.reshape(labels,(-1,1))
    train_data = dic[b'data']
    #print(train_data)
    #print(train_data[:,0])
    #print('labels ',labels)
    #train_data = np.hstack((train_data,labels))
    #print(train_data.shape)
    train_data = pd.DataFrame(train_data)
    train_data = standardize(train_data)
    labels = pd.DataFrame(labels)
    #print(train_data)
    return train_data,labels
        

X,Y = Unpickle('data_batch_1')
X = pcaProcessing(X)
X =pd.DataFrame(X)
X_train, X_test, Y_train ,Y_test = SplitTestTrain(X, Y,percent = 0.8)
print(type(Y_train))
svm_model_linear = SVC(kernel = 'linear', C = 1,decision_function_shape = 'ovr').fit(X_train, Y_train.flatten())
svm_predictions = svm_model_linear.predict(X_test)
accuracy = svm_model_linear.score(X_test, Y_test) 
cm = confusion_matrix(Y_test, svm_predictions) 
print(' f1 - score ', f1_score(Y_test,svm_predictions , average='macro'))
print(' accuracy  score ', accuracy)


# # For C =1 Confusion_matrix 

# In[21]:


skplt.metrics.plot_confusion_matrix(Y_test, svm_predictions, normalize=True)
plt.show()


# # DATA VISUALIZATION

# In[23]:



#print(X_test)
#print(Y_test)
plt.scatter(X_test,X_test)
plt.show()


# # For C = 2

# In[25]:


svm_model_linear = SVC(kernel = 'linear', C = 2,decision_function_shape = 'ovr').fit(X_train, Y_train.flatten())
svm_predictions = svm_model_linear.predict(X_test)
accuracy = svm_model_linear.score(X_test, Y_test) 
cm = confusion_matrix(Y_test, svm_predictions) 
print(' f1 - score ', f1_score(Y_test,svm_predictions , average='macro'))
print(' accuracy ', accuracy)
skplt.metrics.plot_confusion_matrix(Y_test, svm_predictions, normalize=True)
plt.show()


# # For C = 5

# In[26]:


svm_model_linear = SVC(kernel = 'linear', C = 5,decision_function_shape = 'ovr').fit(X_train, Y_train.flatten())
svm_predictions = svm_model_linear.predict(X_test)
accuracy = svm_model_linear.score(X_test, Y_test) 
cm = confusion_matrix(Y_test, svm_predictions) 
print(' f1 - score ', f1_score(Y_test,svm_predictions , average='macro'))
print(' accuracy ', accuracy)
skplt.metrics.plot_confusion_matrix(Y_test, svm_predictions, normalize=True)


# # For  C = 100

# In[27]:


svm_model_linear = SVC(kernel = 'linear', C = 100,decision_function_shape = 'ovr').fit(X_train, Y_train.flatten())
svm_predictions = svm_model_linear.predict(X_test)
accuracy = svm_model_linear.score(X_test, Y_test) 
cm = confusion_matrix(Y_test, svm_predictions) 
print(' f1 - score ', f1_score(Y_test,svm_predictions , average='macro'))
print(' accuracy ', accuracy)
skplt.metrics.plot_confusion_matrix(Y_test, svm_predictions, normalize=True)


# # For All Five Files

# In[ ]:


X1,Y1 = Unpickle('data_batch_1')
X2,Y2 = Unpickle('data_batch_2')
X3,Y3 = Unpickle('data_batch_3')
X4,Y4 = Unpickle('data_batch_4')
X5,Y5 = Unpickle('data_batch_5')
X = np.vstack((X1,X2,X3,X4,X5))
Y = np.vstack((Y1,Y2,Y3,Y4,Y5))
X = pcaProcessing(X)
X =pd.DataFrame(X)
Y = pd.DataFrame(Y)
X_train, X_test, Y_train ,Y_test = SplitTestTrain(X, Y,percent = 0.8)
print(type(Y_train))
svm_model_linear = SVC(kernel = 'linear', C = 1,decision_function_shape = 'ovr').fit(X_train, Y_train.flatten())
svm_predictions = svm_model_linear.predict(X_test)
accuracy = svm_model_linear.score(X_test, Y_test) 
cm = confusion_matrix(Y_test, svm_predictions) 
print(' f1 - score ', f1_score(Y_test,svm_predictions , average='macro'))
print(' accuracy  score ', accuracy)


# # Scores and confusion matrix 

# In[7]:


print(' f1 - score ', f1_score(Y_test,svm_predictions , average='macro'))
print(' accuracy ', accuracy)
skplt.metrics.plot_confusion_matrix(Y_test, svm_predictions, normalize=True)


# # Graph Between C and accuracy

# In[10]:


C = [1,2,5,10,100]
acc = [ 0.4122 ,0.3841091658084449,0.3141091658084449,0.31153450051493303,0.31153450051493303]
plt.plot(C,acc)
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title('C vs Accuracy')
plt.show()


# # Classification report

# In[10]:


print('classification report')
print(classification_report(Y_test, svm_predictions))


# In[ ]:




