#!/usr/bin/env python
# coding: utf-8

# In[6]:


import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from scipy.stats import norm
np.random.seed(0)
import pickle
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')


# In[7]:


def load(name):
    file = open(name,'rb')
    data = pickle.load(file)
    file.close()
    return data


# In[8]:


def save(data,name):
    file = open(name, 'wb')
    pickle.dump(data,file)
    file.close()


# In[9]:


class GMM1D:
    def __init__(self,X,iterations,initmean,initprob,initvariance):
#     """initmean = [a,b,c] initprob=[1/3,1/3,1/3] initvariance=[d,e,f] """    
        self.iterations = iterations
        self.X = X
        self.mu = initmean
        self.pi = initprob
        self.var = initvariance
    
#     """E step"""

    def calculate_prob(self,r):
        for c,g,p in zip(range(3),[norm(loc=self.mu[0],scale=self.var[0]),
                                       norm(loc=self.mu[1],scale=self.var[1]),
                                       norm(loc=self.mu[2],scale=self.var[2])],self.pi):
            r[:,c] = p*g.pdf(self.X.flatten())
        for i in range(len(r)):
            r[i] = r[i]/(np.sum(self.pi)*np.sum(r,axis=1)[i])
        return r
    
    def plot(self,r):
        fig = plt.figure(figsize=(10,10))
        ax0 = fig.add_subplot(111)
        for i in range(len(r)):
            ax0.scatter(self.X[i],0,c=np.array([r[i][0],r[i][1],r[i][2]]),s=100)
        """Plot the gaussians"""
        for g,c in zip([norm(loc=self.mu[0],scale=self.var[0]).pdf(np.linspace(-20,20,num=60)),
                        norm(loc=self.mu[1],scale=self.var[1]).pdf(np.linspace(-20,20,num=60)),
                        norm(loc=self.mu[2],scale=self.var[2]).pdf(np.linspace(-20,20,num=60))],['r','g','b']):
            ax0.plot(np.linspace(-20,20,num=60),g,c=c)
    
    def run(self):
        
        for iter in range(self.iterations):

#             """Create the array r with dimensionality nxK"""
            r = np.zeros((len(self.X),3))  

#             """
#             Probability for each datapo


#          x_i to belong to gaussian g 
#             """
            r = self.calculate_prob(r)


#             """Plot the data"""
            self.plot(r)
            
#             """M-Step"""

#             """calculate m_c"""
            m_c = []
            # write code here
            for i in range(len(r[0])):
                m = np.sum(r[:,i])
                m_c.append(m) # For each cluster c, calculate the m_c and add it to the list m_c
            
            for k in range(len(m_c)):
                self.pi[k] = (m_c[k]/np.sum(m_c)) # For each cluster c, calculate the fraction of points pi_c which belongs to cluster c
            self.mu = np.sum(self.X.reshape(len(self.X),1)*r,axis=0)/m_c
            print ("Updated mu= ",self.mu)

            var_c = []
            #write code here
            for c in range(len(r[0])):
                var_c.append((1/m_c[c])*np.dot(((np.array(r[:,c]).reshape(180,1))*(self.X.reshape(len(self.X),1)-self.mu[c])).T,(self.X.reshape(len(self.X),1)-self.mu[c])))
            print ("Updated var= ", var_c)
            plt.show()


# In[11]:


# To run the code - 
data1 = load("dataset1.pkl")
data2 = load("dataset2.pkl")
data3 = load("dataset3.pkl")
data = np.stack((data1,data2,data3)).flatten()   
print (data.shape)
g = GMM1D(data,10,[-8,8,5],[1/3,1/3,1/3],[5,3,1])
g.run()


# In[ ]:





# In[ ]:




