#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 13:07:42 2018

@author: mutecypher
"""
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy 
import pandas as pd

    
def kmeans(X, k, th):
    if k < 2:
        print('k needs to be at least 2!')
        return
    if (th <= 0.0) or (th >= 1.0):
        print('th values are beyond meaningful bounds')
        return    
    
    N, m = X.shape # dimensions of the dataset
    Y = np.zeros(N, dtype=int) # cluster labels
    C = np.random.uniform(0, 1, [k,m]) # centroids
    d = th + 1.0
    dist_to_centroid = np.zeros(k) # centroid distances
    
    while d > th:
        C_ = deepcopy(C)
        
        for i in range(N): # assign cluster labels to all data points            
            for j in range(k): 
                dist_to_centroid[j] = np.sqrt(sum((X[i,] - C[j,])**2))                
            Y[i] = np.argmin(dist_to_centroid) # assign to most similar cluster            
            
        for j in range(k): # recalculate all the centroids
            ind = FindAll(Y, j) # indexes of data points in cluster j
            n = len(ind)            
            if n > 0: C[j] = sum(X[ind,]) / n
        
        d = np.mean(abs(C - C_)) # how much have the centroids shifted on average?
        
    return Y, C

points =[(1,1), (1,2), (2,1), (2,2), (0,0), (0,1), (1,0), (1.5, 1.5), (0.5, 0.5), (1.5, 0.5), (0.5, 1.5)]

##
def normalize(X): # max-min normalizing
    N, m = X.shape
    Y = np.zeros([N, m])
    
    for i in range(m):
        mX = min(X[:,i])
        Y[:,i] = (X[:,i] - mX) / (max(X[:,i]) - mX)
    
    return Y

points = pd.DataFrame()
points.loc[:,"ex"] = [1,1,2,2,10,11,12,12,1,1,2,2]
points.loc[:,"why"] = [1,2,1,2,1,2,1,2,11,12,11,12]
points = points.values

##
def normalize(X): # max-min normalizing
    N, m = X.shape
    Y = np.zeros([N, m])
    
    for i in range(m):
        mX = min(X[:,i])
        Y[:,i] = (X[:,i] - mX) / (max(X[:,i]) - mX)
    
    return Y

normy = normalize(points)


Y, C = kmeans(normy,3, 0.00001)

print(C)

plt.scatter(bloops[:,0], bloops[:,1])


## question 6
yuck = pd.DataFrame()
yuck.loc[:,0] =  [1,1,2,2,10,11,12,12,1,1,2,2]
yuck.loc[:,1] = [1,2,1,2,1,2,1,2,11,12,11,12]
yuck = yuck.as_matrix()

C = pd.DataFrame()
C.loc[:,0] = [1,2,11,1]
C.loc[:,1]= [1,2,1,11]
C = C.as_matrix()
    
    
def kmeanz(X, k, th):
    if k < 2:
        print('k needs to be at least 2!')
        return
    if (th <= 0.0) or (th >= 1.0):
        print('th values are beyond meaningful bounds')
        return    
    
    N, m = X.shape # dimensions of the dataset
    Y = np.zeros(N, dtype=int) # cluster labels
    d = th + 1.0
    dist_to_centroid = np.zeros(k) # centroid distances
    
    while d > th:
        C_ = deepcopy(C)
        
        for i in range(N): # assign cluster labels to all data points            
            for j in range(k): 
                dist_to_centroid[j] = np.sqrt(sum((X[i,] - C[j,])**2))                
            Y[i] = np.argmin(dist_to_centroid) # assign to most similar cluster            
            
        for j in range(k): # recalculate all the centroids
            ind = FindAll(Y, j) # indexes of data points in cluster j
            n = len(ind)            
            if n > 0: C[j] = sum(X[ind,]) / n
        
        d = np.mean(abs(C - C_)) # how much have the centroids shifted on average?
        
    return Y, C

Y, C <- kmeanz(yuck, 2,  0.001)



