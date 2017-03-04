# -*- coding: utf-8 -*-

"""
Hierarchical Clustering

Created on Fri Mar  3 19:20:27 2017

@author: bharav4
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#import the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values
                
#using dendrogram method to know the best no of cluster
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('eculedian distance')
plt.show()


#fitting H C to the mall dataset 
from sklearn.cluster import AgglomerativeClustering
HClustering = AgglomerativeClustering(n_clusters = 5,affinity='euclidean', linkage='ward')
y_hc = HClustering.fit_predict(X)

#Vizualizing the cluster 
plt.scatter(X[y_hc == 0, 0],X[y_hc == 0,1], s = 100, c='red', label = 'Careful')
plt.scatter(X[y_hc == 1, 0],X[y_hc == 1,1], s = 100, c='green', label = 'Standard')
plt.scatter(X[y_hc == 2, 0],X[y_hc == 2,1], s = 100, c='blue', label = 'Target')
plt.scatter(X[y_hc == 3, 0],X[y_hc == 3,1], s = 100, c='pink', label = 'Careless')
plt.scatter(X[y_hc == 4, 0],X[y_hc == 4,1], s = 100, c='grey', label = 'Sensible')
plt.title('Clusters of client')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1 - 100)')
plt.legend()
plt.show()