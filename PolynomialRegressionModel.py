# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 22:41:25 2017

@author: vivek
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)                    
lin_reg2 = LinearRegression() 
lin_reg2.fit(X_poly,Y) 

#vizualize linear regression
plt.scatter(X,Y, color='red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title("Truth or Bluff");
plt.xlabel('Level')     
plt.ylabel('Salary')    

#vizualize polynomial regression 
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)   
plt.scatter(X,Y, color='red')
plt.plot(X_grid,lin_reg2.predict(poly_reg.fit_transform(X_grid)),color='blue')
plt.title("Truth or Bluff");
plt.xlabel('Level') 
plt.ylabel('Salary')                  

#predicting employee salary using linear 
lin_reg.predict(6.5)   

#predicting employee salary using polynomial
lin_reg2.predict(poly_reg.fit_transform(6.5))

          

