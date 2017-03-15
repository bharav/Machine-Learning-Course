# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 10:39:15 2017

@author: bharav4
"""

# Artificial Neural Network

# Installing Theano
 pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# 
Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html
# Installing Keras
# pip install --upgrade keras

#Part 1 - data preprocessing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values
Y = dataset.iloc[:,13].values
                
     
#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1]=labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:,2]=labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X=onehotencoder.fit_transform(X).toarray()
X = X[:,1:]            
                
#spliting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(X,Y,test_size=0.2, random_state=0)    


#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)    

#part 2 - Now lets make the ANN!
#Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense


#Initializing ANN
classifier = Sequential()

#Adding input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu',input_dim = 11))
#Adding second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

#Adding output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))#if there are more than 2category output_dim will be equal to no. of category and activation will ve softmax its digmoid for more than two category

#compiling ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the ANN to training set
classifier.fit(X_train, Y_train,batch_size = 10,nb_epoch = 100)

#Making the predictions and evaluating the model

#Predicting test set results
Y_Pred = classifier.predict(X_test)
Y_Pred = (Y_Pred > 0.5)

#Making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_Pred)