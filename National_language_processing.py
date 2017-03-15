# -*- coding: utf-8 -*-
"""
Natural Language Processing
Created on Tue Mar  7 22:21:51 2017

@author: bharav4
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset 
dataset = pd.read_csv('Restaurant_Reviews.tsv', sep = '\t', quoting = 3 )


#Cleaning the texts
import re 
#import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    #removing un wanted words and steming 
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

#Creating the Bag of Words Model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

#spliting the dataset into Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(X,y,test_size=0.2, random_state=0)    


#fitting logistic regression to te training set
#Create your classifier
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,Y_train) 

#Predicting test set results
Y_Pred = classifier.predict(X_test)

#Making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_Pred)
                
true_positive = cm[1,1]
true_negative = cm[0,0]
false_positive = cm[0,1]
false_negative = cm[1,0]   

Accuracy = (true_positive + true_negative)/200
Precision = true_positive/(true_positive + false_positive)  
Recall =   true_positive/(true_positive+false_negative)   
f1_score = (2*Precision*Recall)/(Recall+Precision)   