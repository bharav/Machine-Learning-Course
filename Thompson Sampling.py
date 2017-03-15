# -*- coding: utf-8 -*-
"""
Thompson Sampling 

Created on Tue Mar  7 20:11:16 2017

@author: bharav4
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

#import dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Implementing Thompson Sampling
N = 10000
d = 10
ads_selected = []
numbers_of_reward_1 = [0]*d
numbers_of_reward_0 = [0]*d 
total_reward = 0
for n in range(0, N):
        ad = 0
        max_random = 0
        for i in range(0, d):
            random_beta = rbeta(numbers_of_reward_1[i] + 1, numbers_of_reward_0[i] + 1)
            if random_beta > max_random:
               max_random = random_beta
               ad = i
        ads_selected.append(ad)
       
        reward = dataset.values[n, ad]
        if reward == 1:
            numbers_of_reward_1[ad] = numbers_of_reward_1[ad] + 1
        else:
            numbers_of_reward_0[ad] = numbers_of_reward_0[ad] + 1                       
        total_reward = total_reward + reward
        
plt.hist(ads_selected)        
plt.title('Histogram of ad selection')
plt.xlabel('Ads')
plt.ylabel('number of selection')
plt.show()
                            