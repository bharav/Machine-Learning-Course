#EClat Model

#Data Peocessing
setwd("C://Vivek//Learnings//Machine Learning Udemy")
#install.packages('arules')
library(arules)
#Read Data Set
#dataset = read.csv("Market_Basket_Optimisation.csv", header = FALSE)
dataset = read.transactions(file = 'Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)

summary(dataset)
itemFrequencyPlot(dataset, topN=10)

#Training Apriori on the dataset

rules = eclat(data = dataset, parameter = list(support = 0.004, minlen = 2))

#Vizualising the results

inspect(sort(rules, by = 'support')[1:10])
