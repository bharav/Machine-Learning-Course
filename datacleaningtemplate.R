#Data Peocessing
setwd("c://vivek//learnings//Machine Learning Udemy")
dataset = read.csv("Data.csv")
dataset$Age = ifelse(is.na(dataset$Age),ave(dataset$Age, FUN = function(x) mean(x,na.rm = TRUE)),
                     dataset$Age)
dataset$Salary = ifelse(is.na(dataset$Salary),ave(dataset$Salary, FUN = function(x) mean(x,na.rm = TRUE)),
                     dataset$Salary)

#encoding categorical variables
dataset$Country = factor(dataset$Country, levels = c('France','Spain','Germany'), labels = c(1,2,3))
dataset$Purchased = factor(dataset$Purchased, levels = c('No','Yes'), labels = c(0,1))
#splitting the data into training set and test set 
install.packages('caTools')
library(caTools)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#feature scaling
training_set[,2:3] = scale(training_set[,2:3])
test_set[,2:3] = scale(test_set[,2:3])