setwd('C://Vivek//Learnings//Machine Learning Udemy')

#Reading DataSet
dataset = read.csv("50_Startups.csv")

#encoding categorical variables
dataset$State = factor(dataset$State,
                       levels = c('New York','California','Florida'),
                       labels = c(1,2,3))

#split dataset into training and test 
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8 )
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#fitting Multiple linear regression to training set 

#regressor = lm(formula = Profit ~ R.D.Spend + Administration + Market.Spend + State,data = training_set)
regressor = lm(formula = Profit ~ .,
               data = training_set)

#Predict
y_pred = predict(regressor,
                 newdata = test_set)

#Backward elimination
#iteration1
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data = dataset)
summary(regressor)
#iteration2
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
               data = dataset)
summary(regressor)
#iteration3
regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
               data = dataset)
summary(regressor)

#iteration4
regressor = lm(formula = Profit ~ R.D.Spend,
               data = dataset)
summary(regressor)