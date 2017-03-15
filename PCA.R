#PCA
#Data Peocessing
setwd("C://Vivek//Learnings//Machine Learning Udemy")

#Read Data Set
dataset = read.csv("Wine.csv")


#Diving data into trainingset and test set
#install.packages('caTools')
library(caTools)
set.seed(123)
split= sample.split(dataset$Customer_Segment, SplitRatio=0.80)
training_set = subset(dataset, split==TRUE)
test_set = subset(dataset, split==FALSE)


#feature scaling
training_set[-14] = scale(training_set[-14])
test_set[-14]=scale(test_set[-14])

#pca
#install.packages('caret')
library(caret)
library(e1071)
pca = preProcess(x = training_set[-14], method = 'pca', pcaComp = 2)
training_set = predict(pca,training_set)
training_set = training_set[c(2,3,1)]
test_set = predict(pca, test_set)
test_set = test_set[c(2, 3, 1)]
#Fitting Logistic Regression to the training set
#training_set$Customer_Segment = factor(training_set$Customer_Segment)
classifier = svm(formula = Customer_Segment ~ .,
                 type = 'C-classification',
                 data = training_set,
                 kernel='linear')

#Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-3])
#y_pred = prediction(prob_pred,training_set$Customer_Segment)

#Making the Confusion Matrix
cm = table(test_set[,3], y_pred)

#Vizualizing 
#install.packages('ElemStatLearn')
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('PC1', 'PC2')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'Logistic Regression (Training set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 2, 'deepskyblue', ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, pch = 21, bg = ifelse(set[, 3] == 1,'blue3',ifelse(set[, 3] == 1, 'green4', 'red3')))

# Visualising the Test set results
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('PC1', 'PC2')
y_grid = predict(classifier, newdata = grid_set)
 
plot(set[, -3],
     main = 'Logistic Regression (Test set)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))