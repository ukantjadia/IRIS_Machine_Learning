# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 03:32:17 2022

@author: user
"""

# K nearest neighbors

from sklearn.datasets import load_iris


iris_dataset=load_iris()

print("Keys of iris dataset: \n{}".format(iris_dataset.keys()))
#print("iris dataset: \n{}".format(iris_dataset.values()))
target_names=iris_dataset['target_names']
print("iris dataset target names: \n{}".format(iris_dataset['target_names']))
## here we get our taraget names as types of data we have

#print("iris dataset data: \n{}".format(iris_dataset['data']))  ## this is numerical data we have to deal with 
## checking weather it is in form of array or not
print("iris dataset data-type: \n{}".format(type(iris_dataset['data']))) # it is in numpy.ndarray

# for attributes of data we have feature namess
print("iris dataset feature_names: \n{}".format(iris_dataset['feature_names'])) #
feature_names=iris_dataset['feature_names'] 

# now checking the amount of data available 
print("iris dataset amount/shape: \n{}".format(iris_dataset['data'].shape)) # so we are having the 150 different data of flowers and there diff feature name as well 

# taking sample of data ie. features name and data
print("iris dataset sample: \n{} \n {}".format(iris_dataset['feature_names'],iris_dataset['data'][:5]))
iris_data=iris_dataset['data']

# now we have two things target and target_names 
print("iris dataset target: \n {} \n {}".format(iris_dataset['target_names'],iris_dataset['target'])) # so from this we can say setosa=>0,versicolor=>1,virginica=>2
target=iris_dataset['target']

## Now we have seen or understan our data now we will do the spliting of data into test and train data 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)

# checking the shape of test and train data
test=print("X Test data: {}, y Test data: {}".format(X_test.shape,y_test.shape))
print("X train data: {}, y train data: {}".format(X_train.shape,y_train.shape))
#print(test)


## NOw have look at your data

# ispecting your data is good way to find the abnormalities,peculiarities 
# may be there is inconsistencies or unexpected measurements are very common
# scatter way is the best way to visualize the data 
## Panda have a func to create a pair plots calles scatter_matrix and diagonal is a histogram 
import pandas as pd
iris_datafram=pd.DataFrame(X_train,columns=iris_dataset.feature_names)
#pd.sc
import mglearn
grr=pd.plotting.scatter_matrix(iris_datafram,c=y_train,figsize=(15,15),marker='+',hist_kwds={'bins':20},s=60,cmap=mglearn.cm3,alpha=0.8)



## using the k nearest member algo

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1) ## this parameter can be change for changing the matching value of neighbour while predictin 

## training the x and y dataset
knn.fit(X_train,y_train)


## Making prediction 
# for making prediction suppose we find a species in wild with sepal lenght of 5cm, sepal width of 2.9cm,petal lenght of 1cm,petal width of 0.2cm. 
# so we have to find what is the name of this species

import numpy as np
X_new=np.array([[5,2.9,1,0.2]])

print(X_new.shape)

## making prediction
prediction=knn.predict(X_new)
print("Type of predicted species: {}".format(prediction))
print("Name of species: {}".format(iris_dataset['target_names'][prediction]))

# predicting with test data
print(X_test.shape)
print("\n\n\n")
#print(X_train)

# doing prediction
prediction2=knn.predict(X_test)
#print("Type of different predicted data is: {}".format(prediction2))
#print("\nName of species predicted: {} and the test data {} ".format(X_test,iris_dataset['target_names'][prediction2]))


## evaluating theprediction2 ie. result of X_test and we will compare it with y_test because that is the result we have
print("Type of different predicted data is: \n{}".format(prediction2))

## checking the score using mean between y_test and prediction2
print("Test set score : {:.3f}".format(np.mean(prediction2 == y_test)))

## wecan also find score using knn method 
print("test set socre : {} ".format(knn.score(X_test,y_test)))

