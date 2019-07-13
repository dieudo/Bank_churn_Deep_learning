#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 13:52:34 2019

@author: Dieudonne Ouedraogo

A simple deep newral network using python's deep learning package Keras 
to predict if a customer with specific charasteristic will churn as client
using articial neural network

"""




# Importing the  necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


#----------------Data Preprocessing--------------------

# Import the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
# locate the positions of the features to be used for the model
X = dataset.iloc[:, 3:13].values
#locate the position of the label or outcome /output
y = dataset.iloc[:, 13].values 

# Encode the categorical data to numeric data
# to feed the ANN, algorithms implemenation in sklearn and keras use numerics

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

#Remove dummie variable
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

# Feature Scaling

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#  ------------------Build the neural network model-------------------------


# Instantiation = Initialising the neural network object,it's a sequential model
classifier = Sequential()


#We have 11 features which are the input nodes, so input_dim =11: 
# We will add two hidden layers with 6 nodes each 
# We will use a relu activation function on those two hidden layer
# We need just one node as output node to represent the state of the churn, 1 or 0.
# We will use sigmoid as activation function on this node
# We use adam optimizer and binary_crossentropy as the loss
# we train on batch size of 10 and run 50 epochs over the data.

# First hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

# Second hidden 
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Add the Output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))


#Compiling the neural network
# binary_crossentropy loss function used when a binary output is expected
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) 

# Fit the classifier to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 50)



# Predict using the Test set 
y_pred = classifier.predict(X_test)

# Set up a treshold and filter who will leave or not
y_pred = (y_pred > 0.5)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm

