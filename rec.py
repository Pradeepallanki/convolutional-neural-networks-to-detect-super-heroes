#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 19:41:45 2019

@author: pradi
"""
# Import essential libraries to build the neural network
# This is a convolutional neural network to identify the super heroes in the given image
import os
import numpy as np
import pandas as pd

from skimage import io, transform
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import MaxPooling2D, Flatten, Dropout
from keras.layers import Dense, Activation, Convolution2D

NB_EPOCH=10       # epoch size
IMAGE_SIZE=50     # I have transformed my images to the size 50

# TRansform the image
def transform_img(image):
    return transform.resize(image, (IMAGE_SIZE, IMAGE_SIZE, image.shape[2]))


# cnn model
# For 2 layers we shall use the relu as an activation function & for output layer softmax to get maximum accuracy
# Its ok to use any activation function, such as sigmoid
def CNN():
    model = Sequential()
    model.add(Convolution2D(8, 3, 3, border_mode='same',
                            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))
    model.add(Convolution2D(16, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))
    model.add(Flatten())
    model.add(Dense(12))
    model.add(Activation('softmax'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Load the traning and testing data
def load_train_data():
    images = os.listdir('traindata/train')
    train_data=[]
    train_data_name=[]
    
    
    for image in images:
        if image[-4:]=='.jpg':
            train_data_name.append(image)
            transformed_image= transform_img(io.imread('traindata/train/'+image))
            train_data.append(transformed_image)
    return train_data, train_data_name

# This function  is to load test data
def load_test_data():
    images = os.listdir('testdata/test')
    test_data=[]
    test_data_name=[]
    
    
    for image in images:
        if image[-4:]=='.jpg':
            test_data_name.append(image)
            transformed_image= transform_img(io.imread('testdata/test/'+image))
            test_data.append(transformed_image)
    return test_data, test_data_name
    


# call laod_train_data() to load the traning dataset and name of  the images as well    
train_data, data_name=load_train_data()

# in this csv file, we have image name along with labels, just load them
labels_name = pd.read_csv('superhero.csv')
data=pd.DataFrame(labels_name)



#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# This part is implemented because to make it easy to retrieve the
# labels based on the file name
# Here the dictionary has file_name mappped to the labels
# So that you can easily retrieve the labels based on the file_name 

store_data={}  # This is a dictionary(map), which is used to store the file name along with respective labels


name=[] # list which stores name of the image
files=[] # list which stores the labels

for d in data['filename']:
    name.append(d+'.jpg')   # append the jpg to the end of the name, since we have used the same pattern to store the labels  in load_train_dataset(), Means the returned labelname from laod_dataset() have this pattern
for d in data['Superhero']: # Store who is super hero?
    files.append(d)

for i in range(len(name)):
    store_data[name[i]]=files[i]  # store the data in dictionary, in the form ------ name_of_the_file-----> labels



#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

final_labels=[] # this is to store the label name

for name in data_name: 
    final_labels.append(store_data[name]) # Now actually final_labels has labels for the data present in train_data

my_train_data=np.array(train_data) # Convert the train data to array np.array type

temp = set()  # This is to get the all possible super heroes. In this example we have 12 super heroes possible
              # Since set() Doesn't allow you to store duplicates,its easy to get the distinct values
for labels in final_labels:
    temp.add(labels)

#------------------------------------------------------------------------------------------------
#This part is to get an one-hot array
# since there are total of 12 super heroes, Lets create a list which stores 12 values in it
# 0- ant-man .. and goes on
# for a data, which has 'ant-man' label 0th index will be set to 1 , others will be 0, This tredition will be implemented for every labels 

int_label = {}
count=0
for labels in temp:
    int_label[labels]=count
    count+=1
a_label=[]

for label in final_labels:
    b_label=[]
    
    for i in range(12):
        b_label.append(0)
    b_label[int_label[label]]=1
    a_label.append(b_label)
#-----------------------------------------------------------------------------------------------


    





my_labels = np.array(a_label)

model=CNN() # call the model
model.fit(my_train_data, my_labels, nb_epoch=NB_EPOCH) # fit the data to the model, means train the model

test_data, data_n=load_test_data()  # load the test data
test_labels=[]   # this is stores the labels
for name in data_n:
    test_labels.append(store_data[name])
my_test_data=np.array(test_data)


# SImilar to the training label creation, thats creating one hot array
c_label=[]
for label in test_labels:
    b_label=[]
    
    for i in range(12):
        b_label.append(0)
    b_label[int_label[label]]=1
    c_label.append(b_label)


my_test_labels=np.array(c_label)

preds = np.argmax(model.predict(my_test_data), axis=1) # predict the output 
test_labels = np.argmax(my_test_labels, axis=1) # Real test labels
acc=accuracy_score(test_labels, preds) # Compare, the predicted output with the actual aoutput
accuracy= (acc)*100 # calculate the accuracy  in terms of 100%

print("Accuracy of the neural Network is "+str(accuracy)) # print the accuracy


    
    


