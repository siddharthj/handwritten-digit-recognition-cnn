############################################
# Import necessary libraries
############################################
import sys
import tensorflow as tf
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import keras.models
import keras
import cv2
import skimage.transform as skt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils import to_categorical
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
from sklearn.metrics import accuracy_score
from scipy import ndimage, misc
from keras.utils import np_utils


'''Converts the image to black and white'''
def convertToBlackAndWhite(x_train, invert = False) :
    if invert == True:
        x_train[x_train < 0.40] = 0
        x_train[x_train >= 0.40] = 255
    else :
        x_train[x_train > 0.59] = 255
        x_train[x_train <= 0.59] = 0
    return x_train



'''Reshape and Normalize the values'''
def reshapeAndNormalize(x) : 
    # Reshape the image from 54X54 to 54X54X1
    x = x.reshape(x.shape[0], 54, 54, 1)
    # Making sure that the values are float so that we can get decimal points after division
    x = x.astype('float32')
    # Normalizing the RGB codes by dividing it to the max RGB value.
    x /= 255
    return x

'''Rezise the images from 300X300 to 54X54''' 
def resizeClassData(x_train):
    resized_data = []
    for i in range(x_train.shape[0]):
        item = x_train[i]
        img = item[45:245, 45:245]
        img = skt.resize(np.float32(img), (54, 54))       
        resized_data.append(img)
    resized_data = np.asarray(resized_data)
    return resized_data

''' Load Model from Json file'''
def loadModelFromDisk(modelName):
    json_file = open(modelName + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(modelName+".h5")
    print("Loaded model from disk.")
    loaded_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return loaded_model


'''Convert the predicted probabilities to labels '''
def cutoffThreshold(predicted_probabilities, predicted_classes,p):
    for i in range(predicted_probabilities.shape[0]) :
        predicted_probabilities[i][predicted_probabilities[i] < p] = 0
        if (not predicted_probabilities[i].any()):
            predicted_classes[i] = -1
    return predicted_probabilities, predicted_classes

''' PreProcess the test data'''
def preProcessTestData(x_test):
    x_test_class = convertToBlackAndWhite(resizeClassData(x_test))
    x_test_class = reshapeAndNormalize(x_test_class)
    print('Number of images in x_test Class', x_test_class.shape[0])
    return x_test_class

''' 
This is the method that we are supposed to call for prediction of labels
'''
def predict(testFileName):
    print('\n')
    x_test = np.load(testFileName)
    x_test = preProcessTestData(x_test)
    model = loadModelFromDisk('model')

    print('Get the prediction for the test dataset')
    predicted_probabilities = model.predict(x_test)
    predicted_classes = model.predict_classes(x_test)
    p = 0.53 # Probability Threshold value
    predicted_probabilities, predicted_classes = cutoffThreshold(predicted_probabilities, predicted_classes,p)

    print(predicted_classes)
    return np.array(predicted_classes)

