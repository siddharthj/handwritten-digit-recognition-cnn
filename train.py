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

'''Resize the image to 54X54 by padding the image - MNIST dataset'''
def padImage_MNIST(x_train):
    x = 28
    y = 28
    x_pad=54-x
    y_pad=54-y
    y_pad1=math.floor(y_pad/2)
    y_pad2=y_pad-y_pad1
    x_pad1=math.floor(x_pad/2)
    x_pad2=x_pad-x_pad1
    x_train_padded = []
    for i in range(len(x_train)):
        img = np.pad(x_train[i],[(x_pad1,x_pad2),(y_pad1,y_pad2)],mode='constant',constant_values=0)
        x_train_padded.append(cv2.bitwise_not(img))
    x_train_padded = np.array(x_train_padded)
    return x_train_padded

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

def addRotatedImages(x_train, y) :
    newImages = []
    newLabels = []
    for i in range(0, x_train.shape[0]):
        #newImages.append(x_train[i])
        a = x_train[i]
        newLabels.append(y[i])
        img_5 = ndimage.rotate(a, 5, reshape=False,cval = 255)
        img_10 = ndimage.rotate(a, 10, reshape=False, cval = 255)
        img_15 = ndimage.rotate(a, 15, reshape=False, cval = 255)
        img_20 = ndimage.rotate(a, 20, reshape=False,cval = 255)
        img_25 = ndimage.rotate(a, 25, reshape=False, cval = 255)
        img_30 = ndimage.rotate(a, 30, reshape=False, cval = 255)
        img_35 = ndimage.rotate(a, 35, reshape=False, cval = 255)
        img_40 = ndimage.rotate(a, 35, reshape=False, cval = 255)
        img_45 = ndimage.rotate(a, 45, reshape=False, cval = 255)
        img_M_5 = ndimage.rotate(a,  360 - 5, reshape=False,cval = 255)
        img_M_10 = ndimage.rotate(a, 360 -10, reshape=False, cval = 255)
        img_M_15 = ndimage.rotate(a, 360 -15, reshape=False, cval = 255)
        img_M_20 = ndimage.rotate(a, 360 - 20, reshape=False,cval = 255)
        img_M_25 = ndimage.rotate(a, 360 -25, reshape=False, cval = 255)
        img_M_30 = ndimage.rotate(a, 360 -30, reshape=False, cval = 255)
        img_M_35 = ndimage.rotate(a, 360 - 35, reshape=False, cval = 255)
        img_M_40 = ndimage.rotate(a, 360 -35, reshape=False, cval = 255)
        img_M_45 = ndimage.rotate(a, 360 - 45, reshape=False, cval = 255)
        newImages.append(x_train[i])
        newImages.append(img_5)
        newImages.append(img_10)
        newImages.append(img_15)
        newImages.append(img_20)
        newImages.append(img_25)
        newImages.append(img_30)
        newImages.append(img_35)
        newImages.append(img_40)
        newImages.append(img_45)
        newImages.append(img_M_5)
        newImages.append(img_M_10)
        newImages.append(img_M_15)
        newImages.append(img_M_20)
        newImages.append(img_M_25)
        newImages.append(img_M_30)
        newImages.append(img_M_35)
        newImages.append(img_M_40)
        newImages.append(img_M_45)
        for j in range(0,18):
            newLabels.append(y[i])
        #print(i)
    return np.array(newImages) , newLabels

'''Remove bad images where we have more black pixels than white'''
def removeBadImages(x, y):
    cleaned_x = []
    cleaned_y = []
    for i in range(0, x.shape[0]):
        white = cv2.countNonZero(x[i])
        black = 54*54 - white
        if black < white:
            cleaned_x.append(x[i])
            cleaned_y.append(y[i])
    return np.array(cleaned_x), np.array(cleaned_y)

''' Get 1000 images of each data label from MNIST dataset'''
def getMNISTData(numOfImagesofEachLabel = 1000) : 
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = tf.keras.datasets.mnist.load_data()
    x_train_mnist_reduced = []
    y_train_mnist_reduced = []
    ''' Only get 1000 images of each label from MNIST Dataset'''
    for i in range(0, 10):
        count = 0
        for j in range(0, len(y_train_mnist)):
            if(y_train_mnist[j] == i):
                x_train_mnist_reduced.append(x_train_mnist[j])
                y_train_mnist_reduced.append(y_train_mnist[j])
                count+=1
            if count == numOfImagesofEachLabel:
                break
    return np.array(x_train_mnist_reduced) , np.array(y_train_mnist_reduced) 


def preProcessData(file_X, file_Y):
    ############################################
    # MNIST Data pre processing
    ############################################
    '''Load MNIST Data. There are 60,000 images present, all of size 28X28'''
    x_train_mnist, y_train_mnist = getMNISTData()

    ''' Clean the image by performing the following steps
    1. Convert the image to black and white 
    2. Pad the image to convert from 28x28 to 54x54
    ''' 
    x_train_mnist = padImage_MNIST(convertToBlackAndWhite(x_train_mnist, True))

    '''Uncomment the below code to display the preprocessed images'''
    # image_index = 4567# You may select anything up to 60,000
    # print(y_train[image_index]) # The label is 8
    # print(x_train[image_index])
    # plt.imshow(x_train[image_index], cmap='gray')
    # plt.show()

    '''Reshape the data from 54x54 to 54x54x1 and normalize it'''
    x_train = reshapeAndNormalize(x_train_mnist)
    y_train = y_train_mnist

    print('Number of images in x_train MNIST', x_train.shape[0])
    ############################################
    # Data pre processing of the data provided by professor
    ############################################
    # '''Load Dataset provided by the professor'''
    x_train_class  = np.load(file_X)
    y_train_class  = np.load(file_Y)

    '''Resize the data by cropping 45 pixels from all the sides since it doesn't have any data'''
    x_train_class = resizeClassData(x_train_class)
    '''Add rotated images in range -45 degrees to 45 degrees with step of 5 degrees i.e total 18 images'''
    x_train_class , y_train_class = addRotatedImages(x_train_class, y_train_class)
    '''Convert all the black and white'''
    x_train_class = convertToBlackAndWhite(x_train_class)

    '''Now remove the bad images which has more black pixels than white'''
    x_train_class , y_train_class =removeBadImages(x_train_class, y_train_class)

    '''Uncomment the below code to display the preprocessed images'''
    # image_index = 10# You may select anything up to 12,000
    # plt.imshow(x_train_class[image_index], cmap='gray')
    # plt.show()

    '''Reshape the data from 54x54 to 54x54x1 and normalize it'''
    x_train_class = reshapeAndNormalize(x_train_class)


    print('Number of images in x_train Class', x_train_class.shape[0])

    ############################################
    # Append the two datasets from MNIST(10,000 images) 
    # and data provided by professor (21,224 images after cleaning and rotation)
    ############################################

    x_train = np.vstack((x_train, x_train_class))
    print('Training data shape : ',x_train.shape)
    y_train = np.concatenate((y_train_mnist,  y_train_class), axis = 0)
    print('Training labels shape : ', y_train.shape)
    return x_train, y_train




def trainModel(x_train, y_train):
    ############################################
    # Begin model training
    ############################################
    input_shape = (54,54,1)
    model2 = Sequential()

    model2.add(Conv2D(64, kernel_size=(3,3),input_shape=input_shape,activation='relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Dropout(0.2))

    model2.add(Conv2D(32, kernel_size=(5,5),activation='relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Dropout(0.2))

    model2.add(Flatten()) # Flattening the 2D arrays for fully connected layers

    model2.add(Dense(200, activation='relu'))
    model2.add(Dense(128, activation='relu'))
    model2.add(Dense(84, activation='relu'))
    model2.add(Dense(10, activation='softmax'))

    model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    ############################################
    # Fit the model
    ############################################

    #Split appended data into train and test data
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.15, random_state = 0)
    model2.fit(x=x_train,y=y_train, epochs=15, validation_split=0.2)
    model2.evaluate(x_test, y_test)
    return model2

def saveModelToDisk(model):
    ############################################
    # Save the model to disk
    ############################################

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5


    model.save_weights("model.h5")
    model.save('model.h5')
    print("Saved model to disk")

''' 
This is the starting point of the program
It will load x_train and y_train and load the model and save it to disk.
'''
if __name__ == "__main__":
    x_train = []
    y_train = []
    print('\n')
    if len(sys.argv) < 3 :
        print(' You have not defined both the data and label files in the system arguments. The following steps will be performed :\n')
        print(' 1. Load the data from MNIST and file provied by the professor.\n')
        print(' 2. Create a model and save the model to file named "model.json" with its weights in "model.h5" file"\n')
        x_train, y_train = preProcessData('X_train.npy', 'y_train.npy')
    else :
        print(' You have defined both the data and label files in the system arguments. The following steps will be performed :\n')
        print(' 1. Load the image data from first file name and its labels from second file name. THIS WILL NOT PREPROCESS DATA.\n')
        print(' 2. Create a model and save the model to file named "model.json" with its weights in "model.h5" file"\n')
        x_train = np.load(sys.argv[1] + '')
        y_train = np.load(sys.argv[2] + '')   
    model = trainModel(x_train, y_train)
    saveModelToDisk(model)


