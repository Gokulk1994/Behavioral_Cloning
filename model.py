import cv2
import csv
import numpy as np
import os
import sklearn
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split

"""
Read the Data of images and Steering angle from CSV
param: None
return : lines of csv file
"""
def ReadaCSVData():
    lines = []
    with open('data/driving_log.csv') as csvFile:
         reader = csv.reader(csvFile)
         for line in reader:
            lines.append(line)
    return lines

"""
This function returns the path for the captured center,left and right images
param : lines - all lines in csv
return : list of paths of center, left and right images, list of steerig angle
"""
def GetImagePaths(lines):
    center = []
    right  = []
    left   = []
    Angle  = []
    for line in lines[1:]:
        center.append('data/' + line[0].strip())
        left.append('data/' + line[1].strip())
        right.append('data/' + line[2].strip())
        Angle.append(float(line[3]))
    return (center,left,right,Angle)
  
"""
This function merges the individual images from all 3 cameras into single list and 
calculates the steering angle for right and left images also by adding offset of +/- 0.2
param : list of paths of center, left and right images, list of steerig angle
return : list of images and steering angles for all 3 cameras
"""
def MergeImages(center,left,right,Angle):
    Images = []
    Angles = []
    
    Images.extend(center)
    Images.extend(left)
    Images.extend(right)
    
    Angles.extend(Angle)
    Angles.extend([x + 0.2 for x in Angle])
    Angles.extend([x - 0.2 for x in Angle])
    
    return(Images,Angles)

"""
This function generates the Generator for the train and validation data set
param : sample data set
return : yield value of the generator for image and steering angle
"""
def Generator(Data):
    Length = len(Data)
    while 1: 
        Data = sklearn.utils.shuffle(Data)
        for offset in range(0, Length, 32):
            batch_samples = Data[offset:offset+32]

            images = []
            angles = []
            for Pic, angle in batch_samples:
                originalImage = cv2.imread(Pic)
                #convert BGR image to RGB
                image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
                images.append(image)
                angles.append(angle)
                # Flip the images and append the list
                images.append(cv2.flip(image,1))
                angles.append(angle*-1.0)


            inputs = np.array(images)
            outputs = np.array(angles)
            yield sklearn.utils.shuffle(inputs, outputs)    
    
"""
This function executes the convolution neural network the data set
param : none
return : the trained model
"""
def Model():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    
    return model
   
# Reads the CSV    
LinesCSV = ReadaCSVData()

#Get the path and steering angle from csv lines
center,left,right,Angle = GetImagePaths(LinesCSV)

# Get a single list of Images and Angles
Images,Angles  = MergeImages(center,left,right,Angle )

#split the data set into train and validation set
ModelData = list(zip(Images,Angles))
train_data, validation_data = train_test_split(ModelData, test_size=0.2)

# Get the generator fot the train and validation data set
Train_Gen = Generator(train_data)
Valid_Gen = Generator(validation_data)

# Get the trained model
model = Model() 

# compile the model
model.compile(loss='mse', optimizer='adam')

# Fit the model 
history_object = model.fit_generator(Train_Gen, samples_per_epoch= \
                 len(train_data), validation_data=Valid_Gen, \
                 nb_val_samples=len(validation_data), nb_epoch=3, verbose=1)

# save the model           
model.save('model1.h5')

#calculate the loss
print(history_object.history.keys())
print('Loss')
print(history_object.history['loss'])

print('Validation Loss')
print(history_object.history['val_loss'])
