import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import math
import datetime
import glob
import random+
import cv2

import keras
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.preprocessing.image import array_to_img,img_to_array,list_pictures,load_img,ImageDataGenerator
from sklearn.model_selection import train_test_split

img_wedth,img_vertical=224,224
train_path = r'my_folder\train'
test_path = r'my_folder\test'
train_data=[]
train_re=[]
test_data=[]
test_re=[]

for img in list_pictures(train_path):
    data=img_to_array(load_img(img,target_size=(img_wedth,img_vertical)))
    train_data.append(data)
    #train_re.append

for img in list_pictures(test_path):
    data=img_to_array(load_img(img,target_size=(img_wedth, img_vertical)))
    test_data.append(data)

train_data=np.asarray(train_data)
test_data=np.asarray(test_data)

model=Sequential()


train_d_gen = ImageDataGenerator(
    rescale=1/255
)

test_d_gen= ImageDataGenerator(
    rescale=1/255
)

train_gene=train_d_gen.flow_from_directory(
    train_path,
    target_size=(img_wedth,img_vertical),
    batch_size=32,
    class_mode='sparse'
)

test_gene = test_d_gen.flow_from_directory(
    test_path,
    target_size=(img_wedth, img_vertical),
    batch_size=32,
    class_mode='sparse'
)
