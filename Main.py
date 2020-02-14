import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import math
import datetime
import glob
import random
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

train_iters=train_d_gen.flow_from_directory(
    train_path,
    target_size=(img_wedth,img_vertical),
    class_mode='sparse',
    batch_size=50
)

test_iters=test_d_gen.flow_from_directory(
    test_path,
    target_size=(img_wedth, img_vertical),
    class_mode='sparse',
    batch_size=50
)

model.add(Conv2D(
    256,
    (5,5),
    strides=(1,1),
    padding='same',
    activation='relu',
    input_shape=(112,112,3)
))

model.add(Flatten())

model.add(Conv2D(
    192,
    (3,3),
    strides=(1,1),
    padding='same',
    activation='relu'
))

model.add(Dense(
    144,
    activation='relu'
))

model.add(Dropout(0.3))

model.add(Conv2D(
    72,
    (3, 3),
    strides=(1, 1),
    padding='same',
    activation='relu'
))

model.add(Conv2D(
    36,
    (3,3),
    strides=(1,1),
    padding='same',
    activation='relu'
))

model.add(Dense(18,activation='softmax'))
#モデル保存
train_id = datetime.datetime.Now().strftime("%Y/%m/%d-%H:%M:%S")
log_dir=os.path.join("logs",train_id)

ckpt_dir=os.path.join("ckpt",train_id)
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
