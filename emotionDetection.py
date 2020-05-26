import numpy as np
import pandas as pd

import sys 
import os

from keras.models import Sequential
from keras.layers import Dense , Dropout , Activation , Flatten
from keras.layers import Conv2D, MaxPooling2D,BatchNormalization
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import np_utils


df = pd.read_csv('Data\\fer2013.csv')

X_train,train_y,X_test,test_y =[],[],[],[]

for index ,row in df.iterrows():
    
    row_data = row['pixels'].split(" ")
    try:
        if 'Training' in row['Usage']:
            X_train.append(np.array(row_data, 'float32'))
            train_y.append(row['emotion'])
        
        elif 'PublicTest' in row['Usage']:
            X_test.append(np.array(row_data,'float32'))
            test_y.append(row['emotion'])
    except:
        print(f"error occured at index :{index} and row:{row}")

X_train = np.array(X_train,'float32')
train_y = np.array(train_y,'float32')
X_test = np.array(X_test,'float32')
test_y = np.array(test_y,'float32')       
        

#normalizing data between oand 1
X_train -= np.mean(X_train, axis=0)
X_train /= np.std(X_train, axis=0)

X_test -= np.mean(X_test, axis=0)
X_test /= np.std(X_test, axis=0)

        
        