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

print(df.info)

print(df["Usage"].value_counts())

print(df.head())