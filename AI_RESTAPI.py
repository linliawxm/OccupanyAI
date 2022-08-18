#!/usr/bin/env python
# coding: utf-8

# -*- coding: utf-8 -*-
import os
import sys
import urllib.request
import time

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, RepeatVector, TimeDistributed
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.spatial import distance_matrix
from keras.models import load_model
import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback
import requests

# df = pd.read_csv("drive/My Drive/ColabDrive/adjacent_3_room_occupancy.csv", index_col=False)
df = pd.read_csv("adjacent_3_room_occupancy.csv", index_col=False)
#print(f'Raw Data Shape{df.shape}')
df.index = pd.to_datetime(df['capture_at'], format='%Y-%m-%d %H:%M:%S')
df.drop('capture_at', axis=1, inplace=True)
#print(f'New Data Shape{df.shape}')
df = df.astype('float')
df['output'] = df.max(axis=1)

def data_to_X_y(data, windows_size, output_size):
    X, y = [],[]

    for i in range(windows_size, len(data)-windows_size):
        X.append(data[i-windows_size:i].iloc[:,:3])
        y.append(data[i:i+output_size].iloc[:,-1])
    return np.array(X), np.array(y)

input_size = 288
output_size = 12

X, y = data_to_X_y(df, input_size, output_size)
print(X.shape)
print(y.shape)


### Train, Validation, Test Data Split
train_size = 16128  # 8 weeks
val_size = 14112  # 7 weeks

X_train, y_train = X[:val_size], y[:val_size]
X_val, y_val = X[val_size: train_size], y[val_size:train_size]
X_test, y_test = X[train_size:], y[train_size:]
print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)

#Create JSON Object
data = json.dumps({'signature_name': 'serving_default', 'instances': X_test[100:101].tolist()})  #288 x 3




headers = {"content-type": "application/json"}
json_response = requests.post('http://localhost:8501/v1/models/lstm_multistep:predict', data=data, headers=headers)
predictions = json.loads(json_response.text)['predictions']


pred = np.array(predictions[0]).round().reshape(1,12)



#pred=[ np.argmax(predictions[p]) for p in range(len(predictions)) ]
print("Predictions: ",pred)
print("Actual:      ",y_test[100:101])