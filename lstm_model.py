import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def build_model(features):
    model = tf.keras.Sequential() #create model
    model.add(tf.keras.layers.LSTM(200, activation='tanh', return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.LSTM(200, activation='tanh', return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(200, activation='tanh')))
    model.add(tf.keras.layers.Dense(units=600)) #add layers
    model.compile(optimizer='adam', loss='mse') #compile model

    return model 