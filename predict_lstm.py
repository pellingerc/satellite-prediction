import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from preprocessing import read_tle_file, timesereis_train_test_split, create_X_y
from feature_engineering import create_lag_features
import tensorflow as tf


def create_sequences(data, sequence_length):
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length):
        seq = data[i:i+sequence_length]
        target = data[i+sequence_length]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)


file_path = 'data/beesat-1.txt'
tle_dataframe, tle_array = read_tle_file(file_path) #read in data

tle_dataframe = (tle_dataframe.iloc[10:]).reset_index(drop=True) #drop first 10 rows

#Do feature engineering here if needed (possibly add lag features)

train, test = timesereis_train_test_split(tle_dataframe) #train test split


scaler = StandardScaler() #scale data
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)

sequence_length = 20 #create sequences
X_train, y_train = create_sequences(train, sequence_length)
X_test, y_test = create_sequences(test, sequence_length)

#LSTM Model
model = tf.keras.Sequential() #create model
model.add(tf.keras.layers.LSTM(100, activation='relu', input_shape=(sequence_length, tle_dataframe.shape[1]))) #add layers
model.add(tf.keras.layers.Dense(units=tle_dataframe.shape[1])) #add layers
model.compile(optimizer='adam', loss='mse') #compile model

#Train model
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, validation_split=0.2)
loss = model.evaluate(X_test, y_test, verbose=0)
print(f'Mean Squared Error on Test Set: {loss}')






# Display the DataFrame
