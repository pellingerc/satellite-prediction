import pandas as pd
from datetime import datetime, timedelta
from sgp4.api import Satrec
from sgp4.earth_gravity import wgs72
from sgp4.io import twoline2rv
import numpy as np


def read_tle_file(file_path):
    '''
    Input: file_path - path to the TLE file
    Output: tle_dataframe - Pandas DataFrame containing the TLE data
    
    '''
    with open(file_path, 'r') as file:
        lines = file.readlines()

    tle_data = []
    for i in range(0, len(lines), 2):
        tle = []
        line1 = lines[i].strip()
        line2 = lines[i + 1].strip()
        

        # tle_data['SatelliteNumber'].append(line1[2:7])
        # tle_data['Classification'].append(line1[7])
        # tle_data['InternationalDesignator'].append(line1[9:17])
        year = int(line1[18:20]) + 2000 # 2 digit year
        day = float(line1[20:32])# Fractional Julian Day of year
        date_object = datetime(year, 1, 1) + timedelta(days=day - 1)
        time_difference = date_object - datetime(1970, 1, 1)
        tle.append(time_difference.total_seconds()) # Epoch Time since Jan 1 1970 in seconds
        tle.append(float(line1[33:43])) # Ballistic Coefficient
        tle.append(float("0." + line1[45:50]) * (10 ** float(line1[50:52]))) # 2nd derivative of mean motion
        tle.append(float("0." + line1[54:59]) * (10 ** float(line1[59:61]))) # Drag Term




        tle.append(float(line2[8:16])) # Inclination
        tle.append(float(line2[17:25])) # Right Ascension of Ascending Node
        tle.append(float("0." + line2[26:33])) # Eccentricity
        tle.append(float(line2[34:42])) # Argument of Perigee
        tle.append(float(line2[43:51])) # Mean Anomaly
        tle.append(float(line2[52:63])) # Mean Motion
        tle.append(float(line2[63:68])) # Revolution Number at Epoch


        
        
    # Convert to Julian date
        
        assert twoline2rv(lines[i], lines[i + 1], wgs72)
        satellite = Satrec.twoline2rv(lines[i], lines[i + 1])
        bit, position, velocity = satellite.sgp4(satellite.jdsatepoch, satellite.jdsatepochF)
        tle.append(position[0])
        tle.append(position[1])
        tle.append(position[2])

        tle.append(velocity[0])  
        tle.append(velocity[1])  
        tle.append(velocity[2]) 



        tle_data.append(tle)
    df = pd.DataFrame(tle_data, columns=['Epoch Time', 'Ballistic Coefficient', 'Second Derivative Of MeanMotion', 'Drag Term', 'Inclination', 'Right Ascension Of Ascending Node', 'Eccentricity', 'ArgumentOfPerigee', 'Mean Anomaly', 'Mean Motion', 'Revolution Number At Epoch', 'Position X', 'Position Y', 'Position Z', 'Velocity X', 'Velocity Y', 'Velocity Z'])
    return df, tle_data


#Train Test Split
def timesereis_train_test_split(tle_dataframe):
    print(len(tle_dataframe))
    train_size = int(len(tle_dataframe) * 0.7)
    validation_size = int(len(tle_dataframe) * 0.15)

    train, validation, test = tle_dataframe[:train_size], tle_dataframe[train_size:train_size + validation_size], tle_dataframe[train_size + validation_size:]
    return train, validation, test

#Create X and y
def create_X_y(train, test):
    X_train = train.drop(['Ballistic Coefficient', 'Second Derivative Of MeanMotion', 'Drag Term', 'Inclination', 'Right Ascension Of Ascending Node', 'Eccentricity', 'ArgumentOfPerigee', 'Mean Anomaly', 'Mean Motion', 'Revolution Number At Epoch'], axis=1)
    #y_train = train[[ 'Ballistic Coefficient', 'Second Derivative Of MeanMotion', 'Drag Term', 'Inclination', 'Right Ascension Of Ascending Node', 'Eccentricity', 'ArgumentOfPerigee', 'Mean Anomaly', 'Mean Motion', 'Revolution Number At Epoch']]
    y_train = train[['Mean Anomaly']]


    X_test = test.drop(['Ballistic Coefficient', 'Second Derivative Of MeanMotion', 'Drag Term', 'Inclination', 'Right Ascension Of Ascending Node', 'Eccentricity', 'ArgumentOfPerigee', 'Mean Anomaly', 'Mean Motion', 'Revolution Number At Epoch'], axis=1)
    #y_test = test[['Ballistic Coefficient', 'Second Derivative Of MeanMotion', 'Drag Term', 'Inclination', 'Right Ascension Of Ascending Node', 'Eccentricity', 'ArgumentOfPerigee', 'Mean Anomaly', 'Mean Motion', 'Revolution Number At Epoch']]
    y_test = test[['Mean Anomaly']]

    return X_train, y_train, X_test, y_test

def create_sequences(data, data_target, sequence_length, target_steps):
    '''
    Create sequences to predict multiple target_steps in data
    ''' 
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length - target_steps + 1):
        seq = data[i:i+sequence_length]
        target = data_target[i+sequence_length:i+sequence_length+target_steps]
        target = target.reshape(-1)
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

def standardize_interval(df):
    df['Epoch Time'] = pd.to_datetime(df['Epoch Time'], unit='s')
    df.set_index('Epoch Time', inplace=True)
    df_resampled = df.resample('12H').mean()
    df_interpolated = df_resampled.interpolate(method='linear')
    df_interpolated.reset_index(inplace=True)
    df_interpolated['Epoch Time'] = (df_interpolated['Epoch Time'] - pd.Timestamp("1970-01-01")).dt.total_seconds()


    return df_interpolated