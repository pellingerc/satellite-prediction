import pandas as pd
from datetime import datetime, timedelta


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

        tle_data.append(tle)
    df = pd.DataFrame(tle_data, columns=['Epoch Time', 'Ballistic Coefficient', 'Second Derivative Of MeanMotion', 'Drag Term', 'Inclination', 'Right Ascension Of Ascending Node', 'Eccentricity', 'ArgumentOfPerigee', 'Mean Anomaly', 'Mean Motion', 'Revolution Number At Epoch'])
    return df, tle_data


#Train Test Split
def timesereis_train_test_split(tle_dataframe):
    train_size = int(len(tle_dataframe) * 0.8)
    train, test = tle_dataframe[:train_size], tle_dataframe[train_size:]
    return train, test

#Create X and y
def create_X_y(train, test):
    X_train = train.drop(['Ballistic Coefficient', 'Second Derivative Of MeanMotion', 'Drag Term', 'Inclination', 'Right Ascension Of Ascending Node', 'Eccentricity', 'ArgumentOfPerigee', 'Mean Anomaly', 'Mean Motion', 'Revolution Number At Epoch'], axis=1)
    #y_train = train[[ 'Ballistic Coefficient', 'Second Derivative Of MeanMotion', 'Drag Term', 'Inclination', 'Right Ascension Of Ascending Node', 'Eccentricity', 'ArgumentOfPerigee', 'Mean Anomaly', 'Mean Motion', 'Revolution Number At Epoch']]
    y_train = train[['Mean Anomaly']]


    X_test = test.drop(['Ballistic Coefficient', 'Second Derivative Of MeanMotion', 'Drag Term', 'Inclination', 'Right Ascension Of Ascending Node', 'Eccentricity', 'ArgumentOfPerigee', 'Mean Anomaly', 'Mean Motion', 'Revolution Number At Epoch'], axis=1)
    #y_test = test[['Ballistic Coefficient', 'Second Derivative Of MeanMotion', 'Drag Term', 'Inclination', 'Right Ascension Of Ascending Node', 'Eccentricity', 'ArgumentOfPerigee', 'Mean Anomaly', 'Mean Motion', 'Revolution Number At Epoch']]
    y_test = test[['Mean Anomaly']]

    return X_train, y_train, X_test, y_test
