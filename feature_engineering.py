#Create Lag Features 
def create_lag_features(tle_dataframe):
    for i in range(1, 20):
        tle_dataframe[f'Epoch Time Lag {i}'] = tle_dataframe['Epoch Time'].shift(i)
        tle_dataframe[f'Ballistic Coefficient Lag {i}'] = tle_dataframe['Ballistic Coefficient'].shift(i)
        tle_dataframe[f'Second Derivative Of MeanMotion Lag {i}'] = tle_dataframe['Second Derivative Of MeanMotion'].shift(i)
        tle_dataframe[f'Drag Term Lag {i}'] = tle_dataframe['Drag Term'].shift(i)
        tle_dataframe[f'Inclination Lag {i}'] = tle_dataframe['Inclination'].shift(i)
        tle_dataframe[f'Right Ascension Of Ascending Node Lag {i}'] = tle_dataframe['Right Ascension Of Ascending Node'].shift(i)
        tle_dataframe[f'Eccentricity Lag {i}'] = tle_dataframe['Eccentricity'].shift(i)
        tle_dataframe[f'ArgumentOfPerigee Lag {i}'] = tle_dataframe['ArgumentOfPerigee'].shift(i)
        tle_dataframe[f'Mean Anomaly Lag {i}'] = tle_dataframe['Mean Anomaly'].shift(i)
        tle_dataframe[f'Mean Motion Lag {i}'] = tle_dataframe['Mean Motion'].shift(i)
        tle_dataframe[f'Revolution Number At Epoch Lag {i}'] = tle_dataframe['Revolution Number At Epoch'].shift(i)
    tle_dataframe = tle_dataframe.dropna()
    return tle_dataframe

def remove_outliers(tle_dataframe, threshold):
    tle_dataframe_filtered = tle_dataframe[tle_dataframe['Drag Term'] < threshold]
    tle_dataframe_filtered = tle_dataframe_filtered.reset_index(drop=True)
    return tle_dataframe_filtered