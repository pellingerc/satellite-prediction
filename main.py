import pandas as pd
from datetime import datetime, timedelta
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

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
        tle.append(time_difference.total_seconds()) # Epoch Time since Jan 1 1970
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


file_path = 'data/sbudnic.txt'
tle_dataframe, tle_array = read_tle_file(file_path)

#Feature Engineering

#Create Lag Features 
def create_lag_features(tle_dataframe):
    for i in range(1, 11):
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
    return tle_dataframe

tle_dataframe = create_lag_features(tle_dataframe)

#Drop Nulls (First 10 rows)
tle_dataframe = tle_dataframe.dropna()

#Train Test Split
def train_test_split(tle_dataframe):
    train_size = int(len(tle_dataframe) * 0.8)
    train, test = tle_dataframe[:train_size], tle_dataframe[train_size:]
    return train, test

train, test = train_test_split(tle_dataframe)   

#Create X and y
def create_X_y(train, test):
    X_train = train.drop(['Epoch Time', 'Ballistic Coefficient', 'Second Derivative Of MeanMotion', 'Drag Term', 'Inclination', 'Right Ascension Of Ascending Node', 'Eccentricity', 'ArgumentOfPerigee', 'Mean Anomaly', 'Mean Motion', 'Revolution Number At Epoch'], axis=1)
    y_train = train[['Epoch Time', 'Ballistic Coefficient', 'Second Derivative Of MeanMotion', 'Drag Term', 'Inclination', 'Right Ascension Of Ascending Node', 'Eccentricity', 'ArgumentOfPerigee', 'Mean Anomaly', 'Mean Motion', 'Revolution Number At Epoch']]

    X_test = test.drop(['Epoch Time', 'Ballistic Coefficient', 'Second Derivative Of MeanMotion', 'Drag Term', 'Inclination', 'Right Ascension Of Ascending Node', 'Eccentricity', 'ArgumentOfPerigee', 'Mean Anomaly', 'Mean Motion', 'Revolution Number At Epoch'], axis=1)
    y_test = test[['Epoch Time', 'Ballistic Coefficient', 'Second Derivative Of MeanMotion', 'Drag Term', 'Inclination', 'Right Ascension Of Ascending Node', 'Eccentricity', 'ArgumentOfPerigee', 'Mean Anomaly', 'Mean Motion', 'Revolution Number At Epoch']]

    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = create_X_y(train, test)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(X_train_scaled.shape)
#Train Linear Model
model = linear_model.LinearRegression()
model.fit(X_train_scaled, y_train)

# Print coefficients
print("Intercept:", model.intercept_)
print("Coefficients:")
for feature, coef in zip(X_train.columns, model.coef_):
    print(f"{feature}: {coef}")
#Predict
predictions = model.predict(X_test_scaled)

#Evaluate
score = model.score(X_test_scaled, y_test)
print(score)

# Create a DataFrame for predictions and actual values
predictions_df = pd.DataFrame(predictions, columns=y_test.columns)

# Plotting
fig, axs = plt.subplots(nrows=2, ncols=6, figsize=(18, 8), sharex=True)

for i, col in enumerate(y_test.columns):
    row_num = i // 6
    col_num = i % 6
    axs[row_num, col_num].scatter(y_test[col], predictions_df[col], label='Predictions')
    axs[row_num, col_num].plot([y_test[col].min(), y_test[col].max()], [y_test[col].min(), y_test[col].max()], 'k--', lw=2, label='Ideal')
    axs[row_num, col_num].set_title(col)
    axs[row_num, col_num].set_xlabel('Actual')
    axs[row_num, col_num].set_ylabel('Predicted')

plt.tight_layout()
plt.show()





# Display the DataFrame

