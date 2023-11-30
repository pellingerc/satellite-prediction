import pandas as pd
from datetime import datetime, timedelta
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from preprocessing import read_tle_file, train_test_split, create_X_y
from feature_engineering import create_lag_features
import tensorflow as tf





file_path = 'data/beesat-1.txt'
tle_dataframe, tle_array = read_tle_file(file_path)

tle_dataframe = (tle_dataframe.iloc[10:]).reset_index(drop=True) #drop first 10 rows


tle_dataframe = create_lag_features(tle_dataframe).copy()


train, test = train_test_split(tle_dataframe)   


X_train, y_train, X_test, y_test = create_X_y(train, test)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(X_train_scaled.shape)


# Train Gradient Boosted Trees model
gb_model = GradientBoostingRegressor(n_estimators=50, random_state=42)  # You can adjust the parameters as needed
gb_model.fit(X_train_scaled, y_train)

# Make predictions
predictions = gb_model.predict(X_test_scaled)

# Evaluate
score = gb_model.score(X_test_scaled, y_test)
print(f'R2 Score: {score}')

# Create a DataFrame for predictions and actual values
predictions_df = pd.DataFrame(predictions, columns=y_test.columns)

# Plotting
fig, axs = plt.subplots(nrows=2, ncols=6, figsize=(18, 8), sharex=False, sharey=False)

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

