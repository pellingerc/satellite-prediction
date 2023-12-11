import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def plot_features(tle_dataframe):
    tle_dataframe['DateTime'] = pd.to_datetime(tle_dataframe['Epoch Time'], unit='s')

    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(5, 5))
    fig.suptitle('Variables vs Time', fontsize=16)

    # Plot each variable against time
    axs[0, 0].plot(tle_dataframe['DateTime'], tle_dataframe['Ballistic Coefficient'])
    axs[0, 0].set_title('Ballistic Coefficient')

    axs[0, 1].plot(tle_dataframe['DateTime'], tle_dataframe['Second Derivative Of MeanMotion'])
    axs[0, 1].set_title('Second Derivative Of MeanMotion')

    axs[0, 2].plot(tle_dataframe['DateTime'], tle_dataframe['Drag Term'])
    axs[0, 2].set_title('Drag Term')

    axs[1, 0].plot(tle_dataframe['DateTime'], tle_dataframe['Inclination'])
    axs[1, 0].set_title('Inclination')

    axs[1, 1].plot(tle_dataframe['DateTime'], tle_dataframe['Right Ascension Of Ascending Node'])
    axs[1, 1].set_title('Right Ascension Of Ascending Node')

    axs[1, 2].plot(tle_dataframe['DateTime'], tle_dataframe['Eccentricity'])
    axs[1, 2].set_title('Eccentricity')

    axs[2, 0].plot(tle_dataframe['DateTime'], tle_dataframe['ArgumentOfPerigee'])
    axs[2, 0].set_title('ArgumentOfPerigee')

    axs[2, 1].plot(tle_dataframe['DateTime'], tle_dataframe['Mean Anomaly'])
    axs[2, 1].set_title('Mean Anomaly')

    axs[2, 2].plot(tle_dataframe['DateTime'], tle_dataframe['Mean Motion'])
    axs[2, 2].set_title('Mean Motion')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_timeaccuracy(df, features, steps, y_test, y_pred):
    y_test = y_test.reshape(-1, steps, features)
    y_pred = y_pred.reshape(-1, steps, features)
    columns = df.columns[11:17]
    epoch_diff = np.zeros((y_test.shape[0], y_test.shape[1]))
    for i in range(epoch_diff.shape[0]):
        for j in range(epoch_diff.shape[1]):
            epoch_diff[i, j] = df['Epoch Time'][i + j] - df['Epoch Time'][i]

    epoch_diff = epoch_diff.reshape(-1)
    error = np.abs(y_test - y_pred)
    hours = epoch_diff // 3600  # 3600 seconds in an hour
    


    for i in range(features):
        error_feature = error[:, :, i].reshape(-1)
        error_df = pd.DataFrame({'Hours': hours, 'Error': error_feature})
        mean_error_by_hour = error_df.groupby('Hours')['Error'].mean().reset_index()
        
        # Only less than 1200 hours
        mean_error_by_hour = mean_error_by_hour[mean_error_by_hour['Hours'] < 1200]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.scatter(mean_error_by_hour['Hours'], mean_error_by_hour['Error'], alpha=0.5)
        ax.set_xlabel('Hour of Epoch Time Difference')
        ax.set_ylabel(f'Mean Absolute Error - {columns[i]}')
        ax.set_title(f'Mean Absolute Error over 1200 hours - {columns[i]}')

        
        plt.tight_layout()
        plt.show()

        input("Press Enter to continue to the next feature...")


def plot_mse(y_test, y_pred, tle, features):
    for i in range(features):
        fig, ax = plt.subplots(figsize=(15, 4))
        ax.plot(y_test[:, i], label=f'Actual Feature {tle.columns[i + 11]} Value')
        ax.plot(y_pred[:, i], label=f'Predicted Feature {tle.columns[i + 11]}')
        ax.set_title(f'Actual vs Predicted Feature {i + 1}')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel(f'{tle.columns[i]} Value')
        ax.legend()
        ax.yaxis.set_major_formatter(plt.ticker.ScalarFormatter(useMathText=False))
        plt.show()

        input("Press Enter to show the next plot...")


    plt.close()


def calc_mses(y_test, y_pred):
    y_test = y_test.reshape(-1, 100, 6)
    y_pred = y_pred.reshape(-1, 100, 6)

    # Calculate MSEs for each feature and print
    for i in range(6):
        y_test_feature = y_test[:, :, i].reshape(-1, 100)
        y_pred_feature = y_pred[:, :, i].reshape(-1, 100)
        mse = mean_squared_error(y_test_feature, y_pred_feature)
        print(f'MSE for Feature {i + 1}: {mse}')





 


     
        
