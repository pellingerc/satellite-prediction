import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_features(tle_dataframe):
    tle_dataframe['DateTime'] = pd.to_datetime(tle_dataframe['Epoch Time'], unit='s')

    # Create subplots
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

    # Adjust layout and show the plots
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_timeaccuracy(df, features, steps, y_test, y_pred):
    y_test = y_test.reshape(-1, steps, features)
    y_pred = y_pred.reshape(-1, steps, features)

    # Calculate epoch time differences
    epoch_diff = np.zeros((y_test.shape[0], y_test.shape[1]))
    for i in range(epoch_diff.shape[0]):
        for j in range(epoch_diff.shape[1]):
            epoch_diff[i, j] = df['Epoch Time'][i + j] - df['Epoch Time'][i]

    # Reshape epoch_diff to 1D array
    epoch_diff = epoch_diff.reshape(-1)

    # Calculate absolute error
    error = np.abs(y_test - y_pred)

    # Convert epoch_diff to hours
    hours = epoch_diff // 3600  # 3600 seconds in an hour
    


    for i in range(features):
        # Flatten error for the current feature
        error_feature = error[:, :, i].reshape(-1)

        # Create a DataFrame with hours and errors for the current feature
        error_df = pd.DataFrame({'Hours': hours, 'Error': error_feature})

        # Group by hours and calculate mean error for each hour
        mean_error_by_hour = error_df.groupby('Hours')['Error'].mean().reset_index()

        # Plot the mean error for the current feature
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.scatter(mean_error_by_hour['Hours'], mean_error_by_hour['Error'], alpha=0.5)
        ax.set_xlabel('Hour of Epoch Time Difference')
        ax.set_ylabel(f'Mean Error - Feature {i+1}')
        ax.set_title(f'Mean Error vs. Hour of Epoch Time Difference - Feature {i+1}')

        plt.tight_layout()
        plt.show()

        # Wait for user input before moving to the next feature
        input("Press Enter to continue to the next feature...")


def plot_mse(y_test, y_pred, tle, features):
    for i in range(features):
        fig, ax = plt.subplots(figsize=(15, 4))
        ax.plot(y_test[:, i], label=f'Actual Feature {tle.columns[i + 11]} Value')
        ax.plot(y_pred[:, i], label=f'Predicted Feature {tle.columns[i + 11]}')
        ax.set_title(f'Actual vs Predicted Feature {i + 1}')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel(f'Feature {i + 1} Value')
        ax.legend()
        plt.show()

        # Wait for user input to proceed to the next plot
        input("Press Enter to show the next plot...")

# Close the last figure
    plt.close()




    # Create subplots



 


     
        
