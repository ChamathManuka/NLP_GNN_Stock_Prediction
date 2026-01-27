import pickle

import numpy as np
import pandas as pd
from keras.src.optimizers import Adam
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping

with open('../aggregated_feature_lists/aggregated_date_feature.pkl', 'rb') as f:
    loaded_aggregated_list = pickle.load(f)
    print(loaded_aggregated_list)


def process_csv_and_date_list(csv_file, date_list, stock_name):
    """
    Processes a CSV file containing dates and price differences, and a given date list.

    Args:
        csv_file (str): Path to the CSV file.
        date_list (list): List of dates to check.

    Returns:
        pandas.DataFrame: A DataFrame with dates and corresponding price differences.
    """

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Convert the 'Date' column to datetime format
    # df['Date'] = pd.to_datetime(df['Trade Date'], format='%m/%d/%y')
    df['Date'] = pd.to_datetime(df['Trade Date'])

    image_array_list = []
    price_list = []
    # Iterate through the date list and find corresponding price differences
    for date_str in date_list:

        date = pd.to_datetime(date_str[0])
        date_only = date.date().strftime('%Y-%m-%d')
        matching_row = df[df['Date'] == date_only]

        if not matching_row.empty:
            price_diff = matching_row['Price Difference'].values[0]
        else:
            price_diff = 0

        image_array_list.append(date_str[1])
        price_list.append(price_diff)
        print(date, price_diff)

    return np.array(image_array_list), np.array(price_list)


stock_name = 'JKH'
image_arrays, price_fluctuations = process_csv_and_date_list("processed_stock_files/" + stock_name + "2_Processed.csv",
                                                             concatenated_list, stock_name)

# Save the list of (date, image) pairs
with open('../training_files/image_array4.pkl', 'wb') as h:
    pickle.dump(image_arrays, h)
with open('../training_files/price_fluctuations4.pkl', 'wb') as o:
    pickle.dump(price_fluctuations, o)


def build_lstm_model(input_shape):
    """
    Builds an LSTM model for price fluctuation prediction based on image arrays.

    Args:
        input_shape: Shape of the input image arrays (e.g., (1, 12288)).

    Returns:
        A compiled LSTM model.
    """
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    optimizer = Adam(learning_rate=0.001)  # Adjust learning rate
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model


# Assuming you have loaded your image arrays and price fluctuations
# (replace with your actual data loading logic)

# Example:
# image_arrays = ... # Your loaded image arrays (shape: (num_samples, 1, 12288))
# price_fluctuations = ... # Your loaded price fluctuations

# Build the LSTM model
model = build_lstm_model(image_arrays[0].shape)  # Extract input shape from data

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(image_arrays, price_fluctuations, epochs=10, batch_size=5, validation_split=0.2, callbacks=[early_stopping])

# Save the trained model
model.save('stock_prediction_image_model4.h5')
#
# # Load and use the saved model for prediction (example)
# loaded_model = tf.keras.models.load_model('stock_prediction_image_model.h5')
# new_image_array = image_arrays[0:1]  # Take the first image array as an example
# predicted_fluctuation = loaded_model.predict(new_image_array)
# print("Predicted fluctuation:", predicted_fluctuation)
