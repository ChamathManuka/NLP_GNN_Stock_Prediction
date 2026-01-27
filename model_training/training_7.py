import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

with open('../aggregated_feature_lists/aggregated_date_feature.pkl', 'rb') as f:
    loaded_aggregated_list = pickle.load(f)

with open('../aggregated_feature_lists/aggregated_date_feature_2024_9_29.pkl', 'rb') as f:
    loaded_test_aggregated_list = pickle.load(f)


def process_csv_and_date_list(csv_file, date_list, stock_name):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    df['Date'] = pd.to_datetime(df['Trade Date'])
    # Convert the 'Date' column to datetime format
    image_arrays = []
    price_fluctuations = []
    combined_arrays = []
    # Iterate through the date list and find corresponding price differences
    for date_str in date_list:
        predict_date = date_str[0]
        date = pd.to_datetime(predict_date)
        matching_row = df[df['Date'] == date]

        if not matching_row.empty:
            price_diff = matching_row['Price Difference'].values[0]
        else:
            price_diff = 0
        combined_arrays.append([date_str[1], price_diff])
        image_arrays.append(date_str[1].reshape(-1))
        price_fluctuations.append(price_diff)

    return combined_arrays, np.array(image_arrays), np.array(price_fluctuations)


stock_name = 'HNB'
combined_array, image_arrays, price_fluctuations = process_csv_and_date_list(
    "processed_stock_files/" + stock_name + "2_Processed.csv", loaded_aggregated_list, stock_name)

# Assuming you have your data in NumPy arrays
X = image_arrays  # (num_samples, 512)
y = price_fluctuations  # (num_samples,)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an SVR model (using RBF kernel as an example)
svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.2)  # Adjust C and epsilon as needed

# Train the model
svr_model.fit(X_train, y_train)

# Make predictions
y_pred = svr_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
