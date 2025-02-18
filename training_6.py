import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


with open('aggregated_feature_lists/aggregated_date_feature.pkl', 'rb') as f:
    loaded_aggregated_list = pickle.load(f)

with open('aggregated_feature_lists/aggregated_date_feature_2024_9_29.pkl', 'rb') as f:
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


stock_name = 'COMB'
combined_array, image_arrays, price_fluctuations = process_csv_and_date_list(
    "processed_stock_files/" + stock_name + "2_Processed.csv", loaded_aggregated_list, stock_name)



# Assuming you have your data in NumPy arrays
X = image_arrays  # (num_samples, 512)
y = price_fluctuations  # (num_samples,)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

# Create a Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=12)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")


# Save the model to a file
# with open('rf_model.joblib', 'wb') as f:  # Open the file in write binary mode ('wb')
#     pickle.dump(rf_model, f)

# Load the saved model
# with open('rf_model.joblib', 'rb') as f:  # Open the file in read binary mode ('rb')
#     loaded_model = pickle.load(f)



print(rf_model.predict(loaded_test_aggregated_list[0][1]))
# Feature importance
# feature_importances = rf_model.feature_importances_
# print(f"Feature importances: {feature_importances}")
#

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=12)
#
# # Create a Decision Tree Regressor
# model = DecisionTreeRegressor(random_state=12)
#
# # Train the model
# model.fit(X_train, y_train)
#
# # Make predictions
# y_pred = model.predict(X_test)
#
# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# print(f"Mean Squared Error: {mse}")