import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

with open('aggregated_feature_lists/9K_aggregated_date_feature.pkl', 'rb') as f:
    loaded_aggregated_list = pickle.load(f)

with open('aggregated_feature_lists/9K_aggregated_date_feature_2024_9_29.pkl', 'rb') as f:
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
        combined_arrays.append([date_str[1],price_diff])
        image_arrays.append(date_str[1])
        price_fluctuations.append(price_diff)


    return combined_arrays, np.array(image_arrays), np.array(price_fluctuations)


stock_name = 'COMB'
combined_array, image_arrays, price_fluctuations = process_csv_and_date_list("processed_stock_files/"+stock_name+"2_Processed.csv", loaded_aggregated_list, stock_name)


# model = build_lstm_model(image_arrays[0].shape)  # Extract input shape from data
#
# # Train the model
# early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# model.fit(image_arrays, price_fluctuations, epochs=100, batch_size=3, validation_split=0.2, callbacks=[early_stopping])
#
# # Save the trained model
# model.save('aggregated_model.h5')

price_list = []
for item in combined_array:
    for test_item in loaded_test_aggregated_list:
        vector1= test_item[1].reshape(-1)
        vector2 = item[0].reshape(-1)
        cosine_similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        euclidean_similarity = np.linalg.norm(vector1 - vector2)
        # distance = cosine_similarity(item[0],test_item[1])[0][0]
        # print("Cosine Similarity: ",cosine_similarity)
        if(cosine_similarity > .3 ):
            print("Euclidean Similarity: ",euclidean_similarity)
            print("Price Fluctuations: ",item[1])
            price_list.append(item[1])

# Calculate summary statistics
price_list = np.array(price_list)
mean_change = price_list.mean()
std_dev = price_list.std()

# Visualize the distribution
plt.hist(price_list, bins=20)
plt.xlabel('Price Difference')
plt.ylabel('Frequency')
plt.title('Distribution of Price Changes')
plt.show()

# Define risk tolerance (adjust this based on your preference)
risk_tolerance = 'moderate'  # Options: 'conservative', 'moderate', 'aggressive'


# Based on your risk tolerance and analysis, select a suitable range

price_range = (mean_change - 0.5 * std_dev, mean_change + 0.5 * std_dev)
print("conservative price range: ", price_range)
# print("conservative price mean: ", mean_change)

price_range = (mean_change - std_dev, mean_change + std_dev)
print("moderate price range: ", price_range)

price_range = (mean_change - 2 * std_dev, mean_change + 2 * std_dev)
print("aggressive price range: ", price_range)

