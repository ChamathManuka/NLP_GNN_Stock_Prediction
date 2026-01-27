import pickle
from collections import defaultdict

import numpy as np

# with open('vector_testing_files/date_vector.pkl', 'rb') as f:
with open('../date_feature_test_list_files/9K_2024_09_26.pkl', 'rb') as f:
    test_list = pickle.load(f)

date_list = []
for date in test_list:
    date_list.append(date)


# print(date_list)


def aggregate_daily_news(daily_news_arrays):
    """
    Aggregates daily news articles into a single array.

    Args:
      daily_news_arrays: A list of numpy arrays, where each array
                         represents a single news article (shape: (1, 512)).

    Returns:
      A single numpy array representing the aggregated news for the day.
    """

    # Concatenate all arrays along the first axis (axis=0)
    concatenated_array = np.mean(np.array(daily_news_arrays), axis=0)

    # Calculate the mean of the concatenated array along axis 0
    # aggregated_array = np.mean(concatenated_array, axis=0)

    # Reshape the aggregated array to match the expected input shape
    aggregated_array = concatenated_array.reshape(1, 1024)

    return aggregated_array


# Create a dictionary to store dates and their corresponding IDs
dates_dict = defaultdict(list)

for dates in date_list:
    date = dates[0].split()[0]  # Extract date from the string
    dates_dict[date].append((dates[1]).reshape(-1))

# Print the resulting dictionary
aggregated_features = []
for date, ids in dates_dict.items():
    aggregated_features.append([date, aggregate_daily_news(np.array(ids))])

print(aggregated_features)
with open('../aggregated_feature_lists/9K_aggregated_date_feature_2024_9_26.pkl', 'wb') as f:
    pickle.dump(aggregated_features, f)
    print("list has been saved")
