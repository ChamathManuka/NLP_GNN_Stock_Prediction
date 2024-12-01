import os
import pickle
from datetime import datetime
import pandas as pd
from itertools import chain

with open('selected_list_files/.9selected_list_2024-09-30.pkl', 'rb') as f:
    loaded_list10 = pickle.load(f)
    print(loaded_list10)

def process_date_lists(date_lists):
  """Processes a list of lists of dates, removing timestamps and duplicates.

  Args:
    date_lists: A list of lists of datetime strings.

  Returns:
    A list of unique dates, without timestamps.
  """

  unique_dates = set()
  for date_list in date_lists:
    for dates_str in date_list[1]:
        for date_str in dates_str:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')  # Adjust format if needed
            date_only = date_obj.date().strftime('%Y-%m-%d')
            unique_dates.add(date_only)

  return list(unique_dates)

# Example usage:
# Assuming your list of lists is stored in a variable named `date_lists`
result = process_date_lists(loaded_list10)
print(result)



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
    df['Date'] = pd.to_datetime(df['Trade Date'], format='%m/%d/%y')

    # Create a new DataFrame to store the results
    result_df = pd.DataFrame(columns=['Date', 'Price Difference'])

    # Iterate through the date list and find corresponding price differences
    for date_str in date_list:
        date = pd.to_datetime(date_str)
        matching_row = df[df['Date'] == date]

        if not matching_row.empty:
            price_diff = matching_row['Price Difference'].values[0]
        else:
            price_diff = 0

        # Create a new DataFrame for the current date and price difference
        new_row = pd.DataFrame({'Date': [date_str], 'Price Difference': [price_diff]})

        # Concatenate the new row to the result DataFrame
        result_df = pd.concat([result_df, new_row], ignore_index=True)


    return result_df

stock_name = 'JKH'
result_df = process_csv_and_date_list(stock_name+"_Processed.csv", result, stock_name)

output_file = os.path.join("stock_outputs", stock_name+"_output_3.csv")

result_df.to_csv(output_file, index=False)