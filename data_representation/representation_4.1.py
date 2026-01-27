import os
import pickle
from datetime import datetime
import pandas as pd
from itertools import chain

with open('../selected_list_files/2024_09_30_5K_ALL_test.pkl', 'rb') as f:
    loaded_list10 = pickle.load(f)
    print(loaded_list10)

def process_date_lists(date_lists):
  """Processes a list of lists of dates, removing timestamps and duplicates.

  Args:
    date_lists: A list of lists of datetime strings.

  Returns:
    A list of unique dates, without timestamps.
  """

  date_list_final = []
  for date_list in date_lists:
    unique_dates = set()
    temp_list = []
    for date_str in date_list[1]:

        date_obj = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')  # Adjust format if needed
        date_only = date_obj.date().strftime('%Y-%m-%d')
        unique_dates.add(date_only)




    # date_obj = datetime.strptime(date_list[0][0], '%Y-%m-%d %H:%M:%S')  # Adjust format if needed
    date_obj = datetime.strptime(date_list[0][0], '%Y-%m-%d')  # Adjust format if needed
    date_only = date_obj.date().strftime('%Y-%m-%d')
    temp_list.append(date_only)
    temp_list.append(unique_dates)
    date_list_final.append(temp_list)

  return list(date_list_final)

# Example usage:
# Assuming your list of lists is stored in a variable named `date_lists`
result = process_date_lists(loaded_list10)
print(result)



def extract_non_empty_sets(data):
  output = []
  for element in data:
    if element[1]:  # Check if the set is not empty
      output.append(element)
  return output


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

    # Create a new DataFrame to store the results
    result_df = pd.DataFrame(columns=['Date', 'Price Difference'])

    # Iterate through the date list and find corresponding price differences
    for date_str in date_list:
        date = pd.to_datetime(date_str[0])
        matching_row = df[df['Date'] == date]

        if not matching_row.empty:
            price_diff = matching_row['Price Difference'].values[0]
        else:
            price_diff = 0

        # Create a new DataFrame for the current date and price difference
        new_row = pd.DataFrame({'Date': [date_str[0]], 'Price Difference': [price_diff]})

        # Concatenate the new row to the result DataFrame
        result_df = pd.concat([result_df, new_row], ignore_index=True)


    return result_df


def process_csv_and_date_list2(csv_file, date_list, stock_name):

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Convert the 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Trade Date'])
    # df['Date'] = pd.to_datetime(df['Trade Date'], format='%m/%d/%y')

    # Create a new DataFrame to store the results
    result_df = pd.DataFrame(columns=['Date', 'Price Difference'])

    # Iterate through the date list and find corresponding price differences
    for date_str in date_list:
        accu_diff = 0
        for predict_date in date_str[1]:
            date = pd.to_datetime(predict_date)
            matching_row = df[df['Date'] == date]

            if not matching_row.empty:
                price_diff = matching_row['Price Difference'].values[0]
                accu_diff = accu_diff + price_diff
            else:
                price_diff = 0

            # Create a new DataFrame for the current date and price difference
            new_row = pd.DataFrame({'Date': [predict_date], 'Price Difference': [accu_diff/len(date_str[1])]})

        # Concatenate the new row to the result DataFrame
        result_df = pd.concat([result_df, new_row], ignore_index=True)


    return result_df



stock_name = 'HNB'
result_df1 = process_csv_and_date_list("processed_stock_files/"+stock_name+"2_Processed.csv", extract_non_empty_sets(result), stock_name)
result_df2 = process_csv_and_date_list2("processed_stock_files/"+stock_name+"2_Processed.csv", extract_non_empty_sets(result), stock_name)

output_file1 = os.path.join("../stock_outputs", stock_name + "2024_09_30_5K_ALL_test_1.csv")
output_file2 = os.path.join("../stock_outputs", stock_name + "2024_09_30_5K_ALL_test_2.csv")

result_df1.to_csv(output_file1, index=False)
result_df2.to_csv(output_file2, index=False)

def merge_two_column_csvs(file1, file2, output_file):

  df1 = pd.read_csv(file1, header=None, names=['Col1_1', 'Col1_2'])
  df2 = pd.read_csv(file2, header=None, names=['Col2_1', 'Col2_2'])

  # Assuming you want to merge by row index (simple concatenation)
  merged_df = pd.concat([df1, df2], axis=1)

  merged_df.to_csv(output_file, index=False)

# Example usage:
merge_two_column_csvs("stock_outputs/"+ stock_name + "2024_09_30_5K_ALL_test_1.csv", "stock_outputs/"+ stock_name + "2024_09_30_5K_ALL_test_2.csv", "stock_outputs/"+ stock_name + "_merged_output_all.csv")