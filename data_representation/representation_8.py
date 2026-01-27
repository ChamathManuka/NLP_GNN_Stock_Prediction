import pandas as pd
def calculate_mean_by_date(filename):
  """
  Calculates the mean of 'diff1' and 'diff2' for each date in the CSV file.

  Args:
    filename: Path to the CSV file.

  Returns:
    A DataFrame with 'Date', 'mean_diff1', and 'mean_diff2' columns.
  """


  df = pd.read_csv(filename, names=['Date', 'Price Difference1', 'Price Difference2'])

  # Convert 'column_name' to numeric
  df['Price Difference1'] = pd.to_numeric(df['Price Difference1'], errors='coerce')
  df['Price Difference2'] = pd.to_numeric(df['Price Difference2'], errors='coerce')

  grouped_df = df.groupby('Date').mean()
  return grouped_df

# Example usage:
filename = '../stock_outputs/JKH_merged_output.csv'  # Replace with the actual filename
result_df = calculate_mean_by_date(filename)
print(result_df)