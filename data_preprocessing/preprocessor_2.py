import pandas as pd

stock_name = 'NDB2'
column_name = "Price Difference"
# Read the CSV file into a DataFrame
# df = pd.read_csv('test_stock_files/UML_apr1-30.csv')
# df = pd.read_csv('test_stock_files/'+stock_name+'_test.csv')
df = pd.read_csv('stock_price_files/' + stock_name + '.csv')

# Ensure 'Trade Date' is in the correct datetime format if needed
df['Date'] = pd.to_datetime(df['Trade Date'], format='%m/%d/%y')

# Calculate the price change in the 'Close (Rs.)' column
df[column_name] = df['Open (Rs.)'].diff().fillna(df['Open (Rs.)'])
# df[column_name] = df['Close (Rs.)'].diff().fillna(df['Close (Rs.)'])

# Create a new DataFrame with 'Trade Date' and 'Price Change' columns
new_df = df[['Date', column_name]]

# Write the new DataFrame to a CSV file
new_df.to_csv('processed_stock_files/' + stock_name + '_Processed_open.csv', index=False)
# new_df.to_csv('processed_test_stock_files/UML_apr1-30_test.csv', index=False)
# new_df.to_csv('processed_test_stock_files/'+stock_name + '_processed_test.csv', index=False)
