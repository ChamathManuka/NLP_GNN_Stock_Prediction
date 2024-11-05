import pandas as pd

stock_name = 'UML'

# Read the CSV file into a DataFrame
df = pd.read_csv(stock_name+'.csv')

# Ensure 'Trade Date' is in the correct datetime format if needed
# df['Trade Date'] = pd.to_datetime(df['Trade Date'], format='%m/%d/%y')

# Calculate the price change in the 'Close (Rs.)' column
df[stock_name] = df['Close (Rs.)'].diff().fillna(df['Close (Rs.)'])

# Create a new DataFrame with 'Trade Date' and 'Price Change' columns
new_df = df[['Trade Date', stock_name]]

# Write the new DataFrame to a CSV file
new_df.to_csv(stock_name + '_Processed.csv', index=False)
