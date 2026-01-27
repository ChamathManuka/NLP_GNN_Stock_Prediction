import pandas as pd

stock_name = 'UML'
# Read the first CSV (trade date and price change)
price_change_df = pd.read_csv('processed_stock_files/' + stock_name + '_Processed.csv')

# Strip any leading/trailing spaces from column names and data
price_change_df.columns = price_change_df.columns.str.strip()
price_change_df['Trade Date'] = price_change_df['Trade Date'].str.strip()

# Convert 'Trade Date' to datetime format
price_change_df['Trade Date'] = pd.to_datetime(price_change_df['Trade Date'], format='%m/%d/%y')

# Read the second CSV (added_date and filtered_tokens)
added_date_df = pd.read_csv('../price_change_attached.csv')

# Strip any leading/trailing spaces from column names and data
added_date_df.columns = added_date_df.columns.str.strip()
added_date_df['added_date'] = added_date_df['added_date'].str.strip()

# Convert 'added_date' to datetime format
added_date_df['added_date'] = pd.to_datetime(added_date_df['added_date'], format='%Y-%m-%d')

# Print out the date ranges for comparison
print("Trade Date range:", price_change_df['Trade Date'].min(), "to", price_change_df['Trade Date'].max())
print("Added Date range:", added_date_df['added_date'].min(), "to", added_date_df['added_date'].max())

# Merge the DataFrames on dates
merged_df = pd.merge(
    added_date_df,
    price_change_df,
    left_on='added_date',
    right_on='Trade Date',
    how='left'
)

# Forward-fill the 'Price Change' column for missing values
merged_df[stock_name] = merged_df[stock_name].fillna(method='ffill')

# Drop the 'Trade Date' column as it's redundant now
merged_df = merged_df.drop(columns=['Trade Date'])

# Check if 'Price Change' is filled correctly
print(merged_df.head(10))

# Write the result to a new CSV file
merged_df.to_csv('price_change_attached.csv', index=False)
