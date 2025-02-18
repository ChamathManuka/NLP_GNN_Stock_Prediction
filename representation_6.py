import pandas as pd
import matplotlib.pyplot as plt

stock_name = 'UML'
# Load the main CSV file
# df = pd.read_csv('stock_outputs/'+stock_name+'_output_biz_version.csv')
df = pd.read_csv('processed_test_stock_files/'+stock_name+'_apr1-30_test.csv')
df['Date'] = pd.to_datetime(df['Date'])
df_sorted = df.sort_values(by='Date')

# Load the CSV file with additional price difference values
# df_extra = pd.read_csv('processed_test_stock_files/'+stock_name+'_apr1-30_test.csv')
df_extra = pd.read_csv('stock_outputs/'+stock_name+'_output_biz_version.csv')

# Plot the main data
plt.figure(figsize=(10, 6))
plt.plot(df_sorted['Date'], df_sorted['Price Difference'], marker='o', linestyle='-', color='blue', label='Main Data')

# Add horizontal lines for extra price difference values
for value in df_extra['Price Difference']:
    plt.axhline(y=value, color='red', linestyle='--')

# Customize the plot
plt.xlabel('Date')
plt.ylabel('Price Difference')
plt.title('Price Difference Over Time')
plt.grid(True)
plt.xticks(rotation=45)

# Adjust legend to avoid overlap
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.show()