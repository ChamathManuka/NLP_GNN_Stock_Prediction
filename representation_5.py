import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file

df = pd.read_csv('stock_outputs/UML_output_biz_version_1.csv')

# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])
# df_sorted = df.sort_values(by='Date')

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Price Difference'], marker='o', linestyle='-', color='green')

# Customize the plot
plt.xlabel('Date')
plt.ylabel('PDifference')
plt.title('Price Difference Over Time')
plt.grid(True)
plt.xticks(rotation=45)


plt.show()