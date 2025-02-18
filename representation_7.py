import pandas as pd
import matplotlib.pyplot as plt
stock_name = 'JKH'
df = pd.read_csv("stock_outputs/"+stock_name+"_merged_output.csv")

# Create subplots
fig, ax1 = plt.subplots(figsize=(20, 12))

# Plot 'Price Difference1' on the first y-axis
ax1.plot(df['Date'], df['Price Difference1'], color='blue', label='Price Difference 1')
ax1.set_xlabel('Date')
ax1.set_ylabel('Price Difference 1', color='blue')
ax1.tick_params('y', color='blue')

# Create a second y-axis for 'Price Difference2'
ax2 = ax1.twinx()
ax2.plot(df['Date'], df['Price Difference2'], color='red', label='Price Difference 2')
ax2.set_ylabel('Price Difference 2', color='red')
ax2.tick_params('y', color='red')

# Customize plot
plt.title('Price Differences')
plt.legend(loc='upper right')
plt.grid(True)
plt.xticks(rotation=45)

# Show the plot
plt.show()