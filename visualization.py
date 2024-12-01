import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming your data is in a CSV file with columns 'Date' and 'Price Change'
df = pd.read_csv('stock_outputs/JKH_output_3.csv')

# Calculate summary statistics
mean_change = df['Price Difference'].mean()
std_dev = df['Price Difference'].std()

# Visualize the distribution
plt.hist(df['Price Difference'], bins=20)
plt.xlabel('Price Difference')
plt.ylabel('Frequency')
plt.title('Distribution of Price Changes')
plt.show()

# Define risk tolerance (adjust this based on your preference)
risk_tolerance = 'moderate'  # Options: 'conservative', 'moderate', 'aggressive'


# Based on your risk tolerance and analysis, select a suitable range

price_range = (mean_change - 0.5 * std_dev, mean_change + 0.5 * std_dev)
print("conservative price range: ", price_range)

price_range = (mean_change - std_dev, mean_change + std_dev)
print("moderate price range: ", price_range)

price_range = (mean_change - 2 * std_dev, mean_change + 2 * std_dev)
print("aggressive price range: ", price_range)

print("actual price difference: ", )