import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import RobustScaler

df1 = pd.read_pickle("../workflow_files/list_files/price_attached_vectors.pkl")
df2 = pd.read_pickle("../workflow_files/list_files/price_attached_vectors_test.pkl")
df_concat = pd.concat([df1, df2], ignore_index=True)

# Extract rows 30 to 50 (inclusive of 30, exclusive of 50)
df_test = df_concat.iloc[890:1050].copy()
# Extract remaining rows (excluding 30 to 50)
df = pd.concat([df_concat.iloc[:890], df_concat.iloc[1050:]]).copy()

# Ensure 'return' column exists in both datasets
if 'return' in df.columns and 'return' in df_test.columns:
    plt.figure(figsize=(10, 6))

    sns.kdeplot(df["return"], label="Train", shade=True)
    sns.kdeplot(df_test["return"], label="Test", shade=True)

    plt.legend()
    plt.title("Return Distribution: Train vs Test")
    plt.xlabel("Stock Return")
    plt.ylabel("Density")
    plt.grid(True)

    plt.show()
else:
    print("❌ 'return' column not found in one of the datasets!")

scaler = RobustScaler()
df["price_scaled"] = scaler.fit_transform(df[["price"]])
df_test["price_scaled"] = scaler.transform(df_test[["price"]])

scaler = RobustScaler()
df["return_scaled"] = scaler.fit_transform(df[["return"]])
df_test["return_scaled"] = scaler.transform(df_test[["return"]])
# Transform test prices using the existing scaler

# df["log_price"] = np.log1p(df["price"])
# df_test["log_price"] = np.log1p(df_test["price"])
# # Plot scaled price distributions
# plt.figure(figsize=(10, 6))
#
# sns.kdeplot(df["log_price"], label="Train", shade=True)
# sns.kdeplot(df_test["log_price"], label="Test", shade=True)
#
# plt.legend()
# plt.title("Scaled Price Distribution: Train vs Test")
# plt.xlabel("Scaled Stock Price")
# plt.ylabel("Density")
# plt.grid(True)

# plt.show()

import matplotlib.pyplot as plt

df['log_return'] = np.log1p(df['return_scaled'])
plt.hist(df['log_return'], bins=50)
plt.title('Distribution of Stock Returns')
plt.show()
