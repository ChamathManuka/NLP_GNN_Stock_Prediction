import pandas as pd

df1 = pd.read_pickle("../workflow_files/list_files/price_attached_vectors.pkl")
# df1 = pd.read_pickle("vectorization_process_files/list_files/merged_biz_vectors_title.pkl")
df2 = pd.read_pickle("../workflow_files/list_files/price_attached_vectors_test.pkl")
# df2 = pd.read_pickle("vectorization_process_files/list_files/merged_biz_vectors_title_test.pkl")
df = pd.concat([df1, df2], ignore_index=True)

# # Extract rows 30 to 50 (inclusive of 30, exclusive of 50)
# df_test = df_concat.iloc[200:400].copy()
# # Extract remaining rows (excluding 30 to 50)
# df = pd.concat([df_concat.iloc[:200], df_concat.iloc[300:]]).copy()
#
#
# # Reset index (optional)
# df_test.reset_index(drop=True, inplace=True)
# df.reset_index(drop=True, inplace=True)


import seaborn as sns
import matplotlib.pyplot as plt

sns.kdeplot(df["price"], label="Full Dataset", shade=True)
plt.legend()
plt.title("Price Distribution: Full Dataset")
plt.xlabel("Price")
plt.ylabel("Density")
plt.grid(True)
plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split

# Suppose 'df' is your DataFrame with a 'price' column
# Increase the number of bins (e.g., q=20)
df["price_bin"] = pd.qcut(df["price"], q=20, duplicates="drop", labels=False)

df_train, df_test = train_test_split(df, test_size=200, stratify=df["price_bin"], random_state=42)

df_train.drop(columns=["price_bin"], inplace=True)
df_test.drop(columns=["price_bin"], inplace=True)

# import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

sns.kdeplot(df_train["price"], label="Train", shade=True)
sns.kdeplot(df_test["price"], label="Test", shade=True)
plt.legend()
plt.show()

ks_stat, ks_p_value = ks_2samp(df_train["price"], df_test["price"])
print(f"KS p-value = {ks_p_value:.4f}")

df_train.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)

df_train.to_pickle("evaluation_files/list_files/train.pkl")
df_test.to_pickle("evaluation_files/list_files/test.pkl")

# import seaborn as sns
# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(10, 5))
# sns.kdeplot(df["price"], label="Train", shade=True)
# sns.kdeplot(df_test["price"], label="Test", shade=True)
#
# plt.legend()
# plt.title("Price Distribution: Train vs Test")
# plt.xlabel("Price")
# plt.ylabel("Density")
# plt.grid(True)
# plt.show()
#
# from scipy.stats import ks_2samp
#
# ks_stat, ks_p_value = ks_2samp(df["price"], df_test["price"])
# print(f"KS Test Statistic: {ks_stat:.4f}, p-value: {ks_p_value:.4f}")
#
# if ks_p_value > 0.05:
#     print("✅ The distributions are similar (Fail to reject null hypothesis).")
# else:
#     print("⚠️ The distributions are different (Reject null hypothesis).")
