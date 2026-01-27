import pandas as pd
from keras.losses import mean_squared_error, mean_absolute_error
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score

stock_name = "HNB"

df_lstm = pd.read_pickle("evaluation_lstm_files/model_files/" + stock_name + "_predictions.pkl")

df_mlp = pd.read_pickle("evaluation_mpl_files/model_files/" + stock_name + "_results_df.pkl")

df_gnn = pd.read_pickle("../gnn_model_3_files/model_files/results_df.pkl")

df_lstm['date'] = pd.to_datetime(df_lstm['date'])
df_mlp['date'] = pd.to_datetime(df_mlp['date'])
df_gnn['date'] = pd.to_datetime(df_gnn['date'])

# Merge the DataFrames
# Start with the first two
merged_df = pd.merge(df_lstm, df_mlp, on='date', how='outer')

# Then merge the result with the third
final_merged_df = pd.merge(df_gnn, merged_df, on='date', how='outer')
final_merged_df = final_merged_df.drop(columns=['lstm_Actual'])
final_merged_df = final_merged_df.drop(columns=['mlp_Actual'])

# Display the merged DataFrame
print(final_merged_df.head())
print(final_merged_df.info())  # Check for missing values

# If you want to set 'date' as the index:
final_merged_df = final_merged_df.set_index('date')
# final_merged_df.to_csv("evaluation_lstm_files/model_files/" + stock_name + "_predictions.csv")


df_saved = pd.read_csv("evaluation_lstm_files/model_files/" + stock_name + "_predictions.csv")
x_values_test = df_saved['date'].values
test_original = df_saved['gnn_Actual'].values
lstm_test_predicted = df_saved['lstm_Predicted'].values
gnn_test_predicted = df_saved['gnn_Predicted'].values
mlp_test_predicted = df_saved['mlp_Predicted'].values

mse = mean_squared_error(test_original, gnn_test_predicted)
mae = mean_absolute_error(test_original, gnn_test_predicted)
r2 = r2_score(test_original, gnn_test_predicted)

print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R² Score: {r2:.4f}")
plt.figure(figsize=(40, 20))
plt.plot(range(len(x_values_test)), test_original, marker='o', linestyle='-', color='k', label="Actual")
# plt.plot(range(len(x_values_test)), lstm_test_predicted, marker='s', linestyle='-', color='g', label="LSTM Predicted")
plt.plot(range(len(x_values_test)), gnn_test_predicted, marker='s', linestyle='-', color='r', label="GNN Predicted")
# plt.plot(range(len(x_values_test)), mlp_test_predicted, marker='s', linestyle='-', color='b', label="MLP Predicted")

# Connect corresponding points with vertical dashed lines
for i in range(len(test_original)):
    plt.plot([i, i], [test_original[i], gnn_test_predicted[i]], 'k--', alpha=0.5)

# Set X-axis labels as dates
plt.xticks(range(len(x_values_test)), x_values_test, rotation=45, fontsize=12)

# Labels and title
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("Stock Price Prediction Over Time (Unseen Data)")
plt.legend()
plt.grid(True)
plt.show()
