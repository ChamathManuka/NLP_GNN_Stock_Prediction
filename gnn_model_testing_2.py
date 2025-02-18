# 1️⃣ Load the model
import joblib
import networkx as nx
import numpy as np
import pandas as pd
from keras.layers import BatchNormalization
from keras.losses import mean_squared_error, mean_absolute_error
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from stellargraph import StellarGraph
from stellargraph.layer import GraphSAGE, MeanAggregator
from stellargraph.mapper import GraphSAGENodeGenerator
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model

# 1️⃣ Load the model
model = load_model("workflow_files/model_files/graphsage_stock_model.h5",
                   custom_objects={"GraphSAGE": GraphSAGE, "MeanAggregator": MeanAggregator,
                                   "BatchNormalization": BatchNormalization})

# 2️⃣ Load scalers
price_scaler = joblib.load("workflow_files/model_files/min_max_scaler.pkl")
return_scaler = joblib.load("workflow_files/model_files/return_scaler.pkl")

df1 = pd.read_pickle("workflow_files/list_files/price_attached_vectors.pkl")
df2 = pd.read_pickle("workflow_files/list_files/price_attached_vectors_test.pkl")
df_concat = pd.concat([df1, df2], ignore_index=True)


# Extract rows 30 to 50 (inclusive of 30, exclusive of 50)
df_test = df_concat.iloc[890:1050].copy()
# Extract remaining rows (excluding 30 to 50)
df = pd.concat([df_concat.iloc[:890], df_concat.iloc[1050:]]).copy()
# Reset index (optional)
df_test.reset_index(drop=True, inplace=True)
df.reset_index(drop=True, inplace=True)

# 4️⃣ Prepare the test data for prediction
# Normalize return values (scaled return)
df_test['return_scaled'] = return_scaler.transform(df_test[['return']])

# Create a graph for the test dataset similar to how we did for training
graph_test = nx.Graph()

# Add nodes (news & stock returns)
for index, row in df_test.iterrows():
    news_vector = np.concatenate([row['sbert_vectors'], row['fingpt_vectors']])  # Removed sentiment values
    graph_test.add_node(f"news_{index}", type="news", features=news_vector)
    graph_test.add_node(f"stock_{row['added_date']}", type="stock", return_scaled=row['return_scaled'])

# Compute cosine similarity separately
sbert_sim = cosine_similarity(np.stack(df_test['sbert_vectors'].tolist()))
fingpt_sim = cosine_similarity(np.stack(df_test['fingpt_vectors'].tolist()))

# Add news-news edges based on similarity
for i in range(len(df_test)):
    for j in range(i + 1, len(df_test)):
        sim_weight = (sbert_sim[i, j] + fingpt_sim[i, j]) / 2  # Average similarity of SBERT and FinGPT embeddings
        if sim_weight > 0.6:  # threshold, tune for better performance
            graph_test.add_edge(f"news_{i}", f"news_{j}", weight=sim_weight)

# Add news-stock edges
for index, row in df_test.iterrows():
    stock_node = f"stock_{row['added_date']}"
    if stock_node in graph_test.nodes:
        graph_test.add_edge(f"news_{index}", stock_node, weight=np.tanh(row['return_scaled']))

# Add stock-stock edges based on 5-day moving average of returns
df_test['5_day_moving_avg'] = df_test['return'].rolling(window=5).mean()

unique_dates_test = sorted(df_test['added_date'].unique())
for i in range(len(unique_dates_test) - 1):
    prev_return = df_test[df_test['added_date'] == unique_dates_test[i]]['return'].values[0]
    next_return = df_test[df_test['added_date'] == unique_dates_test[i + 1]]['return'].values[0]

    # Calculate the moving average for the window
    moving_avg = np.mean(df_test.loc[(df_test['added_date'] >= unique_dates_test[i]) &
                                     (df_test['added_date'] < unique_dates_test[i + 1]), '5_day_moving_avg'].values)

    # Add edge with the weight as the 5-day moving average (smoothed)
    graph_test.add_edge(f"stock_{unique_dates_test[i]}", f"stock_{unique_dates_test[i + 1]}", weight=moving_avg)

# Convert to StellarGraph
stellar_graph_test = StellarGraph.from_networkx(graph_test, node_features="features")

# Prepare the test nodes for prediction
test_nodes = [n for n in graph_test.nodes if n.startswith("stock_")]
test_targets = np.array([graph_test.nodes[node]['return_scaled'] for node in test_nodes])

# Create GraphSAGE Data Generator for test data
generator_test = GraphSAGENodeGenerator(stellar_graph_test, batch_size=50, num_samples=[10, 10, 5])
test_gen = generator_test.flow(test_nodes, test_targets)

# 5️⃣ Predict with the model (predict returns, not prices)
y_pred_scaled = model.predict(test_gen)
y_pred = return_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()  # Inverse scaling to get original returns
test_targets = return_scaler.inverse_transform(test_targets.reshape(-1, 1)).flatten()  # Inverse scaling to get original returns

# 6️⃣ Evaluation Metrics for Return (MSE, MAE, R² for Returns)
mse = mean_squared_error(test_targets, y_pred)
mae = mean_absolute_error(test_targets, y_pred)
r2 = r2_score(test_targets, y_pred)

print(f"Test MSE: {mse:.4f}\nTest MAE: {mae:.4f}\nTest R² Score: {r2:.4f}")

# 7️⃣ Plot the predictions vs actual return values for the test data
test_dates = [node.replace("stock_", "") for node in test_nodes]
sorted_indices = np.argsort(test_dates)  # Sort by date
x_values_test = np.array(test_dates)[sorted_indices]

y_test_original = test_targets[sorted_indices]
y_pred_original = y_pred[sorted_indices]

# Plot actual vs. predicted returns
plt.figure(figsize=(20, 10))
plt.plot(range(len(x_values_test)), y_test_original, marker='o', linestyle='-', color='b', label="Actual Returns")
plt.plot(range(len(x_values_test)), y_pred_original, marker='s', linestyle='-', color='r', label="Predicted Returns")

# Connect corresponding points with vertical dashed lines
for i in range(len(y_test_original)):
    plt.plot([i, i], [y_test_original[i], y_pred_original[i]], 'k--', alpha=0.5)

# Set X-axis labels as dates
plt.xticks(range(len(x_values_test)), x_values_test, rotation=45, fontsize=12)

# Labels and title
plt.xlabel("Date")
plt.ylabel("Stock Return")
plt.title("Stock Return Prediction Over Time (Unseen Data)")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
