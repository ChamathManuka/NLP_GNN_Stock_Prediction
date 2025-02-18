import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt
from stellargraph import StellarGraph
from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE, MeanAggregator
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import joblib

# 1️⃣ Load Test Data
df_test = pd.read_pickle("gnn_model_3_files/list_files/test.pkl")
# df_test = pd.read_pickle("vectorization_process_files/list_files/test_title.pkl")

# 2️⃣ Load saved scalers
return_scaler = joblib.load("gnn_model_3_files/model_files/return_scaler.pkl")
price_scaler = joblib.load("gnn_model_3_files/model_files/min_max_scaler.pkl")

# 3️⃣ Scale the 'return' column in the test data
df_test['return_scaled'] = return_scaler.transform(df_test[['return']])

# 4️⃣ Create the test graph
graph_test = nx.Graph()

# 5️⃣ Add nodes (news & stock)
for index, row in df_test.iterrows():
    news_vector = np.concatenate([row['sbert_vectors'], row['fingpt_vectors']])
    graph_test.add_node(f"news_{index}", type="news", features=news_vector)
    graph_test.add_node(f"stock_{row['added_date']}", type="stock", price=row['price'], change=row['return_scaled'])

# 6️⃣ Compute cosine similarity for SBERT & FinGPT
sbert_sim_test = cosine_similarity(np.stack(df_test['sbert_vectors'].tolist()))
fingpt_sim_test = cosine_similarity(np.stack(df_test['fingpt_vectors'].tolist()))

# 7️⃣ Add news-news edges based on similarity
for i in range(len(df_test)):
    for j in range(i + 1, len(df_test)):
        sim_weight = sbert_sim_test[i, j]  # Average similarity
        fin_weight = fingpt_sim_test[i, j]   # Average similarity
        if sim_weight > 0.8:
            graph_test.add_edge(f"news_{i}", f"news_{j}", weight=sim_weight)
        if fin_weight > 0.8:
            graph_test.add_edge(f"news_{i}", f"news_{j}", weight=fin_weight)

# 8️⃣ Add news-stock edges
for index, row in df_test.iterrows():
    stock_node = f"stock_{row['added_date']}"
    if stock_node in graph_test.nodes:
        graph_test.add_edge(f"news_{index}", stock_node, weight=np.tanh(row['return_scaled']))

# 9️⃣ Add stock-stock edges (removed rolling avg; using tanh of current return)
unique_dates_test = sorted(df_test['added_date'].unique())
for i in range(len(unique_dates_test) - 1):
    # Use tanh of the scaled return for the current date
    curr_return = df_test.loc[df_test['added_date'] == unique_dates_test[i], 'return_scaled'].values[0]
    graph_test.add_edge(
        f"stock_{unique_dates_test[i]}",
        f"stock_{unique_dates_test[i + 1]}",
        weight=np.tanh(curr_return)
    )

# 🔟 Convert to StellarGraph
stellar_graph_test = StellarGraph.from_networkx(graph_test, node_features="features")

# 1️⃣1️⃣ Prepare for testing
test_nodes = [n for n in graph_test.nodes if n.startswith("stock_")]
test_targets = np.array([graph_test.nodes[node]['price'] for node in test_nodes])

# Normalize stock prices using the saved scaler
test_targets_scaled = price_scaler.transform(test_targets.reshape(-1, 1)).flatten()

# 1️⃣2️⃣ Create GraphSAGE Data Generator for test data
generator_test = GraphSAGENodeGenerator(stellar_graph_test, batch_size=50, num_samples=[10, 10, 5])
test_gen = generator_test.flow(test_nodes, test_targets)  # We'll feed unscaled targets; same approach as training

# 1️⃣3️⃣ Load the trained model
custom_objects = {
    "GraphSAGE": GraphSAGE,
    "MeanAggregator": MeanAggregator,
    "BatchNormalization": BatchNormalization
}
loaded_model = load_model("gnn_model_3_files/model_files/graphsage_stock_model_3.h5", custom_objects=custom_objects)

# 1️⃣4️⃣ Predict on the test data
y_pred = loaded_model.predict(test_gen)
y_pred_original = price_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
y_test_original = price_scaler.inverse_transform(test_targets_scaled.reshape(-1, 1)).flatten()


y_pred = y_pred_original.reshape(-1)
y_test = y_test_original.tolist()
y_pred = y_pred.tolist()
x_values = range(len(y_test))

# Plot both price lines
plt.figure(figsize=(20, 10))
plt.plot(x_values, y_test, marker='o', linestyle='-', color='b', label="Actual")
plt.plot(x_values, y_pred, marker='s', linestyle='-', color='r', label="Predicted")

# Connect each corresponding data point with a vertical line
for i in range(len(y_test)):
    plt.plot([x_values[i], x_values[i]], [y_test[i], y_pred[i]], 'k--', alpha=0.5)  # Dashed vertical line

# Set X-axis labels to dates
plt.xticks(x_values, y_test)

# Labels and title
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Price Fluctuation Over Time - GNN Model Testing")
plt.legend()
plt.grid(True)

plt.show()

# 1️⃣5️⃣ Evaluate
mse = mean_squared_error(y_test_original, y_pred_original)
mae = mean_absolute_error(y_test_original, y_pred_original)
r2 = r2_score(y_test_original, y_pred_original)

print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R² Score: {r2:.4f}")


test_dates = [node.replace("stock_", "") for node in test_nodes]
sorted_indices = np.argsort(test_dates)  # Sort by date
x_values_test = np.array(test_dates)[sorted_indices]

y_test_original = test_targets[sorted_indices]
y_pred_original = y_pred_original[sorted_indices]

# Plot actual vs. predicted
plt.figure(figsize=(40, 20))
plt.plot(range(len(x_values_test)), y_test_original, marker='o', linestyle='-', color='b', label="Actual")
plt.plot(range(len(x_values_test)), y_pred_original, marker='s', linestyle='-', color='r', label="Predicted")

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

results_df = pd.DataFrame({
    'date': x_values_test,
    'gnn_Actual': y_test_original,  # Assuming y_test_original is a NumPy array or list
    'gnn_Predicted': y_pred_original   # Assuming y_pred_original is a NumPy array or list
})

results_df.to_pickle("gnn_model_3_files/model_files/results_df.pkl")