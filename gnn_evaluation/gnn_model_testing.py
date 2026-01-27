import numpy as np
import networkx as nx
import pandas as pd
import joblib
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from stellargraph import StellarGraph
from stellargraph.mapper import GraphSAGENodeGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.layers import BatchNormalization, Dense
from stellargraph.layer import GraphSAGE, MeanAggregator

# 1️⃣ Load trained model & scalers
model = load_model("../workflow_files/model_files/graphsage_stock_model.h5",
                   custom_objects={"GraphSAGE": GraphSAGE, "MeanAggregator": MeanAggregator,
                                   "BatchNormalization": BatchNormalization})

price_scaler = joblib.load("../workflow_files/model_files/min_max_scaler.pkl")
return_scaler = joblib.load("../workflow_files/model_files/return_scaler.pkl")

# 2️⃣ Load unseen dataset
df_test = pd.read_pickle("../workflow_files/list_files/price_attached_vectors_test.pkl")

# Normalize return values
df_test['return_scaled'] = return_scaler.transform(df_test[['return']])

# 3️⃣ Construct the test graph
graph_test = nx.Graph()

for index, row in df_test.iterrows():
    news_vector = np.concatenate([row['sbert_vectors'], row['fingpt_vectors'], [row['neu_sent'], row['pos_sent'], row['neg_sent']]])
    graph_test.add_node(f"news_{index}", type="news", features=news_vector)
    graph_test.add_node(f"stock_{row['added_date']}", type="stock", price=row['price'], change=row['return_scaled'])

# Compute cosine similarity separately
sbert_sim = cosine_similarity(np.stack(df_test['sbert_vectors'].tolist()))
fingpt_sim = cosine_similarity(np.stack(df_test['fingpt_vectors'].tolist()))
sentiment_sim = cosine_similarity(df_test[['neu_sent', 'pos_sent', 'neg_sent']].values)

for i in range(len(df_test)):
    for j in range(i + 1, len(df_test)):
        sim_weight = (sbert_sim[i, j] + fingpt_sim[i, j] + sentiment_sim[i, j]) / 3
        if sim_weight > 0.6:
            graph_test.add_edge(f"news_{i}", f"news_{j}", weight=sim_weight)

for index, row in df_test.iterrows():
    stock_node = f"stock_{row['added_date']}"
    if stock_node in graph_test.nodes:
        graph_test.add_edge(f"news_{index}", stock_node, weight=np.tanh(row['return_scaled']))

unique_dates = sorted(df_test['added_date'].unique())
for i in range(len(unique_dates) - 1):
    prev_return = df_test[df_test['added_date'] == unique_dates[i]]['return_scaled'].values[0]
    graph_test.add_edge(f"stock_{unique_dates[i]}", f"stock_{unique_dates[i + 1]}", weight=np.tanh(df_test[df_test['added_date'] == unique_dates[i]]['return_scaled'].values[0]))

# Ensure stock nodes have features
for node in graph_test.nodes:
    if node.startswith("stock_"):
        graph_test.nodes[node]["features"] = np.zeros(1156)  # Match news node feature size

# Convert to StellarGraph
stellar_graph_test = StellarGraph.from_networkx(graph_test, node_features="features")

# 4️⃣ Prepare the test generator
test_stock_nodes = [n for n in graph_test.nodes if n.startswith("stock_")]
test_targets = np.array([graph_test.nodes[node]['price'] for node in test_stock_nodes])
test_targets = np.array(test_targets, dtype=np.float32)

# Normalize prices
test_targets = price_scaler.transform(test_targets.reshape(-1, 1)).flatten()

generator_test = GraphSAGENodeGenerator(stellar_graph_test, batch_size=50, num_samples=[10, 10, 5])
test_gen = generator_test.flow(test_stock_nodes, test_targets, shuffle=False)

# 5️⃣ Verify batch format
for batch in test_gen:
    print("Batch Type:", type(batch))  # Expect tuple
    print("First Element Type:", type(batch[0]))  # Expect numpy.ndarray
    if isinstance(batch[0], list):
        print("❌ Batch[0] is a list! Expected numpy array.")
    else:
        print("✅ Batch[0] is a NumPy array.")
    break

# 6️⃣ Make predictions
y_pred = model.predict(test_gen)

# Convert back to original scale
y_pred_original = price_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
y_test_original = price_scaler.inverse_transform(test_targets.reshape(-1, 1)).flatten()

# 7️⃣ Compute evaluation metrics
mse = mean_squared_error(y_test_original, y_pred_original)
mae = mean_absolute_error(y_test_original, y_pred_original)
r2 = r2_score(y_test_original, y_pred_original)

print(f"📊 Unseen Data Evaluation Metrics:")
print(f"🔹 Mean Squared Error (MSE): {mse:.4f}")
print(f"🔹 Mean Absolute Error (MAE): {mae:.4f}")
print(f"🔹 R² Score: {r2:.4f}")

# 6️⃣ Plot actual vs predicted stock prices
test_dates = [node.replace("stock_", "") for node in test_stock_nodes]
sorted_indices = np.argsort(test_dates)  # Ensure sorting by date
x_values = np.array(test_dates)[sorted_indices]

x_values = x_values[:100]  # Limit for better visualization
y_test_original = y_test_original[:100]
y_pred_original = y_pred_original[:100]

plt.figure(figsize=(20, 10))
plt.plot(range(len(x_values)), y_test_original, marker='o', linestyle='-', color='b', label="Actual")
plt.plot(range(len(x_values)), y_pred_original, marker='s', linestyle='-', color='r', label="Predicted")

# Draw vertical lines between actual & predicted points
for i in range(len(y_test_original)):
    plt.plot([i, i], [y_test_original[i], y_pred_original[i]], 'k--', alpha=0.5)

# Set X-axis labels as dates
plt.xticks(range(len(x_values)), x_values, rotation=45, fontsize=12)

# Labels and title
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("Stock Price Prediction on Unseen Data")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
