# 1️⃣ Load the model
import joblib
import networkx as nx
import numpy as np
import pandas as pd
from keras.layers import BatchNormalization
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from stellargraph import StellarGraph
from stellargraph.layer import GraphSAGE
from stellargraph.layer import MeanAggregator
from stellargraph.mapper import GraphSAGENodeGenerator
from tensorflow.keras.models import load_model

df_test = pd.read_pickle("vectorization_process_files/list_files/test.pkl")
# df_test = pd.read_pickle("workflow_files/list_files/test.pkl")



# Load saved scaler and model
return_scaler = joblib.load("workflow_files/model_files/return_scaler.pkl")
price_scaler = joblib.load("workflow_files/model_files/min_max_scaler.pkl")

# Assuming you have df_test as your unseen data (with the same structure as df)
# Apply transformations to the test data (e.g., scaling, PCA, etc.)
# Scale the 'return' column in the test data (apply the same scaling as training data)
df_test['return_scaled'] = return_scaler.transform(df_test[['return']])

# Calculate 5-day moving average of returns in the test data
df_test['5_day_moving_avg'] = df_test['return'].rolling(window=5).mean()

# Create the test graph (same way you did for training data)
graph_test = nx.Graph()

# Add nodes (news & stock prices) for the test data
for index, row in df_test.iterrows():
    news_vector = np.concatenate([row['sbert_vectors'], row['fingpt_vectors']])
    graph_test.add_node(f"news_{index}", type="news", features=news_vector)
    graph_test.add_node(f"stock_{row['added_date']}", type="stock", price=row['price'], change=row['return_scaled'])

# Compute cosine similarity separately for test data
sbert_sim_test = cosine_similarity(np.stack(df_test['sbert_vectors'].tolist()))
fingpt_sim_test = cosine_similarity(np.stack(df_test['fingpt_vectors'].tolist()))

# Add news-news edges based on similarity for test data
for i in range(len(df_test)):
    for j in range(i + 1, len(df_test)):
        sim_weight = (sbert_sim_test[i, j] + fingpt_sim_test[i, j]) / 2
        if sim_weight > 0.6:  # threshold, tune for better performance
            graph_test.add_edge(f"news_{i}", f"news_{j}", weight=sim_weight)

# Add news-stock edges for test data
for index, row in df_test.iterrows():
    stock_node = f"stock_{row['added_date']}"
    if stock_node in graph_test.nodes:
        graph_test.add_edge(f"news_{index}", stock_node, weight=np.tanh(row['return_scaled']))

# Add stock-stock edges based on 5-day moving average of returns for test data
unique_dates_test = sorted(df_test['added_date'].unique())
for i in range(len(unique_dates_test) - 1):
    moving_avg = np.mean(df_test.loc[(df_test['added_date'] >= unique_dates_test[i]) &
                                     (df_test['added_date'] < unique_dates_test[i + 1]), '5_day_moving_avg'].values)
    graph_test.add_edge(f"stock_{unique_dates_test[i]}", f"stock_{unique_dates_test[i + 1]}", weight=moving_avg)

# Convert to StellarGraph
stellar_graph_test = StellarGraph.from_networkx(graph_test, node_features="features")

# Extract the nodes from df_test for evaluation
test_nodes = [n for n in graph_test.nodes if n.startswith("stock_")]
test_targets = np.array([graph_test.nodes[node]['price'] for node in test_nodes])

# Normalize stock prices using the saved price scaler
test_targets_scaled = price_scaler.transform(test_targets.reshape(-1, 1)).flatten()

# Create GraphSAGE Data Generator for test data
generator_test = GraphSAGENodeGenerator(stellar_graph_test, batch_size=50, num_samples=[10, 10, 5])
test_gen = generator_test.flow(test_nodes, test_targets)

# Load the trained model
custom_objects = {"GraphSAGE": GraphSAGE, "MeanAggregator": MeanAggregator, "BatchNormalization": BatchNormalization}
loaded_model = load_model("workflow_files/model_files/graphsage_stock_model.h5", custom_objects=custom_objects)

# Predict on the test data (unseen)
y_pred = loaded_model.predict(test_gen)
y_pred_original = price_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
y_test_original = price_scaler.inverse_transform(test_targets_scaled.reshape(-1, 1)).flatten()

# df_pred = pd.DataFrame(y_pred_original, columns=["y_pred"])
# Q1 = df_pred['y_pred'].quantile(0.25)
# Q3 = df_pred['y_pred'].quantile(0.75)
# IQR = Q3 - Q1
# lower_bound = Q1 - 1.5 * IQR
# upper_bound = Q3 + 1.5 * IQR
# filtered_df = df_pred[(df_pred['y_pred'] >= lower_bound) & (df_pred['y_pred'] <= upper_bound)]
# y_pred_original = filtered_df['y_pred'].values
#




# Evaluate the model performance on test data
mse = mean_squared_error(y_test_original, y_pred_original)
mae = mean_absolute_error(y_test_original, y_pred_original)
r2 = r2_score(y_test_original, y_pred_original)

# Print evaluation metrics
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R² Score: {r2:.4f}")
test_dates = [node.replace("stock_", "") for node in test_nodes]
sorted_indices = np.argsort(test_dates)  # Sort by date
x_values_test = np.array(test_dates)[sorted_indices]

y_test_original = test_targets[sorted_indices]
y_pred_original = y_pred_original[sorted_indices]

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