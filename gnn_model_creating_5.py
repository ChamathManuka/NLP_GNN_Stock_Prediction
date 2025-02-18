import pandas as pd
import numpy as np
import networkx as nx
import joblib
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from stellargraph import StellarGraph
from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE, MeanAggregator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from sklearn.metrics.pairwise import cosine_similarity

# 1️⃣ Load Data
df = pd.read_pickle("gnn_model_3_files/list_files/train.pkl")
# df = pd.read_pickle("vectorization_process_files/list_files/train_title.pkl")

# 2️⃣ Normalize return values
return_scaler = MinMaxScaler()
df['return_scaled'] = return_scaler.fit_transform(df[['return']])

# Save return scaler
joblib.dump(return_scaler, "gnn_model_3_files/model_files/return_scaler.pkl")

# 3️⃣ Initialize Graph
graph = nx.Graph()

# 4️⃣ Add nodes (news & stock prices)
for index, row in df.iterrows():
    news_vector = np.concatenate([row['sbert_vectors'], row['fingpt_vectors']])  # Removed sentiment values
    graph.add_node(f"news_{index}", type="news", features=news_vector)
    graph.add_node(f"stock_{row['added_date']}", type="stock", price=row['price'], change=row['return_scaled'])

# 5️⃣ Compute cosine similarity (SBERT & FinGPT)
sbert_sim = cosine_similarity(np.stack(df['sbert_vectors'].tolist()))
fingpt_sim = cosine_similarity(np.stack(df['fingpt_vectors'].tolist()))

# 6️⃣ Add news-news edges based on similarity
for i in range(len(df)):
    for j in range(i + 1, len(df)):
        sim_weight = sbert_sim[i, j]  # Average similarity
        fin_weight = fingpt_sim[i, j] # Average similarity
        if sim_weight > 0.8:  # threshold, tune for better performance
            graph.add_edge(f"news_{i}", f"news_{j}", weight=sim_weight)
        if fin_weight > 0.8:  # threshold, tune for better performance
            graph.add_edge(f"news_{i}", f"news_{j}", weight=fin_weight)

# 7️⃣ Add news-stock edges
for index, row in df.iterrows():
    stock_node = f"stock_{row['added_date']}"
    if stock_node in graph.nodes:
        graph.add_edge(f"news_{index}", stock_node, weight=np.tanh(row['return_scaled']))

# 8️⃣ Add stock-stock edges (simplified)
unique_dates = sorted(df['added_date'].unique())
for i in range(len(unique_dates) - 1):
    # Instead of moving average, just use tanh of the return
    curr_return = df[df['added_date'] == unique_dates[i]]['return_scaled'].values[0]
    graph.add_edge(
        f"stock_{unique_dates[i]}",
        f"stock_{unique_dates[i + 1]}",
        weight=np.tanh(curr_return)
    )

# 9️⃣ Convert to StellarGraph
stellar_graph = StellarGraph.from_networkx(graph, node_features="features")

# 🔟 Train-test split
stock_nodes = [n for n in graph.nodes if n.startswith("stock_")]
train_nodes, test_nodes = train_test_split(stock_nodes, test_size=0.2, random_state=42)
train_nodes, val_nodes = train_test_split(train_nodes, test_size=0.2, random_state=42)

train_targets = np.array([graph.nodes[node]['price'] for node in train_nodes])
val_targets = np.array([graph.nodes[node]['price'] for node in val_nodes])
test_targets = np.array([graph.nodes[node]['price'] for node in test_nodes])

# 1️⃣1️⃣ Normalize stock prices
scaler = MinMaxScaler()
train_targets = scaler.fit_transform(train_targets.reshape(-1, 1)).flatten()
val_targets = scaler.transform(val_targets.reshape(-1, 1)).flatten()
test_targets = scaler.transform(test_targets.reshape(-1, 1)).flatten()

joblib.dump(scaler, "gnn_model_3_files/model_files/min_max_scaler.pkl")

# 1️⃣2️⃣ Create GraphSAGE Data Generator
generator = GraphSAGENodeGenerator(stellar_graph, batch_size=50, num_samples=[10, 10, 5])
train_gen = generator.flow(train_nodes, train_targets, shuffle=False)
val_gen = generator.flow(val_nodes, val_targets)
test_gen = generator.flow(test_nodes, test_targets)

# 1️⃣3️⃣ Build Model
lr_schedule = ExponentialDecay(
    initial_learning_rate=0.001, decay_steps=1000, decay_rate=0.96, staircase=True
)
optimizer = Adam(learning_rate=lr_schedule)

graphsage = GraphSAGE(layer_sizes=[128, 64, 32], generator=generator, bias=True, dropout=0.3)
x_inp, x_out = graphsage.in_out_tensors()
x_out = BatchNormalization()(x_out)
predictions = Dense(1, activation='linear', kernel_regularizer=regularizers.l2(0.0005))(x_out)

model = Model(inputs=x_inp, outputs=predictions)
model.compile(optimizer=optimizer, loss='mse')

# 1️⃣4️⃣ Train Model
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    train_gen,
    epochs=100,
    validation_data=val_gen,
    callbacks=[early_stopping],
    verbose=1
)

# 1️⃣5️⃣ Save Model
model.save("gnn_model_3_files/model_files/graphsage_stock_model_3.h5")

# 1️⃣6️⃣ Load Model
custom_objects = {
    "GraphSAGE": GraphSAGE,
    "MeanAggregator": MeanAggregator,
    "BatchNormalization": BatchNormalization,
}
loaded_model = load_model(
    "gnn_model_3_files/model_files/graphsage_stock_model_3.h5", custom_objects=custom_objects
)

# 1️⃣7️⃣ Predict
y_pred = loaded_model.predict(test_gen)
y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
y_test_original = scaler.inverse_transform(test_targets.reshape(-1, 1)).flatten()

# 1️⃣8️⃣ Evaluation Metrics
mse = mean_squared_error(y_test_original, y_pred_original)
mae = mean_absolute_error(y_test_original, y_pred_original)
r2 = r2_score(y_test_original, y_pred_original)
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Plot Training vs Validation Loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss', color='b')
plt.plot(history.history['val_loss'], label='Validation Loss', color='r')
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.title("Training vs. Validation Loss - Proposed Model Training")
plt.legend()
plt.grid(True)
plt.show()

# Ensure x_values (dates) is sorted
test_dates = [node.replace("stock_", "") for node in test_nodes]
sorted_indices = np.argsort(test_dates)  # Sort by date
x_values = np.array(test_dates)[sorted_indices]  # Sorted date strings

x_values = x_values[:100]
y_test_original = y_test_original[:100]  # Ensure matching length
y_pred_original = y_pred_original[:100]

# Plot actual vs. predicted prices
plt.figure(figsize=(20, 10))
plt.plot(range(len(x_values)), y_test_original, marker='o', linestyle='-', color='b', label="Actual")
plt.plot(range(len(x_values)), y_pred_original, marker='s', linestyle='-', color='r', label="Predicted")

# Connect corresponding points with vertical dashed lines
for i in range(len(y_test_original)):
    plt.plot([i, i], [y_test_original[i], y_pred_original[i]], 'k--', alpha=0.5)

# Set X-axis labels as dates
plt.xticks(range(len(x_values)), x_values, rotation=45, fontsize=12)

# Labels and title
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("Stock Price Prediction Over Time - Proposed Model Testing")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
