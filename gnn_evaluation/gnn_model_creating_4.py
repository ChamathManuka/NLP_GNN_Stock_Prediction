import pandas as pd
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from stellargraph.layer import GraphSAGE, MeanAggregator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import BatchNormalization, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from stellargraph import StellarGraph
from stellargraph.mapper import GraphSAGENodeGenerator
import joblib

df = pd.read_pickle("../workflow_files/list_files/train.pkl")
# df_test = pd.read_pickle("workflow_files/list_files/test.pkl")

# Normalize return values
return_scaler = MinMaxScaler()
df['return_scaled'] = return_scaler.fit_transform(df[['return']])

# Save return scaler
joblib.dump(return_scaler, "../workflow_files/model_files/return_scaler.pkl")

# Feature Engineering: Add 5-day moving average of returns and volatility (standard deviation of returns)
df['5_day_moving_avg'] = df['return'].rolling(window=5).mean()
df['volatility'] = df['return'].rolling(window=5).std()

# Sort and ensure unique dates for chronological splitting
df['added_date'] = pd.to_datetime(df['added_date'])
train_data = df[df['added_date'] < '2024-01-01']
test_data = df[df['added_date'] >= '2024-01-01']

# Initialize Graph
graph = nx.Graph()

# Preprocess news vectors efficiently
news_vectors = np.array([np.concatenate([row['sbert_vectors'], row['fingpt_vectors']]) for index, row in df.iterrows()])

# Add nodes (news & stock returns)
for index, row in df.iterrows():
    graph.add_node(f"news_{index}", type="news", features=news_vectors[index])
    graph.add_node(f"stock_{row['added_date']}", type="stock", return_scaled=row['return_scaled'])

# Compute cosine similarity for news vectors
sbert_sim = cosine_similarity(np.stack(df['sbert_vectors'].tolist()))
fingpt_sim = cosine_similarity(np.stack(df['fingpt_vectors'].tolist()))

# Add news-news edges based on similarity
for i in range(len(df)):
    for j in range(i + 1, len(df)):
        sim_weight = (sbert_sim[i, j] + fingpt_sim[i, j]) / 2
        if sim_weight > 0.6:
            graph.add_edge(f"news_{i}", f"news_{j}", weight=sim_weight)

# Add stock-stock edges based on 5-day moving average of returns
unique_dates_test = sorted(set(df['added_date']))
for i in range(len(unique_dates_test) - 1):
    moving_avg = np.mean(df.loc[(df['added_date'] >= unique_dates_test[i]) &
                                 (df['added_date'] < unique_dates_test[i + 1]), '5_day_moving_avg'].values)
    graph.add_edge(f"stock_{unique_dates_test[i]}", f"stock_{unique_dates_test[i + 1]}", weight=moving_avg)

# Convert to StellarGraph
stellar_graph = StellarGraph.from_networkx(graph, node_features="features")

# Train-test split (chronologically split)
stock_nodes = [n for n in graph.nodes if n.startswith("stock_")]
train_nodes, test_nodes = train_test_split(stock_nodes, test_size=0.2, random_state=42)
train_nodes, val_nodes = train_test_split(train_nodes, test_size=0.2, random_state=42)


train_targets = np.array([graph.nodes[node]['return_scaled'] for node in train_nodes])
test_targets = np.array([graph.nodes[node]['return_scaled'] for node in test_nodes])

# Normalize stock returns (already done for df)
scaler = MinMaxScaler()
train_targets = scaler.fit_transform(train_targets.reshape(-1, 1)).flatten()
test_targets = scaler.transform(test_targets.reshape(-1, 1)).flatten()

# Save the scaler
joblib.dump(scaler, "../workflow_files/model_files/min_max_scaler.pkl")

# Create GraphSAGE Data Generator
generator = GraphSAGENodeGenerator(stellar_graph, batch_size=50, num_samples=[10, 10, 5])
train_gen = generator.flow(train_nodes, train_targets, shuffle=True)
test_gen = generator.flow(test_nodes, test_targets)

# Build Model with higher complexity and more layers
optimizer = Adam(learning_rate=0.0005)  # Reduced learning rate
graphsage = GraphSAGE(layer_sizes=[512, 256, 128], generator=generator, bias=True, dropout=0.5)  # Increased complexity
x_inp, x_out = graphsage.in_out_tensors()
x_out = BatchNormalization()(x_out)
predictions = Dense(1, activation='linear', kernel_regularizer=regularizers.l2(0.01))(x_out)  # Added L2 regularization

model = Model(inputs=x_inp, outputs=predictions)
model.compile(optimizer=optimizer, loss='mse')  # Changed loss function for regression

# Train Model with more epochs
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(train_gen,
                    epochs=150,
                    validation_data=test_gen,
                    callbacks=[early_stopping],
                    verbose=1)

# Save Model
model.save("workflow_files/model_files/graphsage_stock_model.h5")

# Load Model
custom_objects = {"GraphSAGE": GraphSAGE, "MeanAggregator": MeanAggregator, "BatchNormalization": BatchNormalization}
loaded_model = load_model("../workflow_files/model_files/graphsage_stock_model.h5", custom_objects=custom_objects)

# Predict
y_pred = loaded_model.predict(test_gen)
y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
y_test_original = scaler.inverse_transform(test_targets.reshape(-1, 1)).flatten()

# Evaluation Metrics
mse = mean_squared_error(y_test_original, y_pred_original)
mae = mean_absolute_error(y_test_original, y_pred_original)
r2 = r2_score(y_test_original, y_pred_original)
print(f"MSE: {mse:.4f}\nMAE: {mae:.4f}\nR² Score: {r2:.4f}")


# Plot Training vs Validation Loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.title("Training vs. Validation Loss")
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
plt.title("Stock Price Prediction Over Time")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
