import numpy as np
import networkx as nx
import pandas as pd
from keras import regularizers
from keras.losses import Huber

from sklearn.preprocessing import MinMaxScaler
from stellargraph import StellarGraph
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE, MeanAggregator
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model

df = pd.read_pickle("../workflow_files/list_files/biz_price_attached_vectors_not_concat.pkl")
# df2 = pd.read_pickle("workflow_files/list_files/price_attached_vectors_test.pkl")
# df_concat = pd.concat([df1, df2], ignore_index=True)
#
#
# # Extract rows 30 to 50 (inclusive of 30, exclusive of 50)
# df_test = df_concat.iloc[890:1050].copy()
# # Extract remaining rows (excluding 30 to 50)
# df = pd.concat([df_concat.iloc[:890], df_concat.iloc[1050:]]).copy()
# # Reset index (optional)
# df_test.reset_index(drop=True, inplace=True)
# df.reset_index(drop=True, inplace=True)

# Normalize return values
return_scaler = MinMaxScaler()
df['return_scaled'] = return_scaler.fit_transform(df[['return']])

# Save return scaler
joblib.dump(return_scaler, "../workflow_files/model_files/return_scaler.pkl")

# Feature Engineering: Add 5-day moving average of returns and volatility (standard deviation of returns)
df['5_day_moving_avg'] = df['return'].rolling(window=5).mean()
df['volatility'] = df['return'].rolling(window=5).std()

# Additional price-based features (Technical indicators)
df['price_change'] = df['price'].pct_change()  # Price change as percentage
df['log_return'] = np.log(df['price'] / df['price'].shift(1))  # Log return
df['sma_50'] = df['price'].rolling(window=50).mean()  # 50-day Simple Moving Average (SMA)
df['sma_200'] = df['price'].rolling(window=200).mean()  # 200-day Simple Moving Average (SMA)
df['rsi'] = 100 - (100 / (1 + (df['price'].diff().gt(0).rolling(window=14).mean() /
                               df['price'].diff().lt(0).rolling(window=14).mean())))  # Relative Strength Index (RSI)

# Calculate Momentum (Rate of Change)
df['momentum'] = df['price'].pct_change(periods=14)  # 14-day momentum

# Calculate Average True Range (ATR)
df['high_low'] = df['price'].shift(1) - df['price'].shift(1)  # Previous day's high-low
df['high_close'] = np.abs(df['price'].shift(1) - df['price'].shift(1))  # Previous day's high-close
df['low_close'] = np.abs(df['price'] - df['price'].shift(1))  # Previous day's low-close

df['ATR'] = np.max(df[['high_low', 'high_close', 'low_close']], axis=1)  # ATR calculation
df['ATR_14'] = df['ATR'].rolling(window=14).mean()  # 14-day ATR

# Drop missing values created by rolling windows
df = df.dropna()

# Sort and ensure unique dates for chronological splitting
df['added_date'] = pd.to_datetime(df['added_date'])
train_data = df[df['added_date'] < '2024-01-01']
test_data = df[df['added_date'] >= '2024-01-01']

# Initialize Graph
graph = nx.Graph()

# Preprocess news vectors efficiently
news_vectors = np.array([np.concatenate([row['sbert_vectors'], row['fingpt_vectors']]) for index, row in df.iterrows()])

# Add nodes (news & stock prices)
for index, row in df.iterrows():
    news_vector = np.concatenate([row['sbert_vectors'], row['fingpt_vectors']])  # Removed sentiment values
    # Include additional features from price-based indicators
    features = np.concatenate([news_vector,
                               [row['price_change'], row['log_return'], row['sma_50'], row['sma_200'], row['rsi'], row['momentum'], row['ATR_14'], row['volatility']]])
    graph.add_node(f"news_{index}", type="news", features=features)
    graph.add_node(f"stock_{row['added_date']}", type="stock", price=row['price'], change=row['return_scaled'])

# Compute cosine similarity separately
sbert_sim = cosine_similarity(np.stack(df['sbert_vectors'].tolist()))
fingpt_sim = cosine_similarity(np.stack(df['fingpt_vectors'].tolist()))

# Add news-news edges based on similarity
for i in range(len(df)):
    for j in range(i + 1, len(df)):
        sim_weight = (sbert_sim[i, j] + fingpt_sim[i, j]) / 2  # Average similarity of SBERT and FinGPT embeddings
        if sim_weight > 0.6:  # threshold, tune for better performance
            graph.add_edge(f"news_{i}", f"news_{j}", weight=sim_weight)

# Add news-stock edges
for index, row in df.iterrows():
    stock_node = f"stock_{row['added_date']}"
    if stock_node in graph.nodes:
        graph.add_edge(f"news_{index}", stock_node, weight=np.tanh(row['return_scaled']))

# Add stock-stock edges based on price indicators and 5-day moving average of returns
unique_dates = sorted(df['added_date'].unique())
for i in range(len(unique_dates) - 1):
    prev_return = df[df['added_date'] == unique_dates[i]]['return'].values[0]
    next_return = df[df['added_date'] == unique_dates[i + 1]]['return'].values[0]

    # Calculate the moving average for the window
    moving_avg = np.mean(df.loc[(df['added_date'] >= unique_dates[i]) &
                                 (df['added_date'] < unique_dates[i + 1]), '5_day_moving_avg'].values)

    # Get the price change, volatility, and momentum for the edge weight
    price_change = df[df['added_date'] == unique_dates[i]]['price_change'].values[0]
    volatility = df[df['added_date'] == unique_dates[i]]['volatility'].values[0]
    momentum = df[df['added_date'] == unique_dates[i]]['momentum'].values[0]

    # Add edge with the weight as the 5-day moving average (smoothed) + price change, volatility, and momentum
    graph.add_edge(f"stock_{unique_dates[i]}", f"stock_{unique_dates[i + 1]}",
                   weight=moving_avg + price_change + volatility + momentum)

# Convert to StellarGraph
stellar_graph = StellarGraph.from_networkx(graph, node_features="features")


# train_data = train_data.sort_values(by="added_date")  # Sort by date to maintain chronological order
# train_len = len(train_data)
# val_data = train_data[int(0.8 * train_len):]  # 20% for validation
# train_data = train_data[:int(0.8 * train_len)]  # 80% for training
#
# # Create train, validation, and test nodes based on the respective data
# train_nodes = [f"stock_{row['added_date']}" for index, row in train_data.iterrows()]
# val_nodes = [f"stock_{row['added_date']}" for index, row in val_data.iterrows()]
# test_nodes = [f"stock_{row['added_date']}" for index, row in test_data.iterrows()]


# Train-test split
stock_nodes = [n for n in graph.nodes if n.startswith("stock_")]
train_nodes, test_nodes = train_test_split(stock_nodes, test_size=0.2, random_state=42)
train_nodes, val_nodes = train_test_split(train_nodes, test_size=0.2, random_state=42)

train_targets = np.array([graph.nodes[node]['price'] for node in train_nodes])
val_targets = np.array([graph.nodes[node]['price'] for node in val_nodes])
test_targets = np.array([graph.nodes[node]['price'] for node in test_nodes])

# Normalize stock prices
scaler = MinMaxScaler()
train_targets = scaler.fit_transform(train_targets.reshape(-1, 1)).flatten()
val_targets = scaler.transform(val_targets.reshape(-1, 1)).flatten()
test_targets = scaler.transform(test_targets.reshape(-1, 1)).flatten()
joblib.dump(scaler, "../workflow_files/model_files/min_max_scaler.pkl")

# Create GraphSAGE Data Generator
generator = GraphSAGENodeGenerator(stellar_graph, batch_size=50, num_samples=[10, 5, 5])
train_gen = generator.flow(train_nodes, train_targets, shuffle=True)
val_gen = generator.flow(val_nodes, val_targets)
test_gen = generator.flow(test_nodes, test_targets)

# Build Model
lr_schedule = ExponentialDecay(initial_learning_rate=0.001, decay_steps=1000, decay_rate=0.96, staircase=True)
optimizer = Adam(learning_rate=lr_schedule)

graphsage = GraphSAGE(layer_sizes=[128, 64, 32], generator=generator, bias=True, dropout=0.3)
x_inp, x_out = graphsage.in_out_tensors()
x_out = BatchNormalization()(x_out)
predictions = Dense(1, activation='linear', kernel_regularizer=regularizers.l2(0.0005))(x_out)

model = Model(inputs=x_inp, outputs=predictions)
model.compile(optimizer=optimizer, loss='mse')

# Train Model
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(train_gen,
                    epochs=100,
                    validation_data=val_gen,
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
