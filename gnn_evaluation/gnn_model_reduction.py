from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import numpy as np
import pandas as pd

# Load your training data
df = pd.read_pickle("../evaluation_files/list_files/train.pkl")

# Concatenate sbert_vectors and fingpt_vectors for each row
combined_vectors = np.stack(df.apply(lambda row: np.concatenate([row['sbert_vectors'], row['fingpt_vectors']]), axis=1))

# Check shape: (samples, features) → e.g., (1000, 1152)
print(combined_vectors.shape)

# Define input shape from combined_vectors
input_dim = combined_vectors.shape[1]  # e.g., 1152
input_layer = Input(shape=(input_dim,))

# Encoder Layers
encoded = Dense(512, activation='relu')(input_layer)
encoded = Dense(256, activation='relu')(encoded)
encoded = Dense(128, activation='relu')(encoded)  # Bottleneck

# Decoder Layers
decoded = Dense(256, activation='relu')(encoded)
decoded = Dense(512, activation='relu')(decoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

# Build and compile model
autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train autoencoder
autoencoder.fit(combined_vectors, combined_vectors,
                epochs=50, batch_size=32, shuffle=True, validation_split=0.1)


# Create encoder model to extract compressed 128-d vectors
encoder = Model(inputs=input_layer, outputs=encoded)
# Get reduced vectors
reduced_vectors = encoder.predict(combined_vectors)  # shape: (samples, 128)
df['reduced_vectors'] = list(reduced_vectors)

df.to_pickle("evaluation_files/list_files/train_with_reduced_vectors.pkl")
