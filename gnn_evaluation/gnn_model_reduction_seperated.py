from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import numpy as np
import pandas as pd

# Load your DataFrame
df = pd.read_pickle("../evaluation_files/list_files/train.pkl")

# Extract SBERT and FinGPT vectors
sbert_vectors = np.stack(df['sbert_vectors'].values)     # shape: (samples, 384)
fingpt_vectors = np.stack(df['fingpt_vectors'].values)   # shape: (samples, 768)

def train_sae(input_vectors, reduced_dim=128, epochs=50, batch_size=32):
    # Normalize inputs (MinMaxScaler for sigmoid output)
    scaler = MinMaxScaler()
    input_scaled = scaler.fit_transform(input_vectors)

    input_dim = input_vectors.shape[1]
    input_layer = Input(shape=(input_dim,))

    # Encoder
    encoded = Dense(256, activation='relu')(input_layer)
    encoded = Dense(128, activation='relu')(encoded)  # Bottleneck

    # Decoder
    decoded = Dense(256, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)

    # Autoencoder Model
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    # Train
    autoencoder.fit(input_scaled, input_scaled,
                    epochs=epochs, batch_size=batch_size,
                    shuffle=True, validation_split=0.1)

    # Extract encoder
    encoder = Model(inputs=input_layer, outputs=encoded)
    reduced_vectors = encoder.predict(input_scaled)

    return reduced_vectors, scaler


# Train SAE for SBERT
reduced_sbert, sbert_scaler = train_sae(sbert_vectors, reduced_dim=128)

# Train SAE for FinGPT
reduced_fingpt, fingpt_scaler = train_sae(fingpt_vectors, reduced_dim=128)


# Add reduced vectors as new columns
df['reduced_sbert'] = list(reduced_sbert)
df['reduced_fingpt'] = list(reduced_fingpt)

# Save updated DataFrame
df.to_pickle("evaluation_files/list_files/train_with_reduced_vectors.pkl")
