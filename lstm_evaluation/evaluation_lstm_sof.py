import pandas as pd
import tensorflow as tf
from keras.losses import mean_squared_error, mean_absolute_error
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, LeakyReLU, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer
import joblib
import numpy as np
# import keras_tuner as kt
import tensorflow as tf
import itertools

from tensorflow.python.keras.models import load_model

# Initialize QuantileTransformer
# -----------------------------------------------------------------------------------------
qt = QuantileTransformer(output_distribution='uniform')




# -----------------------------------------------------------------------------------------------------------------------
import tensorflow as tf
from tensorflow.keras.layers import (LSTM, Dense, Dropout, BatchNormalization, LeakyReLU,
                                     Bidirectional, Attention, Input, Concatenate)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

def build_ma_bilstm_seq(dropout_rate=0.3, l2_reg=0.001):
    # Now expecting input shape (1, 1152) per sample
    inputs = Input(shape=(1, 1152))

    # No need for tf.expand_dims inside the model
    x = Dense(512, activation='relu', kernel_regularizer=l2(l2_reg))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # First Bi-LSTM (already sees shape (None, 1, 512))
    x = Bidirectional(LSTM(256, return_sequences=True, kernel_regularizer=l2(l2_reg)))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(dropout_rate)(x)

    # Second Bi-LSTM
    x = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(l2_reg)))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(dropout_rate)(x)

    # Self-attention + residual
    attn = Attention()([x, x])
    x = tf.keras.layers.Add()([x, attn])

    # Final Bi-LSTM
    x = Bidirectional(LSTM(64, kernel_regularizer=l2(l2_reg)))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(dropout_rate)(x)

    outputs = Dense(1, activation='linear')(x)
    return Model(inputs=inputs, outputs=outputs)


from tensorflow.keras.layers import Attention as _Attention

class Attention(_Attention):
    def __init__(self, *args, **kwargs):
        # drop score_mode if present
        kwargs.pop("score_mode", None)
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config):
        # drop score_mode before passing to super
        config.pop("score_mode", None)
        return super().from_config(config)

# -----------------------------------------------------------------------------------------------------------------------

def model_building(stock_name):
    df = pd.read_pickle("../evaluation_files/list_files/train.pkl")
    results_df = pd.DataFrame({
        'date': df['added_date']
    })
    df['sbert_vectors'] = df['sbert_vectors'].apply(np.array)
    df['fingpt_vectors'] = df['fingpt_vectors'].apply(np.array)

    df['vector'] = df.apply(lambda row: np.concatenate((row['sbert_vectors'], row['fingpt_vectors'])), axis=1)

    # Assuming the 'vector' column contains lists or arrays of 100 elements
    vector_df = pd.DataFrame(df['vector'].tolist(), index=df.index)

    # Combine the expanded vector columns with the rest of the DataFrame
    df = df.drop(columns=['vector']).join(vector_df)
    # Assuming df is the DataFrame with features and 'Daily Return' as the target

    X = df.drop(columns=['added_date', 'sbert_vectors', 'fingpt_vectors', 'neu_sent', 'pos_sent', 'neg_sent', 'return', 'price', 'Change %'])

    # X = vector_df  # Features
    X = X.to_numpy(dtype=np.float32)

    y = df['price']  # Target
    y = y.to_numpy(dtype=np.float32)

    # X = np.squeeze(X, axis=1)
    # Standardize the features (important for neural networks)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_scaler = StandardScaler()
    X_train_scaled = X_scaler.fit_transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    joblib.dump(X_scaler, "evaluation_lstm_files/model_files/" + stock_name + "_X_scaler.pkl")

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))
    joblib.dump(y_scaler, "evaluation_lstm_files/model_files/" + stock_name + "_y_scaler.pkl")

    # Reshape data for LSTM (Crucial step)
    X_train_scaled_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], 1,
                                                     X_train_scaled.shape[1])  # (samples, timesteps, features)
    X_test_scaled_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

    # X_train_scaled_reshaped = np.squeeze(X_train_scaled_reshaped, axis=1)

    model = build_ma_bilstm_seq()  # Create the model instance

    learning_rate = 0.000149
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Train the model
    history = model.fit(X_train_scaled_reshaped, y_train_scaled, epochs=400, batch_size=32,
                            validation_data=(X_test_scaled_reshaped, y_test_scaled))  # Add validation data

    # Make predictions (and inverse transform if needed)
    predictions_scaled = model.predict(X_test_scaled_reshaped)
    # predictions = y_scaler.inverse_transform(predictions_scaled)  # Inverse transform if you scaled your targets

    # Evaluate the model (example)
    loss = model.evaluate(X_test_scaled_reshaped, y_test_scaled, verbose=0)
    print(f"Test Loss: {loss}")

    # Save the trained model
    model.save("evaluation_lstm_files/model_files/" + stock_name + "_lstm_model")  # Save the model

    # Make predictions

    y_pred = y_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
    y_test = y_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

    y_pred = y_pred.reshape(-1)
    y_test = y_test[:100].tolist()
    y_pred = y_pred[:100].tolist()
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
    plt.title("Price Fluctuation Over Time - MLP-A-BiLSTM Model Testing")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Plot the loss curves
    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.title("Training vs Validation Loss - MLP-A-BiLSTM Model Training")
    plt.legend()
    plt.grid(True)
    plt.show()


# -----------------------------------------------------------------------------------------------------------------------
def parameter_tuning_lstm(stock_name):
    df = pd.read_pickle("../evaluation_lstm_files/list_files/train.pkl")

    df['sbert_vectors'] = df['sbert_vectors'].apply(np.array)
    df['fingpt_vectors'] = df['fingpt_vectors'].apply(np.array)

    df['vector'] = df.apply(lambda row: np.concatenate((row['sbert_vectors'], row['fingpt_vectors'])), axis=1)

    # Assuming the 'vector' column contains lists or arrays of 100 elements
    vector_df = pd.DataFrame(df['vector'].tolist(), index=df.index)

    # Combine the expanded vector columns with the rest of the DataFrame
    df = df.drop(columns=['vector']).join(vector_df)
    # Assuming df is the DataFrame with features and 'Daily Return' as the target

    X = df.drop(
        columns=['added_date', 'sbert_vectors', 'fingpt_vectors', 'neu_sent', 'pos_sent', 'neg_sent', 'change %',
                 'return', 'price'])

    # X = vector_df  # Features
    X = X.to_numpy(dtype=np.float32)

    y = df['price']  # Target
    y = y.to_numpy(dtype=np.float32)

    # Standardize the features (important for neural networks)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_scaler = StandardScaler()
    X_train_scaled = X_scaler.fit_transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    joblib.dump(X_scaler, "evaluation_lstm_files/model_files/" + stock_name + "_X_scaler.pkl")

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))
    joblib.dump(y_scaler, "evaluation_lstm_files/model_files/" + stock_name + "_y_scaler.pkl")

    # Reshape data for LSTM (Crucial step)
    X_train_scaled_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], 1,
                                                     X_train_scaled.shape[1])  # (samples, timesteps, features)
    X_test_scaled_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
    best_loss = float('inf')  # Initialize with a very high value
    best_params = {}

    param_grid = {
        'lstm1_units': [32, 64, 128, 256],  # Example values
        'lstm2_units': [32, 64, 128],
        'lstm3_units': [32, 64],
        'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
        'learning_rate': [0.0001, 0.001, 0.01],
        'batch_size': [16, 32, 64]
    }

    for i, params in enumerate(itertools.product(*param_grid.values())):
        print(f"Trial {i + 1}: Testing parameters: {dict(zip(param_grid.keys(), params))}")  # Print current trial's parameters
        # Create the model with current hyperparameters
        model = LSTMModel(lstm1_units=params[0], lstm2_units=params[1], lstm3_units=params[2],
                          dropout_rate=params[3])  # Pass the hyperparameter values to the model

        optimizer = Adam(learning_rate=params[4])
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        # Train the model
        history = model.fit(X_train_scaled_reshaped, y_train_scaled, epochs=50, batch_size=params[5],
                            validation_data=(X_test_scaled_reshaped, y_test_scaled),
                            verbose=0)  # Don't print for every epoch

        loss = history.history['val_loss'][-1]  # Get the final validation loss

        print(f"Trial {i + 1}: Validation Loss: {loss}")

        if loss < best_loss:
            best_loss = loss
            best_params = dict(zip(param_grid.keys(), params))

    print(f"Best Loss: {best_loss}")
    print(f"Best Parameters: {best_params}")



# -----------------------------------------------------------------------------------------------------------------------
def model_testing(stock_name):
    df = pd.read_pickle("../evaluation_files/list_files/test.pkl")
    results_df = pd.DataFrame({
        'date': df['added_date']
    })
    df['sbert_vectors'] = df['sbert_vectors'].apply(np.array)
    df['fingpt_vectors'] = df['fingpt_vectors'].apply(np.array)

    df['vector'] = df.apply(lambda row: np.concatenate((row['sbert_vectors'], row['fingpt_vectors'])), axis=1)

    # Assuming the 'vector' column contains lists or arrays of 100 elements
    vector_df = pd.DataFrame(df['vector'].tolist(), index=df.index)

    # Combine the expanded vector columns with the rest of the DataFrame
    df = df.drop(columns=['vector']).join(vector_df)
    # Assuming df is the DataFrame with features and 'Daily Return' as the target

    X = df.drop(
        columns=['added_date', 'sbert_vectors', 'fingpt_vectors', 'neu_sent', 'pos_sent', 'neg_sent',
                 'return', 'price', 'change %'])

    # X = vector_df  # Features
    X = X.to_numpy(dtype=np.float32)

    y = df['price']  # Target
    y = y.to_numpy(dtype=np.float32)


    X_scaler = joblib.load("evaluation_lstm_files/model_files/" + stock_name + "_X_scaler.pkl")
    X_scaled = X_scaler.transform(X)

    y_scaler = joblib.load("evaluation_lstm_files/model_files/" + stock_name + "_y_scaler.pkl")
    y_scaled = y_scaler.transform(y.reshape(-1, 1))


    # Reshape data for LSTM (Crucial step)
    X_scaled_reshaped = X_scaled.reshape(X_scaled.shape[0], 1,
                                                     X_scaled.shape[1])  # (samples, timesteps, features)


    # model = load_model("evaluation_lstm_files/model_files/" + stock_name + "_lstm_model")

    model = tf.keras.models.load_model(
        "../evaluation_lstm_files/model_files/JKH_lstm_model",
        custom_objects={"Attention": Attention}
    )
    predictions_scaled = model.predict(X_scaled_reshaped)

    y_pred = y_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
    y_test = y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")



    results_df['lstm_Predicted'] = y_pred
    results_df['lstm_Actual'] = y_test

    results_df.to_pickle("evaluation_lstm_files/model_files/" + stock_name + "_predictions.pkl")
# -----------------------------------------------------------------------------------------------------------------------

model_building("BIL")
# parameter_tuning_lstm("HNB")
# model_testing("HNB")