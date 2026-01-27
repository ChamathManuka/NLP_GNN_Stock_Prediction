import re

import joblib
import nltk
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from keras import Input
from keras.layers import LeakyReLU, BatchNormalization, Dropout
from keras.losses import mean_absolute_error
from keras.optimizers import RMSprop, Adam
from keras.regularizers import l2
from keras.src.layers import Softmax, Multiply
from matplotlib import pyplot as plt
from nltk import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.models import load_model
# from textblob import TextBlob
import keras_tuner as kt
# Initialize QuantileTransformer
# -----------------------------------------------------------------------------------------
qt = QuantileTransformer(output_distribution='uniform')




# -----------------------------------------------------------------------------------------------------------------------

def parameter_tuning(stock_name):
    df = pd.read_pickle("../evaluation_mpl_files/list_files/test.pkl")

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
    joblib.dump(X_scaler, "evaluation_mpl_files/model_files/" + stock_name + "_X_scaler.pkl")

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))
    joblib.dump(y_scaler, "evaluation_mpl_files/model_files/" + stock_name + "_y_scaler.pkl")

    def build_model(hp):
        model = Sequential([
            Dense(hp.Int('units_1', min_value=0, max_value=512, step=64),
                  kernel_regularizer=l2(0.01), input_shape=(1152,)),#492
            LeakyReLU(alpha=0.1),
            BatchNormalization(),
            Dropout(0.2),

            Dense(hp.Int('units_2', min_value=0, max_value=256, step=64),
                  kernel_regularizer=l2(0.01)),
            LeakyReLU(alpha=0.1),
            BatchNormalization(),
            Dropout(0.2),

            Dense(hp.Int('units_3', min_value=0, max_value=128, step=64),
                  kernel_regularizer=l2(0.01)),
            LeakyReLU(alpha=0.1),
            BatchNormalization(),

            # Dense(hp.Int('units_4', min_value=0, max_value=128, step=64),
            #       kernel_regularizer=l2(0.01)),
            # LeakyReLU(alpha=0.1),
            # BatchNormalization(),
            #
            # Dense(hp.Int('units_5', min_value=0, max_value=64, step=64),
            #       kernel_regularizer=l2(0.01)),
            # LeakyReLU(alpha=0.1),
            # BatchNormalization(),

            Dense(1, activation='linear')  # Use 'relu' if output must be non-negative
        ])


        optimizer = RMSprop(learning_rate=hp.Float('learning_rate', min_value=1e-6, max_value=1e-3, sampling='log'))
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        return model


    tuner = kt.RandomSearch(build_model, objective='val_loss', max_trials=10, executions_per_trial=2,
                            directory='E:\Academic\master\master\parameter_tuning', project_name='mpl_tuning_'+stock_name+'_3')
    tuner.search(X_train_scaled, y_train_scaled, epochs=40, batch_size=8, validation_data=(X_test_scaled, y_test_scaled))

    # Retrieve the best trial (trial with the lowest validation loss or highest accuracy)
    best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]

    # Get the optimal hyperparameters from the best trial
    best_units_1 = best_trial.hyperparameters.get('units_1')
    best_units_2 = best_trial.hyperparameters.get('units_2')
    best_units_3 = best_trial.hyperparameters.get('units_3')
    # best_units_4 = best_trial.hyperparameters.get('units_4')
    # best_units_5 = best_trial.hyperparameters.get('units_5')
    best_learning_rate = best_trial.hyperparameters.get('learning_rate')

    # Print the optimal hyperparameters
    print(f"Best units_1: {best_units_1}")
    print(f"Best units_2: {best_units_2}")
    print(f"Best units_3: {best_units_3}")
    # print(f"Best units_4: {best_units_4}")
    # print(f"Best units_5: {best_units_5}")
    print(f"Best learning_rate: {best_learning_rate}")

    print("----------------------------------------------------------------------------------------------------------------------------------")



# -----------------------------------------------------------------------------------------------------------------------
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Add, ReLU

def build_attention_mlp(input_dim):
    x_in = Input(shape=(input_dim,))
    # Attention gate
    attn = Dense(input_dim, activation=None)(x_in)
    attn = Softmax()(attn)           # feature-wise weights
    x = Multiply()([x_in, attn])
    # Two hidden layers
    x = Dense(512, activation='relu', kernel_regularizer='l2')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu', kernel_regularizer='l2')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    out = Dense(1, activation='linear')(x)
    return Model(inputs=x_in, outputs=out)


# def model_building(stock_name):
#     df = pd.read_pickle("evaluation_files/list_files/train.pkl")
#
#     df['sbert_vectors'] = df['sbert_vectors'].apply(np.array)
#     df['fingpt_vectors'] = df['fingpt_vectors'].apply(np.array)
#
#     df['vector'] = df.apply(lambda row: np.concatenate((row['sbert_vectors'], row['fingpt_vectors'])), axis=1)
#
#     # Assuming the 'vector' column contains lists or arrays of 100 elements
#     vector_df = pd.DataFrame(df['vector'].tolist(), index=df.index)
#
#     # Combine the expanded vector columns with the rest of the DataFrame
#     df = df.drop(columns=['vector']).join(vector_df)
#     # Assuming df is the DataFrame with features and 'Daily Return' as the target
#
#     X = df.drop(columns=['added_date', 'sbert_vectors', 'fingpt_vectors', 'neu_sent', 'pos_sent', 'neg_sent','Change %', 'return', 'price'])
#
#     # X = vector_df  # Features
#     X = X.to_numpy(dtype=np.float32)
#
#     y = df['price']  # Target
#     y = y.to_numpy(dtype=np.float32)
#
#     # Standardize the features (important for neural networks)
#
#     # Split the data into training and test sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     X_scaler = StandardScaler()
#     X_train_scaled = X_scaler.fit_transform(X_train)
#     X_test_scaled = (X_scaler.transform
#                      (X_test))
#     joblib.dump(X_scaler, "evaluation_mpl_files/model_files/" + stock_name + "_X_scaler.pkl")
#
#     y_scaler = StandardScaler()
#     y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
#     y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))
#     joblib.dump(y_scaler, "evaluation_mpl_files/model_files/" + stock_name + "_y_scaler.pkl")
#
#
#
#     # Build a neural network model with ReLU hidden layers and linear output layer
#     model = Sequential([
#         Dense(320, activation='relu', input_shape=(1152,), kernel_regularizer=l2(0.001)),  # L2 regularization
#         LeakyReLU(alpha=0.1),
#         BatchNormalization(),
#         Dropout(0.2),  # Dropout layer (20% dropout rate)
#
#         Dense(192, activation='relu', kernel_regularizer=l2(0.001)),  # L2 regularization
#         LeakyReLU(alpha=0.1),
#         BatchNormalization(),
#         Dropout(0.2),  # Dropout layer (20% dropout rate)
#
#         Dense(32, activation='relu', kernel_regularizer=l2(0.001)),  # L2 regularization
#         LeakyReLU(alpha=0.1),
#         BatchNormalization(),
#
#
#         Dense(1, activation='linear')  # Use 'relu' if output must be non-negative
#     ])
#
#     # Compile the model with specified learning rate
#     learning_rate = 0.000149  # Adjust as needed
#     optimizer = Adam(learning_rate=learning_rate)  # Adam optimizer with learning rate
#     model.compile(optimizer=optimizer, loss='mean_squared_error')
#
#     # Train the model
#     history = model.fit(X_train_scaled, y_train_scaled, epochs=400, batch_size=32, verbose=1, validation_data=(X_test_scaled, y_test_scaled))
#     model.save("evaluation_mpl_files/model_files/" + stock_name + "_sequential.h5")
#     # Make predictions
#     y_pred_scaled = model.predict(X_test_scaled)
#     y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
#     y_test = y_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
#     # Evaluate the model
#     mse = mean_squared_error(y_test, y_pred)
#     mae = mean_absolute_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
#
#     print(f"MSE: {mse:.4f}")
#     print(f"MAE: {mae:.4f}")
#     print(f"R² Score: {r2:.4f}")
#
#     y_pred = y_pred.reshape(-1)
#     y_test = y_test[:100].tolist()
#     y_pred = y_pred[:100].tolist()
#     x_values = range(len(y_test))
#
#     # Plot both price lines
#     plt.figure(figsize=(20, 10))
#     plt.plot(x_values, y_test, marker='o', linestyle='-', color='b', label="Actual")
#     plt.plot(x_values, y_pred, marker='s', linestyle='-', color='r', label="Predicted")
#
#     # Connect each corresponding data point with a vertical line
#     for i in range(len(y_test)):
#         plt.plot([x_values[i], x_values[i]], [y_test[i], y_pred[i]], 'k--', alpha=0.5)  # Dashed vertical line
#
#     # Set X-axis labels to dates
#     plt.xticks(x_values, y_test)
#
#     # Labels and title
#     plt.xlabel("Date")
#     plt.ylabel("Price")
#     plt.title("Price Fluctuation Over Time - Sequential Model Testing")
#     plt.legend()
#     plt.grid(True)
#
#     # Show the plot
#     plt.show()
#
#     train_loss = history.history['loss']
#     val_loss = history.history['val_loss']
#
#     # Plot the loss curves
#     plt.plot(train_loss, label="Training Loss")
#     plt.plot(val_loss, label="Validation Loss")
#     plt.xlabel("Epochs")
#     plt.ylabel("Loss (MSE)")
#     plt.title("Training vs Validation Loss - Sequential Model Training")
#     plt.legend()
#     plt.grid(True)
#     plt.show()


# -----------------------------------------------------------------------------------------------------------------------


import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Dense, BatchNormalization, Dropout, Add, ReLU, LeakyReLU
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam


def model_building(stock_name):
    # --- DATA LOADING & PREPARATION ---
    df = pd.read_pickle("../evaluation_files/list_files/train.pkl")
    df['sbert_vectors'] = df['sbert_vectors'].apply(np.array)
    df['fingpt_vectors'] = df['fingpt_vectors'].apply(np.array)
    df['vector'] = df.apply(lambda r: np.concatenate((r['sbert_vectors'], r['fingpt_vectors'])), axis=1)

    # expand the 1D 'vector' into columns
    vector_df = pd.DataFrame(df['vector'].tolist(), index=df.index)
    df = df.drop(columns=['vector']).join(vector_df)

    # features & target
    X = df.drop(columns=[
        'added_date', 'sbert_vectors', 'fingpt_vectors',
        'neu_sent', 'pos_sent', 'neg_sent',
        'Change %', 'return', 'price'
    ]).to_numpy(dtype=np.float32)
    y = df['price'].to_numpy(dtype=np.float32)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # scale X
    X_scaler = StandardScaler()
    X_train_scaled = X_scaler.fit_transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    joblib.dump(X_scaler, f"evaluation_mpl_files/model_files/{stock_name}_X_scaler.pkl")

    # scale y
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))
    joblib.dump(y_scaler, f"evaluation_mpl_files/model_files/{stock_name}_y_scaler.pkl")

    # --- BUILD DEEP RESIDUAL MLP ---
    input_dim = X_train_scaled.shape[1]
    x_input = Input(shape=(input_dim,), name="features")

    # Block 0: initial dense
    x = Dense(1024, activation=None, kernel_regularizer=l2(1e-3))(x_input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.3)(x)

    # Residual Block 1
    y_block = Dense(1024, activation=None, kernel_regularizer=l2(1e-3))(x)
    y_block = BatchNormalization()(y_block)
    y_block = ReLU()(y_block)
    y_block = Dropout(0.3)(y_block)
    x = Add()([x, y_block])

    # Block 2
    x = Dense(512, activation='relu', kernel_regularizer=l2(1e-3))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Block 3
    x = Dense(256, activation='relu', kernel_regularizer=l2(1e-3))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Output
    output = Dense(1, activation='linear', name="price")(x)

    model = Model(inputs=x_input, outputs=output, name="residual_mlp")
    model.compile(
        optimizer=Adam(learning_rate=1.49e-4),
        loss='mean_squared_error'
    )

    # --- TRAIN ---
    history = model.fit(
        X_train_scaled, y_train_scaled,
        epochs=400, batch_size=32, verbose=1,
        validation_data=(X_test_scaled, y_test_scaled)
    )

    model.save(f"evaluation_mpl_files/model_files/{stock_name}_residual_mlp.h5")

    # --- EVALUATE ---
    y_pred_scaled = model.predict(X_test_scaled).flatten()
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = y_scaler.inverse_transform(y_test_scaled).flatten()

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"Residual MLP Results for {stock_name}:")
    print(f" → MSE: {mse:.4f}")
    print(f" → MAE: {mae:.4f}")
    print(f" → R² Score: {r2:.4f}")

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
    plt.title("Price Fluctuation Over Time - Deep Residual MLP Model Testing")
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
    plt.title("Training vs Validation Loss - Deep Residual MLP Model Training")
    plt.legend()
    plt.grid(True)
    plt.show()

def model_testing(stock_name):
    df = pd.read_pickle("../evaluation_mpl_files/list_files/test.pkl")

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
        columns=['added_date', 'sbert_vectors', 'fingpt_vectors', 'neu_sent', 'pos_sent', 'neg_sent', 'change %',
                 'return', 'price'])

    # X = vector_df  # Features
    X = X.to_numpy(dtype=np.float32)

    y = df['price']  # Target
    y = y.to_numpy(dtype=np.float32)


    X_scaler = joblib.load("evaluation_mpl_files/model_files/" + stock_name + "_X_scaler.pkl")

    X_test_scaled = (X_scaler.transform(X))


    y_scaler = joblib.load("evaluation_mpl_files/model_files/" + stock_name + "_y_scaler.pkl")
    y_test_scaled = y_scaler.transform(y.reshape(-1, 1))

    model =  load_model("evaluation_mpl_files/model_files/" + stock_name + "_sequential.h5",
                          custom_objects={'BatchNormalization': BatchNormalization, 'LeakyReLU': LeakyReLU})
    # Make predictions
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_test = y_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

    results_df['mlp_Predicted'] = y_pred
    results_df['mlp_Actual'] = y_test
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

    y_pred = y_pred.reshape(-1)
    y_test = y_test.tolist()
    y_pred = y_pred.tolist()
    x_values = range(len(y_test))

    # Plot both price lines
    plt.figure(figsize=(20, 10))
    plt.plot(x_values, y_test, marker='o', linestyle='-', color='b', label="Stock 1")
    plt.plot(x_values, y_pred, marker='s', linestyle='-', color='r', label="Stock 2")

    # Connect each corresponding data point with a vertical line
    for i in range(len(y_test)):
        plt.plot([x_values[i], x_values[i]], [y_test[i], y_pred[i]], 'k--', alpha=0.5)  # Dashed vertical line

    # Set X-axis labels to dates
    plt.xticks(x_values, y_test)

    # Labels and title
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Price Fluctuation Over Time - Sequential Model Testing")
    plt.legend()
    plt.grid(True)

    # Show the plot
    # plt.show()
    plt.figure(figsize=(40, 20))
    plt.plot(range(len(x_values)), y_test, marker='o', linestyle='-', color='b', label="Actual")
    plt.plot(range(len(x_values)), y_pred, marker='s', linestyle='-', color='r',
             label="Predicted")

    # Connect corresponding points with vertical dashed lines
    for i in range(len(y_test)):
        plt.plot([i, i], [y_test[i], y_pred[i]], 'k--', alpha=0.5)

    # Set X-axis labels as dates
    plt.xticks(range(len(x_values)), x_values, rotation=45, fontsize=12)

    # Labels and title
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.title("Stock Return Prediction Over Time (Unseen Data)")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()


    results_df.to_pickle("evaluation_mpl_files/model_files/" + stock_name + "_results_df.pkl")



# -----------------------------------------------------------------------------------------------------------------------

model_building("BIL")
# parameter_tuning("HNB")
# model_testing("HNB")
