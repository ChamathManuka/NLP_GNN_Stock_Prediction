import pandas as pd
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


def load_and_prepare_data(filename, target_stock='JKH'):
    """
    Loads data from CSV, preprocesses text, and creates sequences.

    Args:
        filename: Path to the CSV file.
        target_stock: Name of the stock to predict (default: 'JKH').

    Returns:
        X_train: List of tokenized and padded news article sequences.
        y_train: List of corresponding stock price fluctuations.
    """
    df = pd.read_csv(filename)

    # Text preprocessing
    tokenizer = Tokenizer(num_words=5000)  # Adjust num_words as needed
    tokenizer.fit_on_texts(df['filtered_tokens'])
    sequences = tokenizer.texts_to_sequences(df['filtered_tokens'])
    max_sequence_length = max(len(seq) for seq in sequences)
    X_train = pad_sequences(sequences, maxlen=max_sequence_length)

    # Stock price fluctuation
    y_train = df[target_stock].values

    return X_train, y_train, tokenizer, max_sequence_length


def build_lstm_model(vocab_size, embedding_dim, max_sequence_length):
    """
    Builds an LSTM model for stock price prediction.

    Args:
        vocab_size: Size of the vocabulary.
        embedding_dim: Dimension of the word embeddings.
        max_sequence_length: Maximum length of the input sequences.

    Returns:
        A compiled LSTM model.
    """
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_sequence_length))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def train_model(model, X_train, y_train, epochs=10, batch_size=32):
    """
    Trains the LSTM model.

    Args:
        model: The compiled LSTM model.
        X_train: Training data (sequences).
        y_train: Training labels (stock price fluctuations).
        epochs: Number of training epochs.
        batch_size: Batch size for training.
    """
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)


def predict_stock_fluctuation(model, new_text, tokenizer, max_sequence_length):
    """
    Predicts the stock price fluctuation for a new news article.

    Args:
        model: The trained LSTM model.
        new_text: New news article text.
        tokenizer: Tokenizer used for text preprocessing.
        max_sequence_length: Maximum length of input sequences.

    Returns:
        Predicted stock price fluctuation.
    """
    sequence = tokenizer.texts_to_sequences([new_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)
    prediction = model.predict(padded_sequence)
    return prediction[0][0]


# Example Usage
filename = '../price_change_attached.csv'
X_train, y_train, tokenizer, max_sequence_length = load_and_prepare_data(filename)

vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100

model = build_lstm_model(vocab_size, embedding_dim, max_sequence_length)
train_model(model, X_train, y_train)


# Example prediction

def predict_news(model, csv_file, tokenizer, max_sequence_length):
    df = pd.read_csv(csv_file)
    test_data = df['text'].tolist()
    predicted_list = []
    for element in test_data:
        predicted_list.append(predict_stock_fluctuation(model, element, tokenizer, max_sequence_length))
    return sum(predicted_list) / len(predicted_list)


new_text = "['important', 'meeting', 'held', 'today', 'discuss', 'future', 'plan']"
predicted_fluctuation = predict_stock_fluctuation(model, new_text, tokenizer, max_sequence_length)
print("Predicted JKH fluctuation:", predicted_fluctuation)

model_path = '../stock_prediction_model.h5'
model.save(model_path)
