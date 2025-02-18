import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

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


def predict_news(model, csv_file, tokenizer, max_sequence_length):
    df = pd.read_csv(csv_file)
    test_data = df['text'].tolist()
    predicted_list = []
    for element in test_data:
        if(element is not None and isinstance(element, str) and element.strip() != ""):
            predicted_list.append(predict_stock_fluctuation(model, element, tokenizer, max_sequence_length))
    return predicted_list, sum(predicted_list) / len(predicted_list)

filename = 'price_change_attached.csv'
X_train, y_train, tokenizer, max_sequence_length = load_and_prepare_data(filename)



loaded_model = load_model("stock_prediction_model.h5" )
print(predict_news(loaded_model, "All_news_CSV_files/2024_09_30_test.csv", tokenizer, max_sequence_length))
