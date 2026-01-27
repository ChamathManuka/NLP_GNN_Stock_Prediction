import pickle
import re
from html.parser import HTMLParser

import joblib
import nltk
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from tensorflow.keras.models import load_model
from torch import nn
from transformers import AutoModel, AutoTokenizer

# Load the pre-trained FinancialBERT model
model_name = "ProsusAI/finBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the last output of the LSTM
        return out


# Define model parameters
# input_size = 768  # Dimension of BERT embeddings
hidden_size = 256  # Adjust as needed
num_layers = 5


class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return ''.join(self.fed)


def data_preprocessing():
    def cleaning():
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')
        processed_data = []

        df = pd.read_csv("../All_news_CSV_files/test_june_to_sept.csv")  # Replace with the actual encoding

        # df = pd.read_csv("All_news_CSV_files/news_articles.csv")
        for index, row in df.iterrows():

            title = row["title"]
            full_text = row["full_text"]
            if (isinstance(title, str) and isinstance(full_text, str)):
                text = title + ' ' + full_text
                text = text.lower()

                text = re.sub(r'[^\w\s]', '', text)

                text = re.sub(r'\d+', '', text)

                text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

                pattern = r"^\s*\/\*.*?\*\/\s*$|^\s*table\..*?\{.*?\}|MicrosoftInternetExplorer4|st1\:*{behavior:url\(#ieooui\)}|body\s*\{.*?\}|Normal\s*\d+\s*st1\:*\*"

                # Remove unwanted patterns using the regular expression
                cleaned_text = re.sub(pattern, "", text, flags=re.DOTALL | re.MULTILINE)

                text = ' '.join(cleaned_text.split())

                words = nltk.word_tokenize(text)

                stop_words = set(stopwords.words('english'))
                words = [word for word in words if not word in stop_words]

                lemmatizer = WordNetLemmatizer()
                words = [lemmatizer.lemmatize(word) for word in words]

                cleaned_text = ' '.join(words)
                print(cleaned_text)
                processed_data.append({"added_date": row["added_date"], "filtered_tokens": cleaned_text})

        processed_df = pd.DataFrame(processed_data)

        processed_df['added_date'] = pd.to_datetime(processed_df['added_date']).dt.date

        processed_df.to_csv("process_9_files/csv_files/test_june_to_sept_processed.csv", index=False)

    def tokenization():
        df = pd.read_csv("../process_9_files/csv_files/test_june_to_sept_processed.csv")
        # Create a new DataFrame to store the processed data
        processed_data = []
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words("english"))

        # Iterate through the rows of the DataFrame
        for index, row in df.iterrows():
            # Access data from each row
            paragraph = row['filtered_tokens']
            if (isinstance(paragraph, str)):
                print(paragraph)
                cleared_tokens = [re.sub(r"[^a-zA-Z]", "", token) for token in word_tokenize(paragraph)]
                tokens = [re.sub(r"\s+", " ", token) for token in cleared_tokens if token]
                stemmed_tokens = [stemmer.stem(token) for token in tokens]
                lemmatized_tokens = [lemmatizer.lemmatize(stemmed_token) for stemmed_token in stemmed_tokens]
                filtered_tokens = [token for token in lemmatized_tokens if token not in stop_words]
                # Add the processed data to the new DataFrame
                processed_data.append({"added_date": row["added_date"], "filtered_tokens": filtered_tokens})

        # Create a new DataFrame from the processed data
        processed_df = pd.DataFrame(processed_data)

        # Save the processed DataFrame to a new CSV file
        processed_df.to_csv("process_9_files/csv_files/tokenized_test_june_to_sept_processed.csv", index=False)

    cleaning()
    tokenization()


# --------------------------------------------------------------------------------------------------------
def stock_data_preprocessing(stock_name):
    stock = stock_name

    def calculate_daily_return():
        data = pd.read_csv("stock_price_files/" + stock + ".csv")
        data['Trade Date'] = pd.to_datetime(data['Trade Date'])
        data["Trade Date"] = data["Trade Date"]
        data = data.dropna(subset=['Close (Rs.)'])
        data = data[data['Close (Rs.)'].notna()]
        data['Daily_Return'] = data['Close (Rs.)'].diff()
        data['Daily_Return'].fillna(method='ffill', inplace=True)
        # Save the DataFrame to a CSV file
        data.to_csv('stock_price_files/percentage_price_files/' + stock + '_stock_data_with_returns.csv', index=False)

    def merge_daily_returns():
        df1 = pd.read_csv("../process_9_files/csv_files/tokenized_test_june_to_sept_processed.csv")
        df2 = pd.read_csv("stock_price_files/percentage_price_files/" + stock_name + "_stock_data_with_returns.csv")

        # Convert 'added_date' and 'Trade Date' to datetime objects
        df1['added_date'] = pd.to_datetime(df1['added_date']).dt.date
        df2['Trade Date'] = pd.to_datetime(df2['Trade Date']).dt.date

        # Select desired columns from each DataFrame
        df1 = df1[['added_date', 'filtered_tokens']]  # Replace with actual column names
        df2 = df2[['Trade Date', 'Daily_Return']]  # Replace with actual column names

        # Merge DataFrames based on 'added_date' and 'Trade Date'
        merged_df = pd.merge(df1, df2, left_on='added_date', right_on='Trade Date', how='inner')

        # Drop the 'trade_date' column from the merged DataFrame
        merged_df = merged_df.drop('Trade Date', axis=1)
        merged_df = merged_df.dropna(subset=['Daily_Return'])
        merged_df = merged_df[merged_df['Daily_Return'].notna()]
        # Save the merged DataFrame to a new CSV file
        merged_df.to_csv(
            'process_9_files/csv_files/' + stock_name + '_tokenized_test_june_to_sept_processed_merged.csv',
            index=False)

        print("Merged data saved process_9_files directory")

    calculate_daily_return()
    merge_daily_returns()
    print("Stock Data Preprocessing Complete", stock)


# --------------------------------------------------------------------------------------------------------
def prediction(stock_name):
    loaded_vectorizer = joblib.load("../process_9_files/tf-idf_model_files/tfidf_vectorizer5K_filtered.joblib")
    data = pd.read_csv('process_9_files/csv_files/' + stock_name + '_tokenized_test_june_to_sept_processed_merged.csv')
    model = load_model('../process_9_files/model_files/sequential_model.h5')

    processed_data = []
    # Iterate through sentences and extract embeddings
    for index, row in data.iterrows():
        if (isinstance(row['filtered_tokens'], str)):
            sentence = row['filtered_tokens']
            daily_return = row['Daily_Return']
            added_date = row['added_date']
            embedding = loaded_vectorizer.transform([sentence]).toarray()
            predicted_value = model.predict(embedding)

            processed_data.append([added_date, daily_return, predicted_value])

    # Convert lists to NumPy arrays

    with open('process_7.4_files/list_files/' + stock_name + '_test_june_to_sept_vectorized.pkl', 'wb') as f:
        pickle.dump(processed_data, f)


# data_preprocessing()
# stock_data_preprocessing("JKH")
prediction("JKH")
