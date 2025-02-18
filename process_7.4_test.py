import pickle
import re
from html.parser import HTMLParser

import nltk
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas.tseries.offsets import BDay
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from scipy.stats import stats, pearsonr
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from gensim.models import Word2Vec

from transformers import AutoModel, AutoTokenizer
from torch import nn
import torch



# Load the pre-trained FinancialBERT model
model_name = "ProsusAI/finBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return ''.join(self.fed)


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

def data_cleaning():

    def strip_tags(html):
        s = MLStripper()
        s.feed(html)
        return s.get_data()

    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    processed_data = []
    df = pd.read_csv("All_news_CSV_files/test_june_to_sept.csv")
    for index, row in df.iterrows():
        # Access data from each row
        full_text = row["full_text"]
        title = row["title"]
        if (isinstance(full_text, str) and isinstance(title, str)):

            full_text = re.sub(r'\{.*?\}', '', full_text)

            # Remove CSS-like style definitions
            full_text = re.sub(r'^\s*\/\*.*?\*\/\s*$|^\s*table\..*?\{.*?\}|body\s*\{.*?\}', '', full_text,
                               flags=re.DOTALL | re.MULTILINE)

            pattern = r"^\s*\/\*.*?\*\/\s*$|^\s*table\..*?\{.*?\}|MicrosoftInternetExplorer4|st1\:*{behavior:url\(#ieooui\)}|body\s*\{.*?\}"
            reg_text = re.sub(pattern, "", full_text, flags=re.DOTALL | re.MULTILINE)

            # Remove HTML/XML tags
            full_text = re.sub(r'<.*?>', '', reg_text)

            # Remove multiple newlines
            full_text = re.sub(r'\n+', '\n', full_text)

            # Remove excessive whitespace
            full_text = ' '.join(full_text.split())

            stripped = strip_tags(full_text)

            text = title + full_text
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

            processed_data.append({"added_date": row["added_date"], "filtered_tokens": cleaned_text})

    processed_df = pd.DataFrame(processed_data)

    processed_df['added_date'] = pd.to_datetime(processed_df['added_date']).dt.date

    # Group by 'Date' and concatenate 'filtered_tokens'
    df_grouped = processed_df.groupby('added_date')['filtered_tokens'].agg(' '.join).reset_index()
    # df_grouped = df_grouped[df_grouped['added_date'].apply(is_business_day_excluding_thursdays_fridays)]

    df_grouped.to_csv("process_7.4_files/csv_files/test_june_to_sept_processed.csv", index=False)

#--------------------------------------------------------------------------------------------------------
def stock_data_preprocessing(stock_name):
    stock = stock_name
    def calculate_percentage_change():
        data = pd.read_csv('stock_price_files/filled_price_files/' + stock + '_filled_stock_data.csv')
        data["Trade Date"] = data["Trade Date"]
        data['Daily_Return'] = data['Close (Rs.)'].diff()
        data['Daily_Return'].fillna(0, inplace=True)
        # Save the DataFrame to a CSV file
        data.to_csv('stock_price_files/percentage_price_files/'+stock+'_stock_data_with_returns.csv', index=False)

    def fill_missing_dates_values():
        df = pd.read_csv("stock_price_files/" + stock + ".csv")

        # Convert 'Trade Date' to datetime format
        df['Trade Date'] = pd.to_datetime(df['Trade Date'])

        # Set 'Trade Date' as the index and ensure uniqueness
        df.set_index('Trade Date', inplace=True)
        df = df[~df.index.duplicated(keep='first')]

        # Generate a DatetimeIndex with all days within the date range
        start_date = df.index.min()
        end_date = df.index.max()
        date_range = pd.date_range(start=start_date, end=end_date)

        # Reindex the DataFrame to include all days
        df = df.reindex(date_range)
        print(df)

        df.to_csv('stock_price_files/filled_price_files/' + stock + '_filled_stock_data.csv', index_label='Trade Date')
    def merge_daily_returns():

        df1 = pd.read_csv("process_7.4_files/csv_files/test_june_to_sept_processed.csv")
        df2 = pd.read_csv("stock_price_files/percentage_price_files/" + stock_name + "_stock_data_with_returns.csv")
        df2 = df2.dropna(subset=['Close (Rs.)'])
        df2 = df2[df2['Close (Rs.)'].notna()]
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

        # Save the merged DataFrame to a new CSV file
        merged_df.to_csv('process_7.4_files/csv_files/' + stock_name + '_sept_to_june_merged_data_with_daily_return.csv',
                         index=False)
        print("Merged data saved process_7.4_files directory")

    fill_missing_dates_values()
    calculate_percentage_change()
    merge_daily_returns()
    print("Stock Data Preprocessing Complete",stock)
#--------------------------------------------------------------------------------------------------------
def sentiment_vectorization(stock_name):

    def get_sentence_embedding(sentence):
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        sentence_vector = outputs.last_hidden_state[0, 0, :].detach().numpy()
        return sentence_vector.reshape(1, -1)

    def create_model_data(stock_name):

        def find_significant_features(embedding):
            with open('process_7.4_files/list_files/' + stock_name + '_significant_dimensions.pkl', 'rb') as f:
                significant_dimensions = pickle.load(f)
            return embedding[:, significant_dimensions]

        # Load data from CSV
        data = pd.read_csv( 'process_7.4_files/csv_files/' + stock_name + '_sept_to_june_merged_data_with_daily_return.csv')

        # Create lists to store embeddings and daily returns
        processed_data = []
        # Iterate through sentences and extract embeddings
        for index, row in data.iterrows():
            if(isinstance(row['filtered_tokens'], str)):
                sentence = row['filtered_tokens']
                daily_return = row['Daily_Return']
                added_date = row['added_date']
                embedding = get_sentence_embedding(sentence)
                significant_embedding = find_significant_features(embedding)
                print(index)

                processed_data.append([added_date, daily_return, significant_embedding])

        # Convert lists to NumPy arrays

        with open('process_7.4_files/list_files/' + stock_name + '_test_june_to_sept_vectorized.pkl', 'wb') as f:
            pickle.dump(processed_data, f)

    create_model_data(stock_name)


#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
def make_prediction(stock_name):
    with open('process_7.4_files/list_files/' + stock_name + '_test_june_to_sept_vectorized.pkl', 'rb') as f:
        loaded_vectorized_test_data = pickle.load(f)

    with open('process_7.4_files/list_files/' + stock_name + '_significant_dimensions.pkl', 'rb') as f:
        significant_dimensions = pickle.load(f)
    # Load the saved model
    loaded_model = LSTMModel(len(significant_dimensions), hidden_size, num_layers)
    loaded_model.load_state_dict(torch.load('process_7.4_files/model_files/'+stock_name+'_lstm_model.pth'))
    loaded_model.eval()

    processed_data = []

    for index, row in enumerate(loaded_vectorized_test_data):
        test_embedding = row[2]
        test_embedding_tensor = torch.from_numpy(test_embedding).float().unsqueeze(0)

        # Make prediction
        predicted_return = loaded_model(test_embedding_tensor)
        predicted_return = predicted_return.item()  # Get the predicted value as a float
        processed_data.append({"added_date": row[0], "daily_return": row[1],
                               "predicted_return": predicted_return})
    df_new = pd.DataFrame(processed_data)
    df_new.to_csv('process_7.4_files/prediction_files/' + stock_name + '_test_june_to_sept_predicted.csv')
#--------------------------------------------------------------------------------------------------------
def prediction_graph(stock_name):

    # Load the CSV file
    df = pd.read_csv('process_7.4_files/prediction_files/' + stock_name + '_test_june_to_sept_predicted.csv')

    # Convert 'added_date' to datetime and extract date only
    df['added_date'] = pd.to_datetime(df['added_date']).dt.date

    # Group by 'added_date' and aggregate
    # grouped_df = df.groupby('added_date').agg({
    #     'daily_return': 'mean',
    #     'predicted_return': 'mean'
    # })

    # Reset index to create a new column for 'added_date'
    grouped_df = df.reset_index()
    # grouped_df = grouped_df[grouped_df['daily_return'] != 0.0].reset_index()
    # Create the plot
    plt.figure(figsize=(40, 24))
    plt.plot(grouped_df['added_date'], grouped_df['daily_return'], label='Daily Return', marker='o')
    plt.plot(grouped_df['added_date'], grouped_df['predicted_return'], label='Predicted Return', marker='x')

    # Set labels and title
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.title(stock_name+' Daily Return vs. Predicted Return')

    # Add legend
    plt.legend()

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Show the plot
    plt.show()
    # Save the aggregated DataFrame to a new CSV file
    # grouped_df.to_csv("process_2_files/"+stock_name+"_test_june_to_sept_predicted_aggregated.csv", index=False)
#--------------------------------------------------------------------------------------------------------


# data_cleaning()
# stock_data_preprocessing("JKH")
# sentiment_vectorization("JKH")
# make_prediction("JKH")
# prediction_graph("JKH")
# business_date_selection_graph("DIAL")

# stock_data_preprocessing("COMB")
# sentiment_vectorization("COMB")
# make_prediction("COMB")
# prediction_graph("COMB")
# business_date_selection_graph("COMB")

# stock_data_preprocessing("HNB")
# sentiment_vectorization("HNB")
# make_prediction("HNB")
# prediction_graph("HNB")
# business_date_selection_graph("COMB")

stock_data_preprocessing("DIAL")
sentiment_vectorization("DIAL")
make_prediction("DIAL")
prediction_graph("DIAL")