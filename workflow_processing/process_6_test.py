import pickle
import re

import nltk
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.stats import pearsonr
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
hidden_size = 128
num_layers = 2


def data_cleaning():
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    processed_data = []
    df = pd.read_csv("../All_news_CSV_files/test_june_to_sept.csv")
    for index, row in df.iterrows():
        # Access data from each row
        full_text = row["full_text"]
        title = row["title"]
        if (isinstance(full_text, str) and isinstance(title, str)):
            text = title + full_text
            text = text.lower()

            # 2. Remove punctuation
            text = re.sub(r'[^\w\s]', '', text)

            # 3. Remove numbers
            text = re.sub(r'\d+', '', text)

            # 4. Remove extra whitespace
            text = ' '.join(text.split())

            # 5. Tokenize the text
            words = nltk.word_tokenize(text)

            # 6. Remove stop words
            stop_words = set(stopwords.words('english'))
            words = [word for word in words if not word in stop_words]

            # 7. Stemming (optional)
            # stemmer = PorterStemmer()
            # words = [stemmer.stem(word) for word in words]

            # 8. Lemmatization (recommended)
            lemmatizer = WordNetLemmatizer()
            words = [lemmatizer.lemmatize(word) for word in words]

            # 9. Join words back into a string
            cleaned_text = ' '.join(words)

            processed_data.append({"added_date": row["added_date"], "filtered_tokens": cleaned_text})

    # Create a new DataFrame from the processed data
    processed_df = pd.DataFrame(processed_data)
    # Save the processed DataFrame to a new CSV file
    # processed_df.to_csv("All_news_CSV_files/more_news_articles_processed.csv", index=False)
    processed_df.to_csv("process_6_files/csv_files/test_june_to_sept_processed.csv", index=False)
    # processed_df.to_csv("processed_news_articles.csv", index=False)


# --------------------------------------------------------------------------------------------------------
def stock_data_preprocessing(stock_name):
    stock = stock_name

    def calculate_percentage_change():
        data = pd.read_csv('stock_price_files/filled_price_files/' + stock + '_filled_stock_data.csv')
        data["Trade Date"] = data["Trade Date"]
        data['Daily_Return'] = data['Close (Rs.)'].pct_change()
        # data['Daily_Return'] = data['Close (Rs.)'].diff()
        data['Daily_Return'].fillna(0, inplace=True)
        # Save the DataFrame to a CSV file
        data.to_csv('stock_price_files/percentage_price_files/' + stock + '_stock_data_with_returns.csv', index=False)

    def fill_missing_dates_values():
        # Load your stock price data into a pandas DataFrame
        # df = pd.read_csv("stock_price_files/" + stock + "2.csv")
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

        # Fill missing values using forward fill (ffill)
        # df = df.fillna(method='ffill')

        # Print the DataFrame with filled missing values
        print(df)

        # Save the DataFrame to a CSV file
        # Save the DataFrame to a CSV file
        df.to_csv('stock_price_files/filled_price_files/' + stock + '_filled_stock_data.csv', index_label='Trade Date')

    def merge_daily_returns():
        df1 = pd.read_csv("../process_6_files/csv_files/test_june_to_sept_processed.csv")
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

        # Save the merged DataFrame to a new CSV file
        merged_df.to_csv('process_6_files/csv_files/' + stock_name + '_merged_data_with_daily_return.csv',
                         index=False)
        print("Merged data saved process_6_files directory")

    fill_missing_dates_values()
    calculate_percentage_change()
    merge_daily_returns()
    print("Stock Data Preprocessing Complete", stock)


# --------------------------------------------------------------------------------------------------------
def sentiment_vectorization(stock_name):
    def get_sentence_embedding(sentence):
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        sentence_vector = outputs.last_hidden_state[0, 0, :].detach().numpy()
        return sentence_vector.reshape(1, -1)

    def create_model_data(stock_name):

        def find_significant_features(embedding):
            with open('process_6_files/list_files/' + stock_name + '_significant_dimensions.pkl', 'rb') as f:
                significant_dimensions = pickle.load(f)
            return embedding[:, significant_dimensions]

        # Load data from CSV
        data = pd.read_csv('process_6_files/csv_files/' + stock_name + '_merged_data_with_daily_return.csv')

        # Create lists to store embeddings and daily returns
        processed_data = []
        # Iterate through sentences and extract embeddings
        for index, row in data.iterrows():
            if (isinstance(row['filtered_tokens'], str)):
                sentence = row['filtered_tokens']
                daily_return = row['Daily_Return']
                added_date = row['added_date']
                embedding = get_sentence_embedding(sentence)
                significant_embedding = find_significant_features(embedding)
                print(index)

                processed_data.append([added_date, daily_return, significant_embedding])

        # Convert lists to NumPy arrays

        with open('process_6_files/list_files/' + stock_name + '_test_june_to_sept_vectorized.pkl', 'wb') as f:
            pickle.dump(processed_data, f)

    create_model_data(stock_name)


# --------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------
def make_prediction(stock_name):
    with open('process_6_files/list_files/' + stock_name + '_test_june_to_sept_vectorized.pkl', 'rb') as f:
        loaded_vectorized_test_data = pickle.load(f)

    with open('process_6_files/list_files/' + stock_name + '_significant_dimensions.pkl', 'rb') as f:
        significant_dimensions = pickle.load(f)
    # Load the saved model
    loaded_model = LSTMModel(len(significant_dimensions), hidden_size, num_layers)
    loaded_model.load_state_dict(torch.load('process_6_files/model_files/' + stock_name + '_lstm_model.pth'))
    loaded_model.eval()

    processed_data = []

    for index, row in enumerate(loaded_vectorized_test_data):
        test_embedding = row[2]
        if (row[1] > 0):
            test_embedding_tensor = torch.from_numpy(test_embedding).float().unsqueeze(0)

            # Make prediction
            predicted_return = loaded_model(test_embedding_tensor)
            predicted_return = predicted_return.item()  # Get the predicted value as a float
            processed_data.append({"added_date": row[0], "daily_return": row[1],
                                   "predicted_return": predicted_return})
    df_new = pd.DataFrame(processed_data)
    df_new.to_csv('process_6_files/prediction_files/' + stock_name + '_test_june_to_sept_predicted.csv')


# --------------------------------------------------------------------------------------------------------
def prediction_graph(stock_name):
    # Load the CSV file
    df = pd.read_csv('process_6_files/prediction_files/' + stock_name + '_test_june_to_sept_predicted.csv')

    # Convert 'added_date' to datetime and extract date only
    df['added_date'] = pd.to_datetime(df['added_date']).dt.date

    # Group by 'added_date' and aggregate
    # grouped_df = df.groupby('added_date').agg({
    #     'daily_return': 'mean',
    #     'predicted_return': 'max'
    # })
    grouped_df = df
    # Reset index to create a new column for 'added_date'
    grouped_df = grouped_df.reset_index()
    # grouped_df = grouped_df[grouped_df['daily_return'] != 0.0].reset_index()
    # Create the plot
    plt.figure(figsize=(40, 24))
    plt.plot(grouped_df['added_date'], grouped_df['daily_return'], label='Daily Return', marker='o')
    plt.plot(grouped_df['added_date'], grouped_df['predicted_return'], label='Predicted Return', marker='x')

    # Set labels and title
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.title(stock_name + ' Daily Return vs. Predicted Return')

    # Add legend
    plt.legend()

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Show the plot
    plt.show()
    # Save the aggregated DataFrame to a new CSV file
    # grouped_df.to_csv("process_2_files/"+stock_name+"_test_june_to_sept_predicted_aggregated.csv", index=False)


# --------------------------------------------------------------------------------------------------------
def business_date_selection_graph(stock_name):
    # Load the CSV file
    df = pd.read_csv('process_6_files/prediction_files/' + stock_name + '_test_june_to_sept_predicted.csv')

    # Convert 'added_date' to datetime
    df['added_date'] = pd.to_datetime(df['added_date'])

    # Filter for business days
    df = df[pd.to_datetime(df['added_date']).dt.dayofweek < 5]  # 0: Monday, 6: Sunday

    # Save the filtered DataFrame to a new CSV file
    df.to_csv('process_6_files/csv_files/' + stock_name + '_test_june_to_sept_predicted_business_dates.csv',
              index=False)

    # Convert 'added_date' to datetime and extract date only
    df['added_date'] = pd.to_datetime(df['added_date']).dt.date

    # Group by 'added_date' and aggregate
    grouped_df = df.groupby('added_date').agg({
        'daily_return': 'mean',
        'predicted_return': 'sum'
    })

    # Reset index to create a new column for 'added_date'
    # grouped_df = grouped_df[grouped_df['daily_return'] != 0.0].reset_index()
    grouped_df = grouped_df.reset_index()

    # Create the plot
    plt.figure(figsize=(40, 24))
    plt.plot(grouped_df['added_date'], grouped_df['daily_return'], label='Daily Return', marker='o')
    plt.plot(grouped_df['added_date'], grouped_df['predicted_return'], label='Predicted Return', marker='x')

    # Set labels and title
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.title(stock_name + ' Daily Return vs. Predicted Return')

    # Add legend
    plt.legend()

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Show the plot
    plt.show()


def correlation_graph(stock_name):
    # Load the CSV file
    df = pd.read_csv('process_2_files/' + stock_name + '_test_june_to_sept_predicted_business_dates.csv')
    # Convert 'added_date' to datetime and extract date only
    df['added_date'] = pd.to_datetime(df['added_date']).dt.date

    # Group by 'added_date' and aggregate
    grouped_df = df.groupby('added_date').agg({
        'daily_return': 'mean',
        'predicted_return': 'sum'
    })

    # Reset index to create a new column for 'added_date'
    grouped_df = grouped_df.reset_index()

    # Calculate Pearson correlation coefficient
    correlation, _ = pearsonr(grouped_df['daily_return'], grouped_df['predicted_return'])
    print(f"Pearson Correlation Coefficient: {correlation}")

    # Create the scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(grouped_df['daily_return'], grouped_df['predicted_return'], alpha=0.5)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs. Predicted Values')

    # Add trendline
    plt.plot(np.unique(grouped_df['daily_return']),
             np.poly1d(np.polyfit(grouped_df['daily_return'], grouped_df['predicted_return'], 1))(
                 np.unique(grouped_df['daily_return'])),
             color='red')

    # Show the plot
    plt.show()


# data_cleaning()
# stock_data_preprocessing("DIAL")
# sentiment_vectorization("DIAL")
make_prediction("DIAL")
# prediction_graph("DIAL")
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
