import pickle
import re
import nltk
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from scipy.stats import stats
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from gensim.models import Word2Vec

from transformers import AutoModel, AutoTokenizer
from torch import nn
import torch
from bs4 import BeautifulSoup

# Load the pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


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
input_size = 768  # Dimension of BERT embeddings
hidden_size = 128
num_layers = 2


def data_cleaning():
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    processed_data = []
    df = pd.read_csv("All_news_CSV_files/old_news_articles.csv")

    def is_irrelevant_row(row):
        # Check for common indicators of irrelevant rows
        if re.search(r"^\s*(Normal|false|true|EN-US|X-NONE)\s*$", row, re.IGNORECASE):
            return True

        # Check for excessive whitespace or newlines
        if len(row.strip()) < 5 or row.count("\n") > 5:
            return True

        # Check for specific patterns (adjust as needed)
        if "MicrosoftInternetExplorer4" in row or "st1\:*{behavior:url(#ieooui)}" in row:
            return True
        return False

    for index, row in df.iterrows():
        # Access data from each row
        full_text = row["full_text"]
        title = row["title"]
        if (isinstance(full_text, str) and isinstance(title, str)):
            text = title

            # soup = BeautifulSoup(text, 'html.parser')
            # for element in soup.find_all(attrs=True):  # Find all elements with attributes
            #     for attr in list(element.attrs.keys()):  # Iterate over attributes
            #         del element[attr]
            # text1 = soup.get_text()

            # Regular expression to match and remove unwanted patterns
            # pattern = r"^\s*\/\*.*?\*\/\s*$|^\s*table\..*?\{.*?\}|MicrosoftInternetExplorer4|st1\:*{behavior:url\(#ieooui\)}|body\s*\{.*?\}"

            # Remove unwanted patterns using the regular expression
            # reg_text = re.sub(pattern, "", text, flags=re.DOTALL | re.MULTILINE)

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
    processed_df['added_date'] = pd.to_datetime(processed_df['added_date']).dt.date

    # Group by 'Date' and concatenate 'filtered_tokens'
    df_grouped = processed_df.groupby('added_date')['filtered_tokens'].agg(' '.join).reset_index()
    df_grouped = df_grouped[df_grouped['added_date'].apply(lambda x: bool(len(pd.bdate_range(x, x))))]

    # Save the processed DataFrame to a new CSV file
    # processed_df.to_csv("All_news_CSV_files/more_news_articles_processed.csv", index=False)
    df_grouped.to_csv("process_2_files/old_news_articles_processed.csv", index=False)
    # processed_df.to_csv("processed_news_articles.csv", index=False)
#--------------------------------------------------------------------------------------------------------
def stock_data_preprocessing(stock_name):
    stock = stock_name
    def calculate_percentage_change():
        data = pd.read_csv('stock_price_files/filled_price_files/' + stock + '_filled_stock_data.csv')
        data["Trade Date"] = data["Trade Date"]
        data['Daily_Return'] = data['Close (Rs.)'].pct_change()
        data['Daily_Return'].fillna(0, inplace=True)
        # Save the DataFrame to a CSV file
        data.to_csv('stock_price_files/percentage_price_files/'+stock+'_stock_data_with_returns.csv', index=False)

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
        df = df.fillna(method='ffill')

        # Print the DataFrame with filled missing values
        print(df)

        # Save the DataFrame to a CSV file
        # Save the DataFrame to a CSV file
        df.to_csv('stock_price_files/filled_price_files/' + stock + '_filled_stock_data.csv', index_label='Trade Date')
    def merge_daily_returns():

        df1 = pd.read_csv("process_2_files/old_news_articles_processed.csv")
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
        merged_df.to_csv('process_2_files/' + stock_name + '_merged_data_with_daily_return.csv',
                         index=False)
        print("Merged data saved process_2_files directory")

    fill_missing_dates_values()
    calculate_percentage_change()
    merge_daily_returns()
    print("Stock Data Preprocessing Complete",stock)
#--------------------------------------------------------------------------------------------------------
def calculate_sentiment(stock_name):

    def analyze_sentiment(article):
        analyzer = SentimentIntensityAnalyzer()
        sentiment = analyzer.polarity_scores(article)
        blob = TextBlob(article)

        return sentiment, blob.sentiment.polarity

    def write_sentiment_to_file():
        nltk.download('vader_lexicon')
        processed_data = []
        df = pd.read_csv("process_2_files/old_news_articles_processed.csv")
        for index, row in df.iterrows():

            article = row['filtered_tokens']
            if(isinstance(article, str)):
                sentiment, blob_sentiment = analyze_sentiment(article)
                df["positive sentiment"] = sentiment.get("pos")
                df["negative sentiment"] = sentiment.get("neg")
                df["neutral sentiment"] = sentiment.get("neu")
                df["compound"] = sentiment.get("compound")
                df["blob_sentiment"] = blob_sentiment
                processed_data.append({"added_date": row["added_date"], "filtered_tokens": article, "positive sentiment": sentiment.get("pos"), "negative sentiment": sentiment.get("neg"), "neutral sentiment": sentiment.get("neu"),"compound": sentiment.get("compound"), "blob_sentiment": blob_sentiment})
                print(article, sentiment, blob_sentiment)
        processed_df = pd.DataFrame(processed_data)
        # Save the processed DataFrame to a new CSV file
        # processed_df.to_csv("All_news_CSV_files/more_news_articles_processed.csv", index=False)
        processed_df.to_csv('process_2_files/old_news_articles_processed_with_sentiment.csv')

    def correlation_analysis(stock_name):

        df = pd.read_csv("process_2_files/old_news_articles_processed_with_sentiment.csv")

        # Convert 'added_date' to datetime and extract date only
        df['added_date'] = pd.to_datetime(df['added_date'])
        df['added_date'] = df['added_date'].dt.date

        # Group by 'added_date' and calculate mean for sentiment columns
        grouped_df = df.groupby('added_date')[['positive sentiment', 'negative sentiment',
                                               'neutral sentiment', 'compound', 'blob_sentiment']].mean()

        # Reset index to bring 'added_date' back as a column
        grouped_df = grouped_df.reset_index()

        # Save the grouped DataFrame to a new CSV file
        grouped_df.to_csv('process_2_files/grouped_sentiment_data.csv', index=False)
        print("Grouped sentiment data saved to 'grouped_sentiment_data.csv'")


        # Load sentiment data
        sentiment_df = pd.read_csv("process_2_files/grouped_sentiment_data.csv")
        sentiment_df['added_date'] = pd.to_datetime(sentiment_df['added_date'])

        # Load daily return data
        return_df = pd.read_csv("stock_price_files/percentage_price_files/" + stock_name + "_stock_data_with_returns.csv")
        return_df['Trade Date'] = pd.to_datetime(return_df['Trade Date'])

        # Merge DataFrames based on date, selecting only the 'Daily_Return' column
        merged_df = pd.merge(sentiment_df, return_df[['Trade Date', 'Daily_Return']], left_on='added_date',
                             right_on='Trade Date', how='left')

        # Drop the 'trade_date' column from the merged DataFrame
        merged_df = merged_df.drop('Trade Date', axis=1)

        # Perform t-tests
        columns_to_test = ['positive sentiment', 'negative sentiment', 'neutral sentiment', 'compound',
                           'blob_sentiment']

        for column in columns_to_test:
            t_stat, p_value = stats.ttest_ind(merged_df[column], merged_df['Daily_Return'])
            print(f"T-test for {column} vs. Daily_Return:")
            print(f"T-statistic: {t_stat}")
            print(f"p-value: {p_value}")
            print("-" * 60)


        # Save the merged DataFrame to a new CSV file
        merged_df.to_csv('process_2_files/'+stock_name+'_merged_data_with_daily_return.csv', index=False)
        print("Merged data saved process_2_files directory")

    write_sentiment_to_file()
    correlation_analysis(stock_name)

#--------------------------------------------------------------------------------------------------------------------------------

def merge_with_sentiment(stock_name):
    df1 = pd.read_csv("process_2_files/old_news_articles_processed_with_sentiment.csv")
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
    merged_df.to_csv('process_2_files/' + stock_name + '_sentiment_processing.csv',
                     index=False)
    print("Merged data saved process_2_files directory")





















#--------------------------------------------------------------------------------------------------------
def sentiment_vectorization(stock_name):

    def get_sentence_embedding(sentence):
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.detach().numpy()

    def create_model_data(stock_name):
        # Load data from CSV
        data = pd.read_csv('process_2_files/' + stock_name + '_merged_data_with_daily_return.csv')

        # Create lists to store embeddings and daily returns
        embeddings = []
        daily_returns = []

        # Iterate through sentences and extract embeddings
        for index, row in data.iterrows():
            if(isinstance(row['filtered_tokens'], str)):
                sentence = row['filtered_tokens']
                daily_return = row['Daily_Return']  # Replace with actual column name
                # if pd.isna(daily_return):
                #     daily_return = 0.0
                embedding = get_sentence_embedding(sentence)
                print(index)
                embeddings.append(embedding)
                daily_returns.append(daily_return)

        # Convert lists to NumPy arrays
        embeddings = np.array(embeddings)
        daily_returns = np.array(daily_returns)

        with open('process_2_files/'+stock_name+'_embeddings_bert.pkl', 'wb') as f:
            pickle.dump(embeddings, f)

        with open('process_2_files/'+stock_name+'_daily_returns.pkl', 'wb') as f:
            pickle.dump(daily_returns, f)

        print("list has been created..")

        # Now you can use the trained model to predict daily returns for new sentences.

    create_model_data(stock_name)
    print("Sentiment Vectorization Complete",stock_name)
#--------------------------------------------------------------------------------------------------------
def create_model(stock_name):
    with open('process_2_files/'+stock_name+'_embeddings_bert.pkl', 'rb') as f:
    # with open('process_2_files/HNB_embeddings_bert.pkl', 'rb') as f:
        embeddings = pickle.load(f)

    with open('process_2_files/'+stock_name+'_daily_returns.pkl', 'rb') as f:
        daily_returns = pickle.load(f)

    # Create the LSTM model
    model = LSTMModel(input_size, hidden_size, num_layers)

    # Define loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error loss for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Convert data to PyTorch tensors
    embeddings_tensor = torch.from_numpy(embeddings).float()
    daily_returns_tensor = torch.from_numpy(daily_returns).float().view(-1, 1)

    # Train the model
    num_epochs = 100
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(embeddings_tensor)
        loss = criterion(outputs, daily_returns_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Save the trained model
    torch.save(model.state_dict(), 'process_2_files/'+stock_name+'_lstm_model.pth')

    # --- Load and use the saved model ---
    print("Create Model Complete",stock_name)
#--------------------------------------------------------------------------------------------------------
def make_prediction():
    def get_sentence_embedding(sentence):
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.detach().numpy()
    # Load the saved model
    loaded_model = LSTMModel(input_size, hidden_size, num_layers)
    loaded_model.load_state_dict(torch.load('process_2_files/lstm_model.pth'))
    loaded_model.eval()

    # Prepare test data (replace with your actual test data)
    test_sentences = ["This is a test sentence 1.", "This is another test sentence."]
    test_embeddings = []

    for sentence in test_sentences:
        test_embedding = get_sentence_embedding(sentence)
        test_embeddings.append(test_embedding)

    test_embeddings_tensor = torch.from_numpy(np.array(test_embeddings)).float()

    # Make predictions
    predicted_returns = loaded_model(test_embeddings_tensor)
    print(predicted_returns)

#--------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
data_cleaning()
# stock_data_preprocessing("JKH")
# calculate_sentiment("JKH")
# merge_with_sentiment("JKH")
# sentiment_vectorization("JKH")
# create_model("BIL")
#
# stock_data_preprocessing("CSF")
# sentiment_vectorization("CSF")
# create_model("CSF")
#
# stock_data_preprocessing("JKH")
# sentiment_vectorization("JKH")
# create_model("JKH")
#
# stock_data_preprocessing("COOP")
# sentiment_vectorization("COOP")
# create_model("COOP")
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# stock_data_preprocessing("VPEL")
# sentiment_vectorization("VPEL")
# create_model("VPEL")
# #
# stock_data_preprocessing("DIAL")
# sentiment_vectorization("DIAL")
# create_model("DIAL")
# #
# stock_data_preprocessing("HNBF")
# sentiment_vectorization("HNBF")
# create_model("HNBF")
# #
# stock_data_preprocessing("LCBF")
# sentiment_vectorization("LCBF")
# create_model("LCBF")
#
# stock_data_preprocessing("TJL")
# sentiment_vectorization("TJL")
# create_model("TJL")
#
# stock_data_preprocessing("CITW")
# sentiment_vectorization("CITW")
# create_model("CITW")
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# make_prediction()