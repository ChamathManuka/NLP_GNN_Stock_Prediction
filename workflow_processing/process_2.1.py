import re

import joblib
import nltk
import pandas as pd
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from scipy.stats import ttest_ind
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from textblob import TextBlob
from torch import nn
from transformers import AutoModel, AutoTokenizer

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
    df = pd.read_csv("../All_news_CSV_files/old_news_articles.csv")
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
    processed_df.to_csv("process_2.1_files/csv_files/old_news_articles_processed.csv", index=False)
    # processed_df.to_csv("processed_news_articles.csv", index=False)


# --------------------------------------------------------------------------------------------------------
def stock_data_preprocessing(stock_name):
    stock = stock_name

    def calculate_percentage_change():
        data = pd.read_csv('stock_price_files/filled_price_files/' + stock + '_filled_stock_data.csv')
        data["Trade Date"] = data["Trade Date"]
        data['Daily_Return'] = data['Close (Rs.)'].pct_change()
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
        df = df.fillna(method='ffill')

        # Print the DataFrame with filled missing values
        print(df)

        # Save the DataFrame to a CSV file
        # Save the DataFrame to a CSV file
        df.to_csv('stock_price_files/filled_price_files/' + stock + '_filled_stock_data.csv', index_label='Trade Date')

    def merge_daily_returns():
        df1 = pd.read_csv("../process_2.1_files/csv_files/old_news_articles_processed.csv")
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
        merged_df.to_csv('process_2.1_files/csv_files/' + stock_name + '_merged_data_with_daily_return.csv',
                         index=False)
        print("Merged data saved process_2.1_files directory")

    fill_missing_dates_values()
    calculate_percentage_change()
    merge_daily_returns()
    print("Stock Data Preprocessing Complete", stock)


# --------------------------------------------------------------------------------------------------------
def calculate_sentiment(stock_name):
    def analyze_sentiment(article):
        analyzer = SentimentIntensityAnalyzer()
        sentiment = analyzer.polarity_scores(article)
        blob = TextBlob(article)

        return sentiment, blob.sentiment.polarity

    def write_sentiment_to_file():
        nltk.download('vader_lexicon')
        processed_data = []
        df = pd.read_csv("../process_2.1_files/csv_files/old_news_articles_processed.csv")
        for index, row in df.iterrows():

            article = row['filtered_tokens']
            if (isinstance(article, str)):
                sentiment, blob_sentiment = analyze_sentiment(article)
                df["positive sentiment"] = sentiment.get("pos")
                df["negative sentiment"] = sentiment.get("neg")
                df["neutral sentiment"] = sentiment.get("neu")
                df["compound"] = sentiment.get("compound")
                df["blob_sentiment"] = blob_sentiment
                processed_data.append({"added_date": row["added_date"], "filtered_tokens": article,
                                       "positive sentiment": sentiment.get("pos"),
                                       "negative sentiment": sentiment.get("neg"),
                                       "neutral sentiment": sentiment.get("neu"), "compound": sentiment.get("compound"),
                                       "blob_sentiment": blob_sentiment})
                print(article, sentiment, blob_sentiment)
        processed_df = pd.DataFrame(processed_data)
        # Save the processed DataFrame to a new CSV file
        # processed_df.to_csv("All_news_CSV_files/more_news_articles_processed.csv", index=False)
        processed_df.to_csv('process_2.1_files/csv_files/old_news_articles_processed_with_sentiment_1.csv')

    def correlation_analysis(stock_name):

        df = pd.read_csv("../process_2.1_files/csv_files/old_news_articles_processed_with_sentiment.csv")

        # Convert 'added_date' to datetime and extract date only
        df['added_date'] = pd.to_datetime(df['added_date'])
        df['added_date'] = df['added_date'].dt.date

        # Group by 'added_date' and calculate mean for sentiment columns
        grouped_df = df.groupby('added_date')[['positive sentiment', 'negative sentiment',
                                               'neutral sentiment', 'compound', 'blob_sentiment']].mean()

        # Reset index to bring 'added_date' back as a column
        grouped_df = grouped_df.reset_index()

        # Save the grouped DataFrame to a new CSV file
        grouped_df.to_csv('process_2.1_files/csv_files/grouped_sentiment_data.csv', index=False)
        print("Grouped sentiment data saved to 'grouped_sentiment_data.csv'")

        # Load sentiment data
        sentiment_df = pd.read_csv("../process_2.1_files/csv_files/grouped_sentiment_data.csv")
        sentiment_df['added_date'] = pd.to_datetime(sentiment_df['added_date'])

        # Load daily return data
        return_df = pd.read_csv(
            "stock_price_files/percentage_price_files/" + stock_name + "_stock_data_with_returns.csv")
        return_df['Trade Date'] = pd.to_datetime(return_df['Trade Date'])

        # Merge DataFrames based on date, selecting only the 'Daily_Return' column
        merged_df = pd.merge(sentiment_df, return_df[['Trade Date', 'Daily_Return']], left_on='added_date',
                             right_on='Trade Date', how='left')

        # Drop the 'trade_date' column from the merged DataFrame
        merged_df = merged_df.drop('Trade Date', axis=1)
        merged_df = merged_df.dropna()
        # Perform t-tests
        columns_to_test = ['positive sentiment', 'negative sentiment', 'neutral sentiment', 'compound',
                           'blob_sentiment']

        for column in columns_to_test:
            t_stat, p_value = ttest_ind(merged_df[column], merged_df['Daily_Return'])
            print(f"T-test for {column} vs. Daily_Return:")
            print(f"T-statistic: {t_stat}")
            print(f"p-value: {p_value}")
            print("-" * 60)

        # Save the merged DataFrame to a new CSV file
        merged_df.to_csv('process_2.1_files/csv_files/' + stock_name + '_merged_data_with_daily_return.csv',
                         index=False)
        print("Merged data saved process_2.1_files directory")

    # write_sentiment_to_file()
    correlation_analysis(stock_name)


# --------------------------------------------------------------------------------------------------------

def sentiment_model_building(stock_name):
    # Load your data (replace with your actual file path)
    data = pd.read_csv('process_2.1_files/csv_files/' + stock_name + '_merged_data_with_daily_return.csv')

    # Define features (sentiments) and target (Daily_Return)
    X = data[["positive sentiment"]]
    # X = data[["positive sentiment", "negative sentiment", "neutral sentiment"]]
    y = data[["Daily_Return"]]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Ridge Regression model
    model = Ridge(alpha=1.0)  # You can adjust the alpha value

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error:", mse)
    print("R-squared:", r2)

    # Get model coefficients
    # coefficients = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_})
    # print(coefficients)
    joblib.dump(model, "../process_2.1_files/model_files/ridge_regression_model.joblib")


# ------------------------------------------------------------------------------------------------------------------------------------------

def prediction_testing(stock_name):
    def analyze_sentiment(article):
        analyzer = SentimentIntensityAnalyzer()
        sentiment = analyzer.polarity_scores(article)
        blob = TextBlob(article)
        return sentiment, blob.sentiment.polarity

    df = pd.read_csv('../process_2.1_files/csv_files/old_news_articles_processed_1.csv')
    nltk.download('vader_lexicon')
    processed_data = []
    for index, row in df.iterrows():

        article = row['filtered_tokens']
        if (isinstance(article, str)):
            sentiment, blob_sentiment = analyze_sentiment(article)
            df["positive sentiment"] = sentiment.get("pos")
            df["negative sentiment"] = sentiment.get("neg")
            df["neutral sentiment"] = sentiment.get("neu")
            df["compound"] = sentiment.get("compound")
            df["blob_sentiment"] = blob_sentiment
            processed_data.append({"added_date": row["added_date"], "filtered_tokens": article,
                                   "positive sentiment": sentiment.get("pos"),
                                   "negative sentiment": sentiment.get("neg"),
                                   "neutral sentiment": sentiment.get("neu"), "compound": sentiment.get("compound"),
                                   "blob_sentiment": blob_sentiment})

    processed_df = pd.DataFrame(processed_data)

    processed_df['added_date'] = pd.to_datetime(processed_df['added_date'])
    processed_df['added_date'] = processed_df['added_date'].dt.date

    # Group by 'added_date' and calculate mean for sentiment columns
    # grouped_df = processed_df.groupby('added_date')[['positive sentiment', 'negative sentiment',
    #                                        'neutral sentiment', 'compound', 'blob_sentiment']].mean()

    # Reset index to bring 'added_date' back as a column
    # grouped_df = grouped_df.reset_index()

    return_df = pd.read_csv("stock_price_files/percentage_price_files/" + stock_name + "_stock_data_with_returns.csv")
    return_df['Trade Date'] = pd.to_datetime(return_df['Trade Date'])
    return_df['Trade Date'] = return_df['Trade Date'].dt.date

    # Merge DataFrames based on date, selecting only the 'Daily_Return' column
    merged_df = pd.merge(processed_df, return_df[['Trade Date', 'Daily_Return']], left_on='added_date',
                         right_on='Trade Date', how='left')

    # Drop the 'trade_date' column from the merged DataFrame
    merged_df = merged_df.drop('Trade Date', axis=1)
    merged_df = merged_df.dropna()

    X = merged_df[["positive sentiment"]]
    # X = merged_df[["positive sentiment", "negative sentiment", "neutral sentiment"]]

    loaded_model = joblib.load("../process_2.1_files/model_files/ridge_regression_model.joblib")
    y_pred = loaded_model.predict(X)
    y_actual = merged_df['Daily_Return']
    mse = mean_squared_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)
    print("Mean Squared Error:", mse)
    print("R-squared:", r2)

    merged_df["Predicted_Return"] = y_pred

    merged_df['added_date'] = pd.to_datetime(merged_df['added_date']).dt.date
    grouped_df = merged_df
    # Group by 'added_date' and aggregate
    # grouped_df = merged_df.groupby('added_date').agg({
    #     'Daily_Return': 'mean',
    #     'Predicted_Return': 'mean'
    # })

    # Reset index to create a new column for 'added_date'
    grouped_df = grouped_df.reset_index()

    # Create the plot
    plt.figure(figsize=(20, 12))
    plt.plot(grouped_df['added_date'], grouped_df['Daily_Return'], label='Daily Return', marker='o')
    plt.plot(grouped_df['added_date'], grouped_df['Predicted_Return'], label='Predicted Return', marker='x')

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


# data_cleaning()
# stock_data_preprocessing("DIAL")
# calculate_sentiment("DIAL")
# sentiment_model_building("DIAL")
prediction_testing("DIAL")
