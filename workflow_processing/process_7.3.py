import csv
import pickle
import re

import joblib
import nltk
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from scipy.stats import pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from torch import nn
from transformers import AutoModel, AutoTokenizer

# Load the pre-trained FinancialBERT model
model_name = "ProsusAI/finBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out1, _ = self.lstm1(x)
        out2, _ = self.lstm2(out1)
        out = self.fc(out2[:, -1, :])  # Use the last output of the LSTM
        return out


# Model parameters
# Assuming embeddings are 2D (num_samples, embedding_dim)
hidden_size = 256  # Adjust as needed
num_layers = 5


def data_cleaning():
    def is_business_day_excluding_thursdays_fridays(date):
        if pd.to_datetime(date).weekday() in [3, 4]:  # Thursday (3) or Friday (4)
            return False
        return bool(len(pd.bdate_range(date, date)))

    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    processed_data = []

    df = pd.read_csv("../All_news_CSV_files/news_articles.csv", encoding="ISO-8859-1")  # Replace with the actual encoding
    # df.to_csv("All_news_CSV_files/news_articles.csv", index=False, encoding='utf-8')

    # df = pd.read_csv("All_news_CSV_files/news_articles.csv")
    for index, row in df.iterrows():

        title = row["title"]
        full_text = row["full_text"]
        if (isinstance(title, str) and isinstance(full_text, str)):
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

    df_grouped.to_csv("process_7.3_files/csv_files/news_articles_processed_concat.csv", index=False)


# --------------------------------------------------------------------------------------------------------
def stock_data_preprocessing(stock_name):
    stock = stock_name

    def calculate_percentage_change():
        data = pd.read_csv('stock_price_files/filled_price_files/' + stock + '_filled_stock_data.csv')
        data["Trade Date"] = data["Trade Date"]
        data['Daily_Return'] = data['Close (Rs.)'].pct_change()
        data['Daily_Return'].fillna(method='ffill', inplace=True)
        # Save the DataFrame to a CSV file
        data.to_csv('stock_price_files/percentage_price_files/' + stock + '_stock_data_with_returns.csv', index=False)

    def fill_missing_dates_values():
        df = pd.read_csv("stock_price_files/" + stock + ".csv")

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

        # Print the DataFrame with filled missing values
        print(df)

        df.to_csv('stock_price_files/filled_price_files/' + stock + '_filled_stock_data.csv', index_label='Trade Date')

    def merge_daily_returns():
        df1 = pd.read_csv("../process_7.3_files/csv_files/news_articles_processed_concat.csv")
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
        merged_df.to_csv('process_7.3_files/csv_files/' + stock_name + '_merged_data_with_daily_return.csv',
                         index=False)

        print("Merged data saved process_7.3_files directory")

    fill_missing_dates_values()
    calculate_percentage_change()
    merge_daily_returns()
    print("Stock Data Preprocessing Complete", stock)


# --------------------------------------------------------------------------------------------------------
def observe_sentiment_correlations(stock_name):
    nltk.download('vader_lexicon')

    def analyze_sentiment(article):
        analyzer = SentimentIntensityAnalyzer()
        sentiment = analyzer.polarity_scores(article)
        blob = TextBlob(article)
        return sentiment, blob.sentiment.polarity

    df = pd.read_csv('process_7.3_files/csv_files/' + stock_name + '_merged_data_with_daily_return.csv')
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
            processed_data.append(
                {"added_date": row["added_date"], "filtered_tokens": article, "daily_return": row['Daily_Return'],
                 "positive sentiment": sentiment.get("pos"),
                 "negative sentiment": sentiment.get("neg"),
                 "neutral sentiment": sentiment.get("neu"), "compound": sentiment.get("compound"),
                 "blob_sentiment": blob_sentiment})
            print(article, sentiment, blob_sentiment)
    processed_df = pd.DataFrame(processed_data)
    # Save the processed DataFrame to a new CSV file
    # processed_df.to_csv("All_news_CSV_files/more_news_articles_processed.csv", index=False)
    processed_df.to_csv('process_2_files/' + stock_name + '_old_news_articles_processed_with_sentiment.csv')


# --------------------------------------------------------------------------------------------------------
def visual_sentiment_correlations(stock_name):
    # Calculate the correlation coefficient
    sentiment = 'neutral sentiment'
    shift = 2
    df = pd.read_csv('process_2_files/' + stock_name + '_old_news_articles_processed_with_sentiment.csv')

    df['daily_return'] = df['daily_return'].shift(shift)
    df = df.iloc[shift:]
    correlation = df[sentiment].corr(df['daily_return'])
    print(f"Correlation coefficient: {correlation}")

    # Perform statistical test (Pearson correlation)
    corr, p_value = pearsonr(df[sentiment], df['daily_return'])
    print(f"Pearson correlation: {corr}")
    print(f"p-value: {p_value}")

    # Create a scatter plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=sentiment, y='daily_return', data=df)
    plt.xlabel(sentiment)
    plt.ylabel("Daily Return")
    plt.title("Correlation between " + sentiment + " and Daily Return")
    plt.show()


# --------------------------------------------------------------------------------------------------------
def observe_tfidf_correlations(stock_name):
    def tokenization():
        df = pd.read_csv("../process_7.3_files/csv_files/news_articles_processed_concat.csv")
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
        processed_df.to_csv("process_7.3_files/csv_files/tokenized_news_articles_processed_concat.csv", index=False)
        # processed_df.to_csv("processed_news_articles.csv", index=False)

    def tfidf_creation():

        documents = []
        all_list = []

        with open('../process_7.3_files/csv_files/tokenized_news_articles_processed_concat.csv', mode='r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)
            # Read each line in the CSV file
            for document in csv_reader:
                # Step 1: Remove single quotes and split the string into a list
                word_list = document[1].replace("'", "").split(", ")
                # Step 2: Join the list into a single sentence
                sentence = ' '.join(word_list)
                # Step 1: Remove the brackets and extra spaces
                formatted_string = sentence.strip("[]").strip()  # Remove the brackets
                # formatted_string = sentence.strip()  # Remove leading/trailing whitespace

                documents.append(formatted_string)
                all_list.append([document[0], formatted_string])
        file.close()

        # Apply TF-IDF
        tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=1024,
            min_df=5,
            max_df=0.8,
            stop_words='english',
            sublinear_tf=True,
            use_idf=True
        )
        tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

        with open('../process_7.3_files/list_files/processed_documents_1K.pkl', 'wb') as f:
            pickle.dump(all_list, f)
        joblib.dump(tfidf_vectorizer, "../process_7.3_files/tf-idf_model_files/tfidf_vectorizer1K.joblib")
        joblib.dump(tfidf_matrix, "../process_7.3_files/tf-idf_model_files/tfidf_matrix1K.joblib")
        print("Model, matrix and list saved successfully.")

    def tfidf_vectorization():

        loaded_vectorizer = joblib.load("../process_7.3_files/tf-idf_model_files/tfidf_vectorizer1K.joblib")
        loaded_matrix = joblib.load("../process_7.3_files/tf-idf_model_files/tfidf_matrix1K.joblib")
        with open('../process_7.3_files/list_files/processed_documents_1K.pkl', 'rb') as f:
            document_list = pickle.load(f)

        print("Model, matrix and list loaded successfully.")
        tfidf_list = []
        for document in document_list:
            # tfidf_list.append({"added_date": document[0], "tfidf_values": loaded_vectorizer.transform([document[1]]).toarray()[0]})
            tfidf_list.append([document[0], loaded_vectorizer.transform([document[1]]).toarray()[0]])
        with open('../process_7.3_files/list_files/tf-idf_documents_1K.pkl', 'wb') as f:
            pickle.dump(tfidf_list, f)
        # processed_df = pd.DataFrame(tfidf_list)
        # processed_df.to_csv("process_7.3_files/csv_files/processed_tf-idf_documents_1K.csv", index=False)

    def merge_tfidf_daily_returns():
        with open('../process_7.3_files/list_files/tf-idf_documents_1K.pkl', 'rb') as f:
            tfidf_list = pickle.load(f)

        df1 = pd.read_csv("stock_price_files/percentage_price_files/" + stock_name + "_stock_data_with_returns.csv")
        # df = pd.read_csv('your_csv_file.csv', parse_dates=['date'])
        df1['Trade Date'] = pd.to_datetime(df1['Trade Date']).dt.date
        # Create a new list to store the combined data
        new_list = []

        # Create DataFrames from lists
        df2 = pd.DataFrame(tfidf_list, columns=['dates', 'vector'])
        df2['dates'] = pd.to_datetime(df2['dates']).dt.date
        # Merge DataFrames based on key columns
        merged_df = pd.merge(df2, df1, left_on='dates', right_on='Trade Date',
                             how='inner')
        merged_df = merged_df.dropna(subset=['Close (Rs.)'])
        merged_df = merged_df[merged_df['Close (Rs.)'].notna()]

        # Extract daily returns
        daily_returns = merged_df['Daily_Return']

        # Create an empty DataFrame to store correlations
        correlation_df = pd.DataFrame(index=range(1024), columns=['Correlation'])

        # Iterate through each element index
        for i in range(1024):
            # Extract the i-th element from each row's array
            element_series = merged_df['vector'].apply(lambda x: x[i])

            # Calculate correlation with daily returns
            correlation, _ = pearsonr(element_series, daily_returns)
            correlation_df.loc[i, 'Correlation'] = correlation

        correlated_values_df = correlation_df[correlation_df['Correlation'].abs() > 0.7]
        print(correlated_values_df)

    # tokenization()
    # tfidf_creation()
    # tfidf_vectorization()
    merge_tfidf_daily_returns()


# --------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# stock_data_preprocessing("BIL")
# sentiment_vectorization("BIL")
# create_model("BIL")
#
# stock_data_preprocessing("CSF")
# sentiment_vectorization("CSF")
# create_model("CSF")

# stock_data_preprocessing("COOP")
# sentiment_vectorization("COOP")
# create_model("COOP")

# stock_data_preprocessing("VPEL")
# sentiment_vectorization("VPEL")
# create_model("VPEL")
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
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# data_cleaning()
# stock_data_preprocessing("JKH")
# observe_sentiment_correlations("JKH")
# visual_sentiment_correlations("JKH")
observe_tfidf_correlations("JKH")

# sentiment_vectorization("JKH")
# create_model("JKH")
#
# stock_data_preprocessing("JKH")
# sentiment_vectorization("JKH")
# create_model("JKH")
#
#
# stock_data_preprocessing("COMB")
# sentiment_vectorization("COMB")
# create_model("COMB")
#
#
# stock_data_preprocessing("HNB")
# sentiment_vectorization("HNB")
# create_model("HNB")
