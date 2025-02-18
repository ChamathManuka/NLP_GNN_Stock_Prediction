import ast
import csv
import pickle
import re

import joblib
import nltk
import numpy as np
import pandas as pd
from keras.src.callbacks import LearningRateScheduler
from keras.src.layers import BatchNormalization, Dropout
from keras.src.optimizers import Adam
from nltk import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from pandas.tseries.offsets import BDay
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from scipy.stats import stats, pearsonr
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import keras_tuner as kt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from textblob import TextBlob

from transformers import AutoModel, AutoTokenizer
from torch import nn
import torch
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns



# Load the pre-trained FinancialBERT model
model_name = "ProsusAI/finBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def data_preprocessing():

    def cleaning():

        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')
        processed_data = []

        df = pd.read_csv("All_news_CSV_files/biz_news_articles.csv",
                         encoding="ISO-8859-1")  # Replace with the actual encoding
        # df.to_csv("All_news_CSV_files/news_articles.csv", index=False, encoding='utf-8')

        # df = pd.read_csv("All_news_CSV_files/news_articles.csv")
        for index, row in df.iterrows():

            title = row["title"]
            full_text = row["full_text"]
            if (isinstance(title, str) and isinstance(full_text, str)):
                text = title +' '+ full_text
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

        # Group by 'Date' and concatenate 'filtered_tokens'
        # df_grouped = processed_df.groupby('added_date')['filtered_tokens'].agg(' '.join).reset_index()
        # df_grouped = df_grouped[df_grouped['added_date'].apply(is_business_day_excluding_thursdays_fridays)]

        processed_df.to_csv("process_10_files/csv_files/biz_news_articles_processed.csv", index=False)

    def tokenization():
        df = pd.read_csv("process_10_files/csv_files/biz_news_articles_processed.csv")
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

        # processed_df['added_date'] = pd.to_datetime(processed_df['added_date']).dt.date
        # df_grouped = processed_df.groupby('added_date')['filtered_tokens'].agg(' '.join).reset_index()

        # Save the processed DataFrame to a new CSV file
        processed_df.to_csv("process_10_files/csv_files/tokenized_biz_news_articles_processed.csv", index=False)

    cleaning()
    tokenization()
#--------------------------------------------------------------------------------------------------------
def stock_data_preprocessing(stock_name):
    stock = stock_name
    def calculate_daily_return():
        data = pd.read_csv("stock_price_files/" + stock + ".csv")
        data['Trade Date'] = pd.to_datetime(data['Trade Date'])
        data["Trade Date"] = data["Trade Date"]
        data = data.dropna(subset=['Close (Rs.)'])
        data = data[data['Close (Rs.)'].notna()]
        data['Daily_Return'] = data['Close (Rs.)'].pct_change()
        data['Daily_Return'].fillna(method='ffill', inplace=True)
        # Save the DataFrame to a CSV file
        data.to_csv('stock_price_files/percentage_price_files/'+stock+'_stock_data_with_returns.csv', index=False)

    def merge_daily_returns():

        df1 = pd.read_csv("process_10_files/csv_files/tokenized_biz_news_articles_processed.csv")
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
        merged_df.to_csv('process_10_files/csv_files/' + stock_name + '_merged_data_with_daily_return.csv',
                         index=False)

        print("Merged data saved process_10_files directory")

    calculate_daily_return()
    merge_daily_returns()
    print("Stock Data Preprocessing Complete",stock)
#--------------------------------------------------------------------------------------------------------
def tfidf_tokenization(stock_name):

    def load_corpus():
        documents = []
        all_list = []
        with open('process_10_files/csv_files/' + stock_name + '_merged_data_with_daily_return.csv', mode='r') as file:
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
                all_list.append([document[2], formatted_string])
        file.close()
        return documents, all_list

    def tfidf_creation(documents, all_list):

        # Apply TF-IDF
        tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=1000,
            min_df=5,
            max_df=0.8,
            stop_words='english',
            sublinear_tf=True,
            use_idf=True
        )
        tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

        with open('process_10_files/list_files/processed_documents_1K.pkl', 'wb') as f:
            pickle.dump(all_list, f)
        joblib.dump(tfidf_vectorizer, "process_10_files/tf-idf_model_files/tfidf_vectorizer1K.joblib")
        joblib.dump(tfidf_matrix, "process_10_files/tf-idf_model_files/tfidf_matrix1K.joblib")
        print("Model, matrix and list saved successfully.")

    documents, all_list = load_corpus()
    tfidf_creation(documents, all_list)

#--------------------------------------------------------------------------------------------------------
def news_clustering(stock_name):
    def news_filtering():
        filtered_news = []
        with open('process_10_files/list_files/' + stock_name + '_clustered_documents_1K.pkl', 'rb') as f:
            clustered_news = pickle.load(f)

        for key, lst in clustered_news.items():
            if (key != 4 and key != 6):  # Assuming 0-based indexing for keys
                filtered_news.extend(lst)

        with open('process_10_files/list_files/' + stock_name + '_filtered_documents_5K.pkl', 'wb') as f:
            pickle.dump(np.array(filtered_news), f)
        print("filtered documents saved process_10_files directory")


    documents = []
    all_list = []
    with open('process_10_files/csv_files/' + stock_name + '_merged_data_with_daily_return.csv', mode='r') as file:
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
            all_list.append([document[0],document[2], formatted_string])
    file.close()

    loaded_vectorizer = joblib.load("process_10_files/tf-idf_model_files/tfidf_vectorizer1K.joblib")
    loaded_matrix = joblib.load("process_10_files/tf-idf_model_files/tfidf_matrix1K.joblib")

    kmeans = KMeans(n_clusters=8, random_state=42)  # Choose the number of clusters
    kmeans.fit(loaded_matrix)

    # 4. Analyze Clusters
    cluster_labels = kmeans.labels_

    clustered_articles = {}
    for i, label in enumerate(cluster_labels):
        if label not in clustered_articles:
            clustered_articles[label] = []
        clustered_articles[label].append(all_list[i])
    with open('process_10_files/list_files/'+stock_name+'_clustered_documents_1K.pkl', 'wb') as f:
        pickle.dump(clustered_articles, f)
    print("Clustered documents saved process_10_files directory")

    # news_filtering()
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
        df = pd.read_csv('process_10_files/csv_files/' + stock_name + '_merged_data_with_daily_return.csv')
        for index, row in df.iterrows():

            # Step 1: Remove single quotes and split the string into a list
            word_list = row['filtered_tokens'].replace("'", "").split(", ")
            # Step 2: Join the list into a single sentence
            sentence = ' '.join(word_list)
            # Step 1: Remove the brackets and extra spaces
            article = sentence.strip("[]").strip()



            if(isinstance(article, str)):
                sentiment, blob_sentiment = analyze_sentiment(article)
                df["positive sentiment"] = sentiment.get("pos")
                df["negative sentiment"] = sentiment.get("neg")
                df["neutral sentiment"] = sentiment.get("neu")
                df["compound"] = sentiment.get("compound")
                df["blob_sentiment"] = blob_sentiment
                processed_data.append({"added_date": row["added_date"], "filtered_tokens": article, "daily_return":row["Daily_Return"], "positive sentiment": sentiment.get("pos"), "negative sentiment": sentiment.get("neg"), "neutral sentiment": sentiment.get("neu"),"compound": sentiment.get("compound"), "blob_sentiment": blob_sentiment})
                print(article, sentiment, blob_sentiment)
        processed_df = pd.DataFrame(processed_data)
        # Save the processed DataFrame to a new CSV file
        # processed_df.to_csv("All_news_CSV_files/more_news_articles_processed.csv", index=False)
        processed_df.to_csv('process_10_files/csv_files/' + stock_name + '_merged_data_with_daily_return_and_sentiment.csv')

    def correlation_analysis(stock_name):

        df = pd.read_csv('process_10_files/csv_files/' + stock_name + '_merged_data_with_daily_return_and_sentiment.csv')

        # Convert 'added_date' to datetime and extract date only
        df['added_date'] = pd.to_datetime(df['added_date'])
        df['added_date'] = df['added_date'].dt.date

        # Group by 'added_date' and calculate mean for sentiment columns
        grouped_df = df.groupby('added_date')[['daily_return','positive sentiment', 'negative sentiment',
                                               'neutral sentiment', 'compound', 'blob_sentiment']].mean()

        # Reset index to bring 'added_date' back as a column
        grouped_df = grouped_df.reset_index()

        # Save the grouped DataFrame to a new CSV file
        grouped_df.to_csv('process_10_files/csv_files/' + stock_name + '_merged_data_with_daily_return_and_sentiment_grouped.csv', index=False)
        print("Grouped sentiment data saved")


        # Load sentiment data
        sentiment_df = pd.read_csv('process_10_files/csv_files/' + stock_name + '_merged_data_with_daily_return_and_sentiment.csv')
        sentiment_df['added_date'] = pd.to_datetime(sentiment_df['added_date'])

        # Perform t-tests
        columns_to_test = ['positive sentiment', 'negative sentiment', 'neutral sentiment', 'compound',
                           'blob_sentiment']

        for column in columns_to_test:
            t_stat, p_value = stats.ttest_ind(sentiment_df[column], sentiment_df['daily_return'])
            print(f"T-test for {column} vs. Daily_Return:")
            print(f"T-statistic: {t_stat}")
            print(f"p-value: {p_value}")
            print("-" * 60)


    write_sentiment_to_file()
    correlation_analysis(stock_name)

#--------------------------------------------------------------------------------------------------------
def correlation_analysis(stock_name):
    # Load the CSV file
    df = pd.read_csv('process_10_files/csv_files/' + stock_name + '_merged_data_with_daily_return_and_sentiment.csv')

    # Display the first few rows to confirm
    print(df.head())
    correlation_type = 'negative sentiment'
    correlation = df['daily_return'].corr(df[correlation_type])
    print(f"Correlation between daily_return and sentiment : {correlation}")

    # # Scatter plot with regression line
    # sns.set(style="whitegrid")
    # plt.figure(figsize=(8, 6))
    # sns.regplot(x=correlation_type, y='daily_return', data=df, scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'})
    # plt.title(f"Correlation: {correlation:.2f}", fontsize=14)
    # plt.xlabel('Sentiment', fontsize=12)
    # plt.ylabel('Daily Return', fontsize=12)
    # plt.show()
    #
    # # Correlation matrix
    # corr_matrix = df[['daily_return', correlation_type]].corr()
    #
    # # Heatmap
    # plt.figure(figsize=(6, 4))
    # sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    # plt.title('Correlation Matrix', fontsize=14)
    # plt.show()
    scaler = StandardScaler()
    df[['daily_return', 'positive sentiment', 'neutral sentiment', 'negative sentiment']] = scaler.fit_transform(df[['daily_return', 'positive sentiment', 'neutral sentiment', 'negative sentiment']])
    correlation_matrix = df[['daily_return', 'positive sentiment', 'neutral sentiment', 'negative sentiment']].corr()

    # Print correlation matrix
    print(correlation_matrix)

    # Visualize correlation matrix using a heatmap
    plt.figure(figsize=(16, 12))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
# data_preprocessing()
# stock_data_preprocessing("HNB")
# tfidf_tokenization("BIL")
# news_clustering("BIL")
stock_data_preprocessing("BIL")
calculate_sentiment("BIL")
correlation_analysis("BIL")