import re

import joblib
import nltk
import numpy as np
import pandas as pd
from keras.src.losses import Huber
from matplotlib import pyplot as plt
from nltk import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import QuantileTransformer
from tensorflow.keras.models import load_model
from textblob import TextBlob

# Initialize QuantileTransformer
qt = QuantileTransformer(output_distribution='uniform')


def data_preprocessing():
    def number_to_words(num):

        units = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        teens = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
                 "nineteen"]
        tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
        scales = ["", "thousand", "million", "billion", "trillion"]

        if num == 0:
            return "zero"

        if num < 10:
            return units[num]

        if num < 20:
            return teens[num - 10]

        if num < 100:
            return tens[num // 10] + ("" if num % 10 == 0 else " " + units[num % 10])

        result = ""
        for i, scale in enumerate(scales):
            if num < 1000 ** (i + 1):
                if num >= 1000 ** i:
                    result += f"{number_to_words(num // 1000 ** i)} {scale}"
                    num %= 1000 ** i
                break

        if num > 0:
            result += " " + number_to_words(num)

        return result.strip()

    def replace_numbers_with_words(sentence):

        def replace_func(match):
            return number_to_words(int(match.group(0)))

        return re.sub(r'\b\d+\b', replace_func, sentence)

    def tfidf_calculation(documents, dataframe):
        tfidf_vector = []
        tfidf_vectorizer = joblib.load("../process_15_files/model_files/tfidf_vectorizer.pkl")

        tfidf_vector = [tfidf_vectorizer.transform([row]).toarray()[0] for row in documents]
        dataframe["vector"] = tfidf_vector
        return tfidf_vector, dataframe

    def analyze_sentiment(article):
        analyzer = SentimentIntensityAnalyzer()
        sentiment = analyzer.polarity_scores(article)
        blob = TextBlob(article)

        return sentiment, blob.sentiment.polarity

    def preprocessing():

        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('vader_lexicon')
        processed_data = []

        df = pd.read_csv("../All_news_CSV_files/biz_news_articles_test_backup.csv",
                         encoding="ISO-8859-1")

        for index, row in df.iterrows():

            full_text = row["full_text"]
            if (isinstance(full_text, str)):
                text = full_text
                text = text.lower()

                # text = replace_numbers_with_words(text)

                text = re.sub(r'[^\w\s]', '', text)

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

                stemmer = PorterStemmer()
                words = [stemmer.stem(word) for word in words]

                sentence = ' '.join(words)
                sentiment, blob_sentiment = analyze_sentiment(sentence)
                df["positive sentiment"] = sentiment.get("pos")
                df["negative sentiment"] = sentiment.get("neg")
                df["neutral sentiment"] = sentiment.get("neu")
                df["compound"] = sentiment.get("compound")
                df["blob_sentiment"] = blob_sentiment

                print(words)
                processed_data.append({"added_date": row["added_date"], "filtered_tokens": words,
                                       "positive sentiment": sentiment.get("pos"),
                                       "negative sentiment": sentiment.get("neg"),
                                       "neutral sentiment": sentiment.get("neu"), "compound": sentiment.get("compound"),
                                       "blob sentiment": blob_sentiment})

        processed_df = pd.DataFrame(processed_data)

        processed_df['added_date'] = pd.to_datetime(processed_df['added_date']).dt.date
        # df_grouped = processed_df

        df_grouped = processed_df.groupby('added_date').agg(
            # Aggregate 'filtered_tokens' by joining all lists together
            filtered_tokens=('filtered_tokens', lambda token_lists: sum(token_lists, [])),

            # Calculate mean for sentiment columns
            positive_sentiment=('positive sentiment', 'mean'),
            negative_sentiment=('negative sentiment', 'mean'),
            neutral_sentiment=('neutral sentiment', 'mean'),
            compound=('compound', 'mean'),
            blob_sentiment=('blob sentiment', 'mean')
        ).reset_index()

        tf_idf_list = [' '.join(row) for row in df_grouped['filtered_tokens']]
        tfidf_calculation(tf_idf_list, df_grouped)

        # processed_documents = df_grouped['filtered_tokens']
        # model = Doc2Vec.load("process_15_files/model_files/doc2vec_model.model")

        # df_grouped['doc_vector'] = [model.infer_vector(doc) for doc in processed_documents]
        # df_grouped['vector'] = df_grouped.apply(lambda row: np.concatenate([row['doc_vector'], row['tfidf_vector']]),
        #                                         axis=1)
        # df_grouped.drop(columns=['doc_vector', 'tfidf_vector'])
        df_grouped.to_pickle("process_15_files/list_files/document_vectors_test.pkl")

        print(df_grouped.head())

    preprocessing()


# -----------------------------------------------------------------------------------------------------------------------
def stock_data_preprocessing(stock_name):
    stock = stock_name

    def calculate_daily_return():
        df = pd.read_csv("stock_price_files/" + stock + ".csv")
        df2 = pd.read_csv("stock_price_files/" + stock + ".csv")

        df['Trade Date'] = pd.to_datetime(df['Trade Date'])
        df2['Trade Date'] = pd.to_datetime(df['Trade Date'])

        # Sort by date
        df = df.sort_values('Trade Date')
        df2 = df.sort_values('Trade Date')

        # Calculate daily return
        df['Daily Return'] = df['Close (Rs.)'].diff().round(2)
        df2['Daily Return'] = df['Close (Rs.)']

        # Generate a complete date range
        date_range = pd.date_range(start=df['Trade Date'].min(), end=df['Trade Date'].max())
        date_range = pd.date_range(start=df2['Trade Date'].min(), end=df2['Trade Date'].max())

        # Reindex the DataFrame to include all dates
        df = df.drop_duplicates(subset='Trade Date', keep='first')
        df2 = df2.drop_duplicates(subset='Trade Date', keep='first')
        df = df.set_index('Trade Date').reindex(date_range).reset_index()
        df2 = df2.set_index('Trade Date').reindex(date_range).reset_index()
        df = df.rename(columns={'index': 'Trade Date'})
        df2 = df2.rename(columns={'index': 'Trade Date'})
        # ----------------------------------------------------------------------------------------------------------------
        # Moving Averages (SMA and EMA)
        df['SMA_10'] = df['Close (Rs.)'].rolling(window=10).mean()  #
        df2['SMA_10'] = df['SMA_10']
        df['EMA_10'] = df['Close (Rs.)'].ewm(span=10, adjust=False).mean()  #
        df2['EMA_10'] = df['EMA_10']

        # MACD (difference between two EMAs) and Signal Line
        df['EMA_12'] = df['Close (Rs.)'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close (Rs.)'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']  #
        df2['MACD'] = df['MACD']
        df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # RSI (Relative Strength Index)
        delta = df['Close (Rs.)'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=14, min_periods=1).mean()
        avg_loss = pd.Series(loss).rolling(window=14, min_periods=1).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))  #
        df2['RSI'] = df['RSI']

        # ATR (Average True Range)
        df['TR'] = np.maximum(df['High (Rs.)'] - df['Low (Rs.)'],
                              np.maximum(abs(df['High (Rs.)'] - df['Close (Rs.)'].shift(1)),
                                         abs(df['Low (Rs.)'] - df['Close (Rs.)'].shift(1))))
        df['ATR'] = df['TR'].rolling(window=14).mean()  #
        df2['ATR'] = df['ATR']

        # Bollinger Bands
        df['SMA_20'] = df['Close (Rs.)'].rolling(window=20).mean()
        df['STD_20'] = df['Close (Rs.)'].rolling(window=20).std()
        df['Upper Band'] = df['SMA_20'] + (2 * df['STD_20'])
        df['Lower Band'] = df['SMA_20'] - (2 * df['STD_20'])

        # OBV (On-Balance Volume)
        df['OBV'] = (np.sign(df['Close (Rs.)'].diff()) * df['TradeVolume']).fillna(0).cumsum()  #
        df2['OBV'] = df['OBV']

        # CMF (Chaikin Money Flow)
        df['Money Flow Multiplier'] = ((df['Close (Rs.)'] - df['Low (Rs.)']) -
                                       (df['High (Rs.)'] - df['Close (Rs.)'])) / (df['High (Rs.)'] - df['Low (Rs.)'])
        df['Money Flow Volume'] = df['Money Flow Multiplier'] * df['TradeVolume']
        df['CMF'] = df['Money Flow Volume'].rolling(window=20).sum() / df['TradeVolume'].rolling(window=20).sum()  #
        df2['CMF'] = df['CMF']

        # Replace NaN values with 0 for simplicity (optional)
        # df2.fillna(0, inplace=True)

        technical_indicators = df2[['RSI', 'MACD', 'ATR', 'SMA_10', 'EMA_10', 'OBV', 'CMF']]
        df2_transformed = qt.fit_transform(technical_indicators)

        # Assign the transformed values back to the DataFrame
        df2[['RSI_quantile', 'MACD_quantile', 'ATR_quantile', 'SMA_10_quantile', 'EMA_10_quantile', 'OBV_quantile',
             'CMF_quantile']] = df2_transformed

        # ---------------------------------------------------------------------------------------------------------------

        # df = data.dropna(subset=['Close (Rs.)'])
        # df = data[data['Close (Rs.)'].notna()]

        # Backward fill missing values
        df2.fillna(0, inplace=True)
        df2['Daily Return'] = df2['Daily Return'].replace(0, np.nan)
        df2['Daily Return'] = df2['Daily Return'].fillna(method='ffill').fillna(method='bfill')

        # Calculate Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = df2['Daily Return'].quantile(0.25)
        Q3 = df2['Daily Return'].quantile(0.75)

        # Calculate IQR
        IQR = Q3 - Q1

        # Define lower and upper bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filter out outliers
        filtered_df = df2[(df2['Daily Return'] >= lower_bound) & (df2['Daily Return'] <= upper_bound)]

        # Display the final DataFrame
        print(df2)
        filtered_df.to_csv('stock_price_files/processed_price_files/' + stock + '_stock_data_with_returns.csv',
                           index=False)

    def merge_daily_returns():
        df1 = pd.read_pickle("../process_15_files/list_files/document_vectors_test.pkl")

        df2 = pd.read_csv("stock_price_files/processed_price_files/" + stock_name + "_stock_data_with_returns.csv")

        # Convert 'added_date' and 'Trade Date' to datetime objects
        df1['added_date'] = pd.to_datetime(df1['added_date']).dt.date
        df2['Trade Date'] = pd.to_datetime(df2['Trade Date']).dt.date

        # Select desired columns from each DataFrame
        # df1 = df1[['added_date', 'vector']]  # Replace with actual column names
        # df2 = df2[['Trade Date', 'Daily Return']]  # Replace with actual column names

        # Merge DataFrames based on 'added_date' and 'Trade Date'
        merged_df = pd.merge(df1, df2, left_on='added_date', right_on='Trade Date', how='inner')

        # Drop the 'trade_date' column from the merged DataFrame
        merged_df = merged_df.drop('Trade Date', axis=1)
        merged_df = merged_df.drop('filtered_tokens', axis=1)
        merged_df = merged_df.drop('Open (Rs.)', axis=1)
        merged_df = merged_df.drop('High (Rs.)', axis=1)
        merged_df = merged_df.drop('Low (Rs.)', axis=1)
        merged_df = merged_df.drop('TradeVolume', axis=1)
        merged_df = merged_df.drop('ShareVolume', axis=1)
        merged_df = merged_df.drop('Turnover (Rs.)', axis=1)
        merged_df = merged_df.drop('SMA_10', axis=1)
        merged_df = merged_df.drop('EMA_10', axis=1)
        merged_df = merged_df.drop('MACD', axis=1)
        merged_df = merged_df.drop('RSI', axis=1)
        merged_df = merged_df.drop('ATR', axis=1)
        merged_df = merged_df.drop('OBV', axis=1)
        merged_df = merged_df.drop('CMF', axis=1)

        merged_df = merged_df.dropna(subset=['Daily Return'])
        merged_df = merged_df[merged_df['Daily Return'].notna()]
        # Save the merged DataFrame to a new CSV file
        merged_df.to_pickle("process_15_files/list_files/" + stock_name + "_merged_data_with_daily_return_test.pkl")

        print("Merged data saved process_15_files directory")

    calculate_daily_return()
    merge_daily_returns()
    print("Stock Data Preprocessing Complete", stock)


# -----------------------------------------------------------------------------------------------------------------------
def prediction(stock_name):
    df = pd.read_pickle("process_15_files/list_files/" + stock_name + "_merged_data_with_daily_return_test.pkl")

    df["previous_return"] = df["Daily Return"].shift(1)
    df.dropna(inplace=True)

    X = df["previous_return"]
    X = X.to_numpy(dtype=np.float32)

    y = df['Daily Return']  # Target
    y_test = y.to_numpy(dtype=np.float32)

    # y_test = np.log1p(y_test)

    scaler = joblib.load("process_15_files/model_files/" + stock_name + "_scaler.pkl")
    X_test_scaled = scaler.transform(X.reshape(-1, 1))

    loaded_model = load_model("process_15_files/model_files/" + stock_name + "_sequential.h5")

    y_pred_log = loaded_model.predict(X_test_scaled)
    y_pred = np.expm1(y_pred_log)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    huber = Huber(delta=1.0)(y_test, y_pred).numpy()

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Huber Error (HUBER): {huber}")
    print(f"R² Score: {r2}")

    y_pred = y_pred.reshape(-1)
    y_test = y_test.tolist()
    y_pred = y_pred.tolist()
    x_values = range(len(y_test))

    # Plot both price lines
    plt.figure(figsize=(40, 24))
    plt.plot(x_values, y_test, marker='o', linestyle='-', color='b', label="actual")
    plt.plot(x_values, y_pred, marker='s', linestyle='-', color='r', label="predicted")

    # Connect each corresponding data point with a vertical line
    for i in range(len(y_test)):
        plt.plot([x_values[i], x_values[i]], [y_test[i], y_pred[i]], 'k--', alpha=0.5)  # Dashed vertical line

    # Set X-axis labels to dates
    plt.xticks(x_values, y_test)

    # Labels and title
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Price Fluctuation Over Time")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()


# -----------------------------------------------------------------------------------------------------------------------
# data_preprocessing()
# stock_data_preprocessing("HNB")
prediction("HNB")
