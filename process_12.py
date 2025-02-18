import re

import joblib
import nltk
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from keras import Input
from keras.src.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.src.layers import Conv1D, Flatten, LeakyReLU, BatchNormalization, Dropout
from keras.src.optimizers import RMSprop, Adam, SGD
from keras.src.regularizers import l2
from matplotlib import pyplot as plt
from nltk import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModel
import keras_tuner as kt
from tensorflow.keras.losses import Huber


from tensorflow.keras.optimizers import AdamW
# Initialize QuantileTransformer
# -----------------------------------------------------------------------------------------
qt = QuantileTransformer(output_distribution='uniform')
model_name = "ProsusAI/finBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def data_preprocessing():

    def get_sentence_embedding(sentence):
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        sentence_vector = outputs.last_hidden_state[0, 0, :].detach().numpy()
        return sentence_vector

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

    def analyze_sentiment(article):
        analyzer = SentimentIntensityAnalyzer()
        sentiment = analyzer.polarity_scores(article)
        blob = TextBlob(article)

        return sentiment, blob.sentiment.polarity

    def tfidf_calculation(documents, dataframe):

        tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',  # Remove common stop words
            max_df=0.9,  # Ignore terms that appear in more than 70% of documents 0.5
            min_df=5,  # Ignore terms that appear in fewer than 2 documents
            sublinear_tf=True,  # Use logarithmic scale for term frequency
            norm='l2',  # Apply L2 normalization to improve comparability
            ngram_range=(1, 2),  # Use unigrams and bigrams
            # max_features=99000             # Limit to 5000 most important features
            max_features=490  # Limit to 5000 most important features 700
        )
        tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
        joblib.dump(tfidf_vectorizer, "process_12_files/model_files/tfidf_vectorizer.pkl")
        joblib.dump(tfidf_matrix, "process_12_files/model_files/tfidf_matrix.pkl")

        tfidf_vector = [tfidf_vectorizer.transform([row]).toarray()[0] for row in documents]
        dataframe["vector"] = tfidf_vector
        return tfidf_vector, dataframe

    def preprocessing():

        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('vader_lexicon')
        processed_data = []


        df = pd.read_csv("All_news_CSV_files/biz_news_articles.csv",
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

        processed_documents = df_grouped['filtered_tokens']
        tf_idf_list = [' '.join(row) for row in df_grouped['filtered_tokens']]
        tfidf_calculation(tf_idf_list, df_grouped)
        tagged_data = [TaggedDocument(words=doc, tags=[str(i)]) for i, doc in enumerate(processed_documents)]

        # Train Doc2Vec model
        # model = Doc2Vec(vector_size=1500, min_count=2, epochs=240, workers=6)
        # model.build_vocab(tagged_data)
        # model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

        # Infer vectors for all documents
        # df_grouped['vector'] = [model.infer_vector(doc) for doc in processed_documents]

        # df_grouped['filtered_tokens'] = df_grouped['filtered_tokens'].apply(lambda words: ' '.join(words))
        # processed_documents = df_grouped['filtered_tokens']
        # df_grouped['bert_vector'] = [get_sentence_embedding(doc) for doc in processed_documents]

        # df_grouped['vector'] = df_grouped.apply(lambda row: np.concatenate([ row['tfidf_vector'], row['bert_vector'],row['doc_vector']]), axis=1)
        # df_grouped = df_grouped.drop(columns=['doc_vector', 'tfidf_vector','bert_vector'],axis=1)
        # Save the model and DataFrame
        # model.save("process_12_files/model_files/doc2vec_model.model")
        df_grouped.to_pickle("process_12_files/list_files/document_vectors.pkl")

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

        # -------------------------------------------------------------------------
        # df2['Daily Return'] = qt.fit_transform(df2[['Daily Return']])
        # --------------------------------------------------------------------------
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
        # --------------------------------------------------------------------------
        # Display the final DataFrame
        print(df2)
        filtered_df.to_csv('stock_price_files/processed_price_files/' + stock + '_stock_data_with_returns.csv',
                           index=False)

    def merge_daily_returns():
        df1 = pd.read_pickle("process_12_files/list_files/document_vectors.pkl")

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
        # -----------------------------------------------------------------------------------
        # sentiments = merged_df[['positive_sentiment', 'negative_sentiment', 'neutral_sentiment', 'compound', 'blob_sentiment']]
        # sentiments_transformed = qt.fit_transform(sentiments)
        #
        # # Assign the transformed values back to the DataFrame
        # merged_df[['positive_quantile', 'negative_quantile', 'neutral_quantile', 'compound_quantile', 'blob_quantile']] = sentiments_transformed

        # ------------------------------------------------------------------------------------------------------------------------------
        # Save the merged DataFrame to a new CSV file
        merged_df.to_pickle("process_12_files/list_files/"+stock_name+"_merged_data_with_daily_return.pkl")

        print("Merged data saved process_12_files directory")

    calculate_daily_return()
    merge_daily_returns()
    print("Stock Data Preprocessing Complete", stock)


# -----------------------------------------------------------------------------------------------------------------------
def model_building_1(stock_name):
    df = pd.read_pickle("process_12_files/list_files/"+stock_name+"_merged_data_with_daily_return.pkl")
    # Assuming the 'vector' column contains lists or arrays of 100 elements
    vector_df = pd.DataFrame(df['vector'].tolist(), index=df.index)

    # Combine the expanded vector columns with the rest of the DataFrame
    df = df.drop(columns=['vector']).join(vector_df)
    # Assuming df is the DataFrame with features and 'Daily Return' as the target
    X = df.drop(columns=['added_date', 'Daily Return', 'Close (Rs.)', 'blob_sentiment', 'compound'])  # Features
    # X = vector_df  # Features
    X = X.to_numpy(dtype=np.float32)

    y = df['Daily Return']  # Target
    y = y.to_numpy(dtype=np.float32)

    # Standardize the features (important for neural networks)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build a neural network model with ReLU hidden layers and linear output layer
    model = Sequential([
        Dense(128, activation='relu', input_shape=(210,)),  # Input layer
        Dense(64, activation='relu'),  # Hidden layer
        Dense(32, activation='relu'),  # Hidden layer
        Dense(1, activation='relu')  # Output layer
    ])
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Train the model
    history = model.fit(X_train, y_train, epochs=200, batch_size=64, verbose=1, validation_data=(X_test, y_test))
    model.save("process_12_files/model_files/" + stock_name + "_sequential.h5")
    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    y_pred = y_pred.reshape(-1)
    y_test = y_test[:100].tolist()
    y_pred = y_pred[:100].tolist()
    x_values = range(len(y_test))

    # Plot both price lines
    plt.figure(figsize=(40, 24))
    plt.plot(x_values, y_test, marker='o', linestyle='-', color='b', label="Stock 1")
    plt.plot(x_values, y_pred, marker='s', linestyle='-', color='r', label="Stock 2")

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

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Plot the loss curves
    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()
# -----------------------------------------------------------------------------------------------------------------------

def model_building(stock_name):
    df = pd.read_pickle("process_12_files/list_files/"+stock_name+"_merged_data_with_daily_return.pkl")
    # Assuming the 'vector' column contains lists or arrays of 100 elements
    vector_df = pd.DataFrame(df['vector'].tolist(), index=df.index)

    df['added_date'] = pd.to_datetime(df['added_date'])
    df = df.sort_values(by='added_date')
    train_size = int(len(df) * 0.8)


    df = df.drop(columns=['vector']).join(vector_df)
    # Assuming df is the DataFrame with features and 'Daily Return' as the target

    # df.insert(0, 'previous_return', df["Daily Return"].shift(1))
    # df.dropna(inplace=True)  # Remove first row since it has NaN

    X = df.drop(columns=['added_date', 'positive_sentiment', 'negative_sentiment', 'neutral_sentiment', 'Daily Return', 'Close (Rs.)','RSI_quantile', 'MACD_quantile', 'ATR_quantile', 'SMA_10_quantile', 'EMA_10_quantile', 'OBV_quantile',
    # X = df.drop(columns=['added_date', 'Daily Return', 'Close (Rs.)', 'blob_sentiment', 'compound','positive_sentiment', 'negative_sentiment', 'neutral_sentiment','RSI_quantile', 'MACD_quantile', 'ATR_quantile', 'SMA_10_quantile', 'EMA_10_quantile', 'OBV_quantile',
             'CMF_quantile'])  # Features

    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    X_train = X_train.to_numpy(dtype=np.float32)
    X_test = X_test.to_numpy(dtype=np.float32)


    y = df['Daily Return']  # Target
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    y_train = y_train.to_numpy(dtype=np.float32)
    y_test = y_test.to_numpy(dtype=np.float32)

    # Standardize the features (important for neural networks)

    # Split the data into training and test sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_train = np.log1p(y_train)
    y_test = np.log1p(y_test)

    scaler = StandardScaler()
    X_train_extra = X_train[:, 0:2]
    X_test_extra = X_test[:, 0:2]

    X_train_scaled = scaler.fit_transform(X_train[:, 2:502])
    X_test_scaled = scaler.transform(X_test[:,2 :502])
    joblib.dump(scaler, "process_15_files/model_files/"+stock_name+"_scaler.pkl")
    # pca = PCA(n_components=150)  #
    # X_train_scaled = pca.fit_transform(X_train_scaled)
    # X_test_scaled = pca.transform(X_test_scaled)

    features = 490
    selector = RFE(Ridge(alpha=1.0), n_features_to_select=490)



    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)

    X_train_scaled = np.hstack((X_train_selected, X_train_extra))
    X_test_scaled = np.hstack((X_test_selected, X_test_extra))

    joblib.dump(selector, "process_15_files/model_files/"+stock_name + "_selector.pkl")
    # X_train_extra = X_train_scaled[:, 300:]
    # X_test_extra = X_test_scaled[:, 300:]

    # pca = PCA(n_components=50)  #
    # pca1 = PCA(n_components=300)  #
    # X_train_scaled = pca.fit_transform(X_train_scaled)
    # X_test_scaled = pca.transform(X_test_scaled)

    # X_train_scaled = pca1.fit_transform(X_train_scaled[:, :300])
    # X_test_scaled = pca1.fit_transform(X_test_scaled[:, :300])
    # X_train_scaled = np.hstack((pca.transform(X_train_scaled[:,:300]), X_test_extra))
    # X_test_scaled = np.hstack((pca.transform(X_test_scaled[:,:300]), X_test_extra))

    # print(f"Explained Variance: {sum(pca.explained_variance_ratio_)}")

    def build_model(hp):
        model = Sequential([
            Dense(hp.Int('units_1', min_value=0, max_value=512, step=64),
                  kernel_regularizer=l2(0.01), input_shape=(492,)),#492
            LeakyReLU(alpha=0.1),
            BatchNormalization(),
            Dropout(0.2),

            Dense(hp.Int('units_2', min_value=0, max_value=256, step=64),
                  kernel_regularizer=l2(0.01)),
            LeakyReLU(alpha=0.1),
            BatchNormalization(),
            Dropout(0.2),

            Dense(hp.Int('units_3', min_value=0, max_value=128, step=64),
                  kernel_regularizer=l2(0.01)),
            LeakyReLU(alpha=0.1),
            BatchNormalization(),

            Dense(hp.Int('units_4', min_value=0, max_value=128, step=64),
                  kernel_regularizer=l2(0.01)),
            LeakyReLU(alpha=0.1),
            BatchNormalization(),

            Dense(hp.Int('units_5', min_value=0, max_value=128, step=64),
                  kernel_regularizer=l2(0.01)),
            LeakyReLU(alpha=0.1),
            BatchNormalization(),

            Dense(1, activation='linear')  # Use 'relu' if output must be non-negative
        ])


        optimizer = RMSprop(learning_rate=hp.Float('learning_rate', min_value=1e-6, max_value=1e-3, sampling='log'))
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        return model


    tuner = kt.RandomSearch(build_model, objective='val_loss', max_trials=10, executions_per_trial=2,
                            directory='E:\Academic\master\master\parameter_tuning', project_name='model_tuning_'+stock_name+'_2')
    tuner.search(X_train_scaled, y_train, epochs=40, batch_size=8, validation_data=(X_test_scaled, y_test))

    # Retrieve the best trial (trial with the lowest validation loss or highest accuracy)
    best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]

    # Get the optimal hyperparameters from the best trial
    best_units_1 = best_trial.hyperparameters.get('units_1')
    best_units_2 = best_trial.hyperparameters.get('units_2')
    # best_units_3 = best_trial.hyperparameters.get('units_3')
    # best_units_4 = best_trial.hyperparameters.get('units_4')
    # best_units_5 = best_trial.hyperparameters.get('units_5')
    best_learning_rate = best_trial.hyperparameters.get('learning_rate')

    # Print the optimal hyperparameters
    print(f"Best units_1: {best_units_1}")
    print(f"Best units_2: {best_units_2}")
    # print(f"Best units_3: {best_units_3}")
    # print(f"Best units_4: {best_units_4}")
    # print(f"Best units_5: {best_units_5}")
    print(f"Best learning_rate: {best_learning_rate}")

    print("----------------------------------------------------------------------------------------------------------------------------------")
    # Build a neural network model with ReLU hidden layers and linear output layer
    model = Sequential([ #HNB
        Dense(896, activation='relu', kernel_regularizer=l2(0.01), input_shape=(492,)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.2),

        Dense(448, activation='relu', kernel_regularizer=l2(0.01)),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        Dropout(0.2),

        Dense(192, activation='relu', kernel_regularizer=l2(0.01)),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        Dropout(0.2),


        Dense(1, activation='linear')
    ])

    # model = Sequential([ #JKH
    #     Dense(896, activation='relu', kernel_regularizer=l2(0.01), input_shape=(1202,)),
    #     BatchNormalization(),
    #     LeakyReLU(alpha=0.1),
    #     Dropout(0.2),
    #
    #     Dense(448, activation='relu', kernel_regularizer=l2(0.01)),
    #     LeakyReLU(alpha=0.1),
    #     BatchNormalization(),
    #     Dropout(0.2),
    #
    #     Dense(224, activation='relu', kernel_regularizer=l2(0.01)),
    #     LeakyReLU(alpha=0.1),
    #     BatchNormalization(),
    #     Dropout(0.2),
    #
    #     Dense(112, activation='relu', kernel_regularizer=l2(0.01)),
    #     LeakyReLU(alpha=0.1),
    #     BatchNormalization(),
    #     Dropout(0.2),
    #
    #     Dense(56, activation='relu'),
    #     LeakyReLU(alpha=0.1),
    #     BatchNormalization(),
    #     Dropout(0.2),
    #
    #     Dense(1, activation='linear')
    # ])

    # Compile the model using the optimal learning rate
    # optimizer = Adam(learning_rate=0.0001279)  # Optimal learning rate
    # optimizer = AdamW(learning_rate=0.0002, weight_decay=1e-4)

    # model = Sequential([  # HNB
    #     Dense(192, activation='relu', kernel_regularizer=l2(0.01), input_shape=(492,)),
    #     BatchNormalization(),
    #     LeakyReLU(alpha=0.1),
    #     Dropout(0.2),
    #
    #     Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    #     LeakyReLU(alpha=0.1),
    #     BatchNormalization(),
    #     Dropout(0.2),
    #
    #     Dense(1, activation='linear')
    # ])

    optimizer = SGD(learning_rate=0.0009, momentum=0.9, nesterov=True)
    model.compile(optimizer=optimizer,  loss=Huber(delta=5.0))

    # Train the model

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)

    history = model.fit(X_train_scaled, y_train, epochs=200, batch_size=16, verbose=1,
                        validation_data=(X_test_scaled, y_test), callbacks=[reduce_lr, early_stopping])

    model.save("process_12_files/model_files/" + stock_name + "_sequential.h5")

    y_pred_log = model.predict(X_test_scaled)
    y_pred = np.expm1(y_pred_log)
    y_test = np.expm1(y_test)

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
    y_test = y_test[:100].tolist()
    y_pred = y_pred[:100].tolist()
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

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Plot the loss curves
    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()
# -----------------------------------------------------------------------------------------------------------------------
# data_preprocessing()
stock_data_preprocessing("HNB")
model_building("HNB")
