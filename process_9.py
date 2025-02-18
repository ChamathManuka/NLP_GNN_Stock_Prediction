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
from pandas.tseries.offsets import BDay
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from scipy.stats import stats, pearsonr
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import keras_tuner as kt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR

from transformers import AutoModel, AutoTokenizer
from torch import nn
import torch
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense




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

def data_preprocessing():

    def cleaning():

        def is_business_day_excluding_thursdays_fridays(date):
            if pd.to_datetime(date).weekday() in [3, 4]:  # Thursday (3) or Friday (4)
                return False
            return bool(len(pd.bdate_range(date, date)))

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

        processed_df.to_csv("process_9_files/csv_files/news_articles_processed.csv", index=False)

    def tokenization():
        df = pd.read_csv("process_9_files/csv_files/news_articles_processed.csv")
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
        processed_df.to_csv("process_9_files/csv_files/tokenized_news_articles_processed.csv", index=False)

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
        data['Daily_Return'] = data['Close (Rs.)'].diff()
        data['Daily_Return'].fillna(method='ffill', inplace=True)
        # Save the DataFrame to a CSV file
        data.to_csv('stock_price_files/percentage_price_files/'+stock+'_stock_data_with_returns.csv', index=False)

    def merge_daily_returns():

        df1 = pd.read_csv("process_9_files/csv_files/tokenized_news_articles_processed.csv")
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
        merged_df.to_csv('process_9_files/csv_files/' + stock_name + '_merged_data_with_daily_return.csv',
                         index=False)

        print("Merged data saved process_9_files directory")

    calculate_daily_return()
    merge_daily_returns()
    print("Stock Data Preprocessing Complete",stock)
#--------------------------------------------------------------------------------------------------------
def tfidf_tokenization(stock_name):

    def load_corpus():
        documents = []
        all_list = []
        with open('process_9_files/csv_files/' + stock_name + '_merged_data_with_daily_return.csv', mode='r') as file:
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

    def tfidf_creation():
        documents, all_list = load_corpus()
        # Apply TF-IDF
        tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            min_df=5,
            max_df=0.8,
            stop_words='english',
            sublinear_tf=True,
            use_idf=True
        )
        tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

        with open('process_9_files/list_files/processed_documents_5K.pkl', 'wb') as f:
            pickle.dump(all_list, f)
        joblib.dump(tfidf_vectorizer, "process_9_files/tf-idf_model_files/tfidf_vectorizer5K.joblib")
        joblib.dump(tfidf_matrix, "process_9_files/tf-idf_model_files/tfidf_matrix5K.joblib")
        print("Model, matrix and list saved successfully.")
    tfidf_creation()
#--------------------------------------------------------------------------------------------------------
def news_clustering(stock_name):
    def load_corpus():
        documents = []
        all_list = []
        with open('process_9_files/csv_files/' + stock_name + '_merged_data_with_daily_return.csv', mode='r') as file:
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
        return documents, all_list

    loaded_vectorizer = joblib.load("process_9_files/tf-idf_model_files/tfidf_vectorizer5K.joblib")
    loaded_matrix = joblib.load("process_9_files/tf-idf_model_files/tfidf_matrix5K.joblib")

    kmeans = KMeans(n_clusters=8, random_state=42)  # Choose the number of clusters
    kmeans.fit(loaded_matrix)

    # 4. Analyze Clusters
    cluster_labels = kmeans.labels_

    documents, all_list = load_corpus()

    clustered_articles = {}
    for i, label in enumerate(cluster_labels):
        if label not in clustered_articles:
            clustered_articles[label] = []
        clustered_articles[label].append(all_list[i])
    with open('process_9_files/list_files/'+stock_name+'_clustered_documents_5K.pkl', 'wb') as f:
        pickle.dump(clustered_articles, f)
    print("Clustered documents saved process_9_files directory")


    def filter_relevant_news():
        filtered_news = []
        with open('process_9_files/list_files/'+stock_name+'_clustered_documents_5K.pkl', 'rb') as f:
            clustered_news = pickle.load(f)

        for key, lst in clustered_news.items():
            if( key != 4 and key != 6 ):  # Assuming 0-based indexing for keys
                filtered_news.extend(lst)

        with open('process_9_files/list_files/'+stock_name+'_filtered_documents_5K.pkl', 'wb') as f:
            pickle.dump(np.array(filtered_news), f)
        print("filtered documents saved process_9_files directory")

    filter_relevant_news()
#--------------------------------------------------------------------------------------------------------
def create_model(stock_name):
    with open('process_9_files/list_files/' + stock_name + '_filtered_documents_5K.pkl', 'rb') as f:
        filtered_news = pickle.load(f)

    def tfidf_creation():
        documents = [item[1] for item in filtered_news]
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


        joblib.dump(tfidf_vectorizer, "process_9_files/tf-idf_model_files/tfidf_vectorizer5K_filtered.joblib")
        joblib.dump(tfidf_matrix, "process_9_files/tf-idf_model_files/tfidf_matrix5K_filtered.joblib")
        print("Model, matrix and list saved successfully.")

    def create_model_data(stock_name):

        loaded_vectorizer = joblib.load("process_9_files/tf-idf_model_files/tfidf_vectorizer5K_filtered.joblib")

        # Load data from CSV
        with open('process_9_files/list_files/' + stock_name + '_filtered_documents_5K.pkl', 'rb') as f:
            filtered_news = pickle.load(f)

        # Create lists to store embeddings and daily returns
        embeddings = []
        daily_returns = []
        all_list = []
        # Iterate through sentences and extract embeddings
        for index, row in enumerate(filtered_news):
            if(isinstance(row[2], str)):
                sentence = row[2]
                daily_return = row[1]  # Replace with actual column name
                vector = loaded_vectorizer.transform([sentence]).toarray()
                print(index)
                embeddings.append(vector)
                daily_returns.append(float(daily_return))
                all_list.append([row[0], vector, float(daily_return)])
        # Convert lists to NumPy arrays
        # df = pd.DataFrame(all_list, columns=['date', 'vector', 'daily_return'])
        # df.to_pickle('process_9_files/list_files/' + stock_name + '_dataframe.pkl')
        with open('process_9_files/list_files/' + stock_name + '_all_list.pkl', 'wb') as f:
            pickle.dump(all_list, f)



    def prebuild_model():
        tfidf_creation()
        create_model_data(stock_name)

    def make_model():
        with open('process_9_files/list_files/' + stock_name + '_all_list.pkl', 'rb') as f:
            all_list = pickle.load(f)
        # df = pd.read_pickle('process_9_files/list_files/' + stock_name + '_dataframe.pkl')
        # df['date'] = pd.to_datetime(df['date'])
        # df['daily_return'] = pd.to_numeric(df['daily_return'], errors='coerce')
        # aggregated_df = df
        # df['vector'] = df['vector'].apply(lambda x: np.array(ast.literal_eval(x), dtype=float))
        # Group by 'Date' and concatenate 'filtered_tokens'

        # aggregated_df = (
        #     df.groupby('date')
        #     .agg({
        #         'vector': lambda vectors: np.mean(np.vstack(vectors), axis=0),  # Average vector
        #         'daily_return': 'mean'  # Average of other_column
        #     })
        #     .reset_index()
        # )
        X = np.array([item[1] for item in all_list])  # Convert list of arrays to 2D NumPy array
        X = X.reshape(-1, 1000)
        y = np.array([item[2] for item in all_list])
        # 2. Normalize the input data (optional, but recommended for neural networks)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # 3. Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 4. Create a neural network model
        model = Sequential()

        # Input layer (5000 features) + hidden layers
        model.add(Dense(256, input_dim=1000, activation='relu'))
        model.add(BatchNormalization())  # Add Batch Normalization
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(32, activation='relu'))

        # Output layer: Single unit for daily return prediction (regression)
        model.add(Dense(1))

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # 5. Train the model
        def scheduler(epoch, lr):
            if epoch % 10 == 0 and epoch > 0:
                lr = lr * 0.1
            return lr

        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

        lr_scheduler = LearningRateScheduler(scheduler)

        # Train with learning rate scheduler
        history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test),
                            callbacks=[lr_scheduler])

        # 6. Evaluate the model
        test_loss = model.evaluate(X_test, y_test)
        print(f'Test Loss: {test_loss}')

        # 7. Make predictions
        predictions = model.predict(X_test)

        # If you want to print a few predictions
        print(predictions[:5])
        model.save('process_9_files/model_files/sequential_model.h5')
        print("Sequential model saved successfully.")

    def make_model_2():

        with open('process_9_files/list_files/' + stock_name + '_embeddings_bert.pkl', 'rb') as f:
            embeddings = pickle.load(f)

        with open('process_9_files/list_files/' + stock_name + '_daily_returns.pkl', 'rb') as f:
            daily_returns = pickle.load(f)

        # 1. Reshape the TF-IDF data
        X = embeddings.reshape(-1, 100)  # Reshape to (119188, 5000)

        # 2. Normalize the input data (optional, but recommended for neural networks)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # 3. Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, daily_returns, test_size=0.2, random_state=42)

        def build_model(hp):
            model = Sequential()
            model.add(Dense(hp.Int('units', min_value=64, max_value=512, step=64), input_dim=5000, activation='relu'))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(1))
            model.compile(
                optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG')),
                loss='mean_squared_error')
            return model

        # Instantiate the tuner
        tuner = kt.Hyperband(build_model, objective='val_loss', max_epochs=10, hyperband_iterations=2,
                             directory='kt_dir')

        # Search for the best hyperparameters
        tuner.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
        model.save('process_9_files/model_files/sequential_model.h5')

    def make_model_3():
        with open('process_9_files/list_files/' + stock_name + '_embeddings_bert.pkl', 'rb') as f:
            embeddings = pickle.load(f)

        with open('process_9_files/list_files/' + stock_name + '_daily_returns.pkl', 'rb') as f:
            daily_returns = pickle.load(f)

        # 1. Reshape the TF-IDF data
        X = embeddings.reshape(-1, 100)  # Reshape to (119188, 5000)
        y = daily_returns

        # 2. Normalize the input data (optional, but recommended for neural networks)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        # Assuming X and y are your data (TF-IDF features and daily returns)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = SVR(kernel='rbf')
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)
        print(predictions[:5])

        # Calculate the Mean Absolute Error (MAE)
        mae = mean_absolute_error(y_test, predictions)
        print(f'Mean Absolute Error (MAE): {mae}')

        # Calculate the Mean Squared Error (MSE)
        mse = mean_squared_error(y_test, predictions)
        print(f'Mean Squared Error (MSE): {mse}')

        # Calculate the Root Mean Squared Error (RMSE)
        rmse = np.sqrt(mse)
        print(f'Root Mean Squared Error (RMSE): {rmse}')

        # Calculate the R-squared (R²)
        r2 = r2_score(y_test, predictions)
        print(f'R-squared (R²): {r2}')

    # prebuild_model()
    make_model()
    # make_model_2()
    # make_model_3()

#--------------------------------------------------------------------------------------------------------

# data_preprocessing()
# stock_data_preprocessing("JKH")
# tfidf_tokenization("JKH")

news_clustering("JKH")
# create_model("JKH")
# tfidf_tokenization("DIAL")
# stock_data_preprocessing("DIAL")
#
# news_clustering("DIAL")
# create_model("DIAL")