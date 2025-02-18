import re
from collections import defaultdict
from itertools import chain

import pandas as pd
import nltk
from matplotlib import pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import csv
from tqdm import tqdm
import pickle
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Download NLTK data if not already downloaded
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('wordnet')
# nltk.download('stopwords')


def data_preprocessing():
    # Read the CSV file into a DataFrame
    # df = pd.read_csv("All_news_CSV_files/more_news_articles.csv")
    df = pd.read_csv("All_news_CSV_files/news_articles90k.csv")
    # Create a new DataFrame to store the processed data
    processed_data = []
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    # Iterate through the rows of the DataFrame
    for index, row in df.iterrows():
        # Access data from each row
        full_text = row["full_text"]
        title = row["title"]
        if (isinstance(full_text, str) and isinstance(title, str)):
            paragraph = full_text + title
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
    # processed_df.to_csv("All_news_CSV_files/more_news_articles_processed.csv", index=False)
    processed_df.to_csv("All_news_CSV_files/news_articles90k_processed.csv", index=False)
    # processed_df.to_csv("processed_news_articles.csv", index=False)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def vector_creation():
    # Sample documents
    documents = []
    all_list = []
    with open('All_news_CSV_files/news_articles90k_processed.csv', mode='r') as file:
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

    # with open('All_news_CSV_files/more_news_articles_processed.csv', mode='r') as file:
    #     csv_reader = csv.reader(file)
    #     next(csv_reader)
    #     # Read each line in the CSV file
    #     for document in csv_reader:
    #         # Step 1: Remove single quotes and split the string into a list
    #         word_list = document[1].replace("'", "").split(", ")
    #         # Step 2: Join the list into a single sentence
    #         sentence = ' '.join(word_list)
    #         # Step 1: Remove the brackets and extra spaces
    #         formatted_string = sentence.strip("[]").strip()  # Remove the brackets
    #         # formatted_string = sentence.strip()  # Remove leading/trailing whitespace
    #
    #         documents.append(formatted_string)
    #         all_list.append([document[0], formatted_string])
    # file.close()

    with open('All_news_CSV_files/processed_documents_1K.pkl', 'wb') as f:
        pickle.dump(all_list, f)

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

    joblib.dump(tfidf_vectorizer, "tf-idf_model_files/tfidf_vectorizer1K.joblib")
    joblib.dump(tfidf_matrix, "tf-idf_model_files/tfidf_matrix1K.joblib")
    print("Model and matrix saved successfully.")

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def news_vectorization():

    # loaded_vectorizer = joblib.load("tf-idf_model_files/tfidf_vectorizer99k.joblib")
    loaded_vectorizer = joblib.load("tf-idf_model_files/tfidf_vectorizer1K.joblib")
    # loaded_matrix = joblib.load("tf-idf_model_files/tfidf_matrix99k.joblib")
    loaded_matrix = joblib.load("tf-idf_model_files/tfidf_matrix1K.joblib")
    print("Loaded model and matrix successfully.")

    def represent_data(csvreader, name):
        date_vector_list = []
        count = 1
        # Process each line
        for row in csvreader:
            date = row[0]
            doc1_str = row[1]

            vector1 = loaded_vectorizer.transform([doc1_str]).toarray()[0]

            date_vector = [date, vector1]
            date_vector_list.append(date_vector)
            count = count + 1
            print(count)
        with open('date_feature_list_files/1K_date_vector_list' + name + '.pkl', 'wb') as f:
            pickle.dump(date_vector_list, f)
            print("Date-Feature list saved successfully! " + name)

    def train_news_vectorization():
        with open('All_news_CSV_files/processed_documents_1K.pkl', 'rb') as f:
            loaded_list = pickle.load(f)

        represent_data(loaded_list, '_1.1')


    def test_news_vectorization():
        date = '26'
        with open('All_news_CSV_files/2024_09_'+date+'_test.csv', 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)

            date_vector_list = []
            count = 1
            # Process each line
            for document in csvreader:
                # Step 1: Remove single quotes and split the string into a list
                word_list = document[1].replace("'", "").split(", ")
                # Step 2: Join the list into a single sentence
                sentence = ' '.join(word_list)
                # Step 1: Remove the brackets and extra spaces
                formatted_string = sentence.strip("[]").strip()  # Remove the brackets
                # formatted_string = sentence.strip()  # Remove leading/trailing whitespace
                date = document[0]

                vector1 = loaded_vectorizer.transform([formatted_string]).toarray()[0]

                date_vector = [date, vector1]
                date_vector_list.append(date_vector)
                count = count + 1
                print(count)
            with open('All_news_test_files/1K_2024_09_'+date+'_test.pkl', 'wb') as f:
                pickle.dump(date_vector_list, f)
                print("Date-Feature list saved successfully!")

    # train_news_vectorization()
    test_news_vectorization()

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def news_aggregation():

    def aggregate_daily_news(daily_news_arrays):
        # Concatenate all arrays along the first axis (axis=0)
        concatenated_array = np.mean(np.array(daily_news_arrays), axis=0)

        # Calculate the mean of the concatenated array along axis 0
        # aggregated_array = np.mean(concatenated_array, axis=0)

        # Reshape the aggregated array to match the expected input shape
        aggregated_array = concatenated_array.reshape(1, 1024)

        return aggregated_array

    # Create a dictionary to store dates and their corresponding IDs

    def train_data_aggregation():

        with open('date_feature_list_files/1K_date_vector_list_1.1.pkl', 'rb') as f:
            concatenated_list = pickle.load(f)

        date_list = []
        for date in concatenated_list:
            date_list.append(date)
        dates_dict = defaultdict(list)

        for dates in date_list:
            date = dates[0].split()[0]  # Extract date from the string
            dates_dict[date].append((dates[1]).reshape(-1))

        # Print the resulting dictionary
        aggregated_vectors = []
        for date, ids in dates_dict.items():
            aggregated_vectors.append([date, aggregate_daily_news(np.array(ids))])

        print(aggregated_vectors)
        with open('aggregated_vector_list_files/1K_aggregated_date_vector.pkl', 'wb') as f:
            pickle.dump(aggregated_vectors, f)
            print("list has been saved")

    def test_data_aggregation():
        value = '26'
        with open('All_news_test_files/1K_2024_09_'+value+'_test.pkl', 'rb') as f:
            test_list = pickle.load(f)

        date_list = []
        for date in test_list:
            date_list.append(date)

        dates_dict = defaultdict(list)

        for dates in date_list:
            date = dates[0].split()[0]  # Extract date from the string
            dates_dict[date].append((dates[1]).reshape(-1))

        # Print the resulting dictionary
        aggregated_vectors = []
        for date, ids in dates_dict.items():
            aggregated_vectors.append([date, aggregate_daily_news(np.array(ids))])

        print(aggregated_vectors)
        with open('aggregated_vector_list_files/1K_aggregated_date_vector_2024_9_'+value+'.pkl', 'wb') as f:
            pickle.dump(aggregated_vectors, f)
            print("list has been saved")

    # train_data_aggregation()
    test_data_aggregation()

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def test_similarity():
    with open('aggregated_vector_list_files/1K_aggregated_date_vector.pkl', 'rb') as f:
        loaded_aggregated_list = pickle.load(f)

    with open('aggregated_vector_list_files/1K_aggregated_date_vector_2024_9_26.pkl', 'rb') as f:
        loaded_test_aggregated_list = pickle.load(f)

    def process_csv_and_date_list(csv_file, date_list, stock_name):

        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file)
        df['Date'] = pd.to_datetime(df['Date'])
        # Convert the 'Date' column to datetime format
        image_arrays = []
        price_fluctuations = []
        combined_arrays = []
        # Iterate through the date list and find corresponding price differences
        for date_str in date_list:
            predict_date = date_str[0].split()[0]
            date = pd.to_datetime(predict_date)
            matching_row = df[df['Date'] == date]

            if not matching_row.empty:
                # open = matching_row['Open Price'].values[0]
                # close = matching_row['Close Price'].values[0]
                price_diff = matching_row['Price Difference'].values[0]
            else:
                # open = 0
                # close = 0
                price_diff = 0
            combined_arrays.append([date_str[1], price_diff])
            image_arrays.append(date_str[1])
            price_fluctuations.append(price_diff)

        return combined_arrays, np.array(image_arrays), np.array(price_fluctuations)

    stock_name = 'COMB'
    combined_array, image_arrays, price_fluctuations = process_csv_and_date_list(
        "processed_stock_files/" + stock_name + "2_Processed.csv", loaded_aggregated_list, stock_name)

    price_list = []

    for test_item in loaded_test_aggregated_list:

        for item in combined_array:
            vector1 = test_item[1].reshape(1,-1)
            vector2 = item[0].reshape(1,-1)
            distance = cosine_similarity(vector1, vector2)[0][0]
            # cosine_similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
            # euclidean_similarity = np.linalg.norm(vector1 - vector2)
            # distance = cosine_similarity(item[0],test_item[1])[0][0]
            # print("Cosine Similarity: ",cosine_similarity)
            if (distance > .3):

                print("Price Difference: ", item[1])
                print("Open Price: ", item[1])
                price_list.append(item[1])

    # Calculate summary statistics
    price_list = np.array(price_list)
    price_list_except_zeros = np.array([x for x in price_list if x != 0])
    mean_change = price_list_except_zeros.mean()
    print("mean change: ", mean_change)
    std_dev = price_list_except_zeros.std()

    # Visualize the distribution
    plt.hist(price_list_except_zeros, bins=20)
    # plt.xlabel('Open Price')
    plt.xlabel('Price Difference')
    plt.ylabel('Frequency')
    plt.title('Distribution of Price Changes')
    plt.show()

    # Define risk tolerance (adjust this based on your preference)
    risk_tolerance = 'moderate'  # Options: 'conservative', 'moderate', 'aggressive'

    # Based on your risk tolerance and analysis, select a suitable range

    price_range = (mean_change - 0.5 * std_dev, mean_change + 0.5 * std_dev)
    print("conservative price range: ", price_range)
    # print("conservative price mean: ", mean_change)

    price_range = (mean_change - std_dev, mean_change + std_dev)
    print("moderate price range: ", price_range)

    price_range = (mean_change - 2 * std_dev, mean_change + 2 * std_dev)
    print("aggressive price range: ", price_range)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# data_preprocessing()
# vector_creation()
news_vectorization()
# news_aggregation()
test_similarity()