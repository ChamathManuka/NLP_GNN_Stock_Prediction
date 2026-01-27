import re

import joblib
import nltk
import numpy as np
import pandas as pd
import torch
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from nltk import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import QuantileTransformer
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

# Initialize QuantileTransformer
# -----------------------------------------------------------------------------------------
qt = QuantileTransformer(output_distribution='uniform')
# model_name = "ProsusAI/finBERT"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name)



def data_preprocessing():



    def tfidf_calculation(documents, dataframe):

        tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',  # Remove common stop words
            max_df=0.9,  # Ignore terms that appear in more than 70% of documents 0.5
            min_df=5,  # Ignore terms that appear in fewer than 2 documents
            sublinear_tf=True,  # Use logarithmic scale for term frequency
            norm='l2',  # Apply L2 normalization to improve comparability
            ngram_range=(1, 2),  # Use unigrams and bigrams
            # max_features=99000             # Limit to 5000 most important features
            max_features=200  # Limit to 5000 most important features 700
        )
        tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
        joblib.dump(tfidf_vectorizer, "../vectorization_process_files/model_files/tfidf_vectorizer.pkl")
        joblib.dump(tfidf_matrix, "../vectorization_process_files/model_files/tfidf_matrix.pkl")

        tfidf_vector = [tfidf_vectorizer.transform([row]).toarray()[0] for row in documents]
        dataframe["tfidf_vectors"] = tfidf_vector
        return tfidf_vector, dataframe

    def preprocessing():

        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('vader_lexicon')
        processed_data = []


        df = pd.read_csv("../All_news_CSV_files/biz_news_articles.csv",
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

                print(words)
                processed_data.append({"added_date": row["added_date"], "filtered_tokens": words, "cleaned_news": sentence})

        processed_df = pd.DataFrame(processed_data)
        processed_df['added_date'] = pd.to_datetime(processed_df['added_date']).dt.date
        print("Initial processing done")

        tfidf_calculation(processed_df["cleaned_news"].values, processed_df)
        print("tfidf_calculation done")

        processed_documents = processed_df['filtered_tokens']
        tagged_data = [TaggedDocument(words=doc, tags=[str(i)]) for i, doc in enumerate(processed_documents)]

        # Train Doc2Vec model
        model = Doc2Vec(vector_size=200, min_count=2, epochs=40, workers=4)
        model.build_vocab(tagged_data)
        model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
        processed_df['doc2vec_vectors'] = [model.infer_vector(doc) for doc in processed_documents]
        model.save("vectorization_process_files/model_files/doc2vec_model.model")
        print("doc2vec_vectors done")

        df_daily = processed_df.groupby('added_date').agg({
            'doc2vec_vectors': 'mean',
            'tfidf_vectors': 'mean'
        }).reset_index()

        df_daily.to_pickle("vectorization_process_files/list_files/biz_document_vectors_tfdoc.pkl")
        print("concat pickle done")

    preprocessing()

def stock_price_data_processing(stock_name):

    df_daily = pd.read_pickle("../vectorization_process_files/list_files/biz_document_vectors_tfdoc.pkl")
    df_stock = pd.read_csv("stock_price_files/" + stock_name + "_NEW.csv", parse_dates=['Date'])


    df_stock.fillna(0, inplace=True)
    df_stock['price'] = df_stock['price'].replace(0, np.nan)
    df_stock['price'] = df_stock['price'].fillna(method='ffill').fillna(method='bfill')

    df_stock['return'] = df_stock['price'].diff()
    df_stock.dropna(inplace=True)

    Q1 = df_stock['price'].quantile(0.25)
    Q3 = df_stock['price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df_stock[(df_stock['price'] >= lower_bound) & (df_stock['price'] <= upper_bound)]


    # Ensure `df_daily` has a date column named `added_date`
    filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])
    df_daily['added_date'] = pd.to_datetime(df_daily['added_date'])
    df_daily = df_daily.merge(filtered_df, left_on='added_date', right_on='Date', how='left')

    # Drop redundant date column
    df_daily = df_daily.drop(columns=["Date", "Open","High","Low"],axis=1)

    # Ensure no missing values
    df_daily.dropna(inplace=True)

    df_daily.to_pickle("vectorization_process_files/list_files/merged_biz_document_vectors_tfdoc.pkl")

    print("stock price data processing done")

def data_preprocessing_test():



    def tfidf_calculation(documents, dataframe):
        tfidf_vectorizer = joblib.load("../vectorization_process_files/model_files/tfidf_vectorizer.pkl")

        tfidf_vector = [tfidf_vectorizer.transform([row]).toarray()[0] for row in documents]
        dataframe["tfidf_vectors"] = tfidf_vector
        return tfidf_vector, dataframe

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

                print(words)
                processed_data.append({"added_date": row["added_date"], "filtered_tokens": words, "cleaned_news": sentence})

        processed_df = pd.DataFrame(processed_data)
        processed_df['added_date'] = pd.to_datetime(processed_df['added_date']).dt.date
        print("Initial processing done")

        tfidf_calculation(processed_df["cleaned_news"].values, processed_df)
        print("tfidf_calculation done")

        processed_documents = processed_df['filtered_tokens']
        model = Doc2Vec.load("vectorization_process_files/model_files/doc2vec_model.model")

        # Infer vectors for all documents
        processed_df['doc2vec_vectors'] = [model.infer_vector(doc) for doc in processed_documents]
        print("doc2vec_vectors done")

        df_daily = processed_df.groupby('added_date').agg({
            'doc2vec_vectors': 'mean',
            'tfidf_vectors': 'mean'
        }).reset_index()

        df_daily.to_pickle("vectorization_process_files/list_files/biz_document_vectors_tfdoc_test.pkl")
        print("concat pickle done")

    preprocessing()

def stock_price_data_processing_test(stock_name):

    df_daily = pd.read_pickle("../vectorization_process_files/list_files/biz_document_vectors_tfdoc_test.pkl")
    df_stock = pd.read_csv("stock_price_files/" + stock_name + "_NEW.csv", parse_dates=['Date'])


    df_stock.fillna(0, inplace=True)
    df_stock['price'] = df_stock['price'].replace(0, np.nan)
    df_stock['price'] = df_stock['price'].fillna(method='ffill').fillna(method='bfill')

    df_stock['return'] = df_stock['price'].diff()
    df_stock.dropna(inplace=True)

    Q1 = df_stock['price'].quantile(0.25)
    Q3 = df_stock['price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df_stock[(df_stock['price'] >= lower_bound) & (df_stock['price'] <= upper_bound)]


    # Ensure `df_daily` has a date column named `added_date`
    filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])
    df_daily['added_date'] = pd.to_datetime(df_daily['added_date'])
    df_daily = df_daily.merge(filtered_df, left_on='added_date', right_on='Date', how='left')

    # Drop redundant date column
    df_daily = df_daily.drop(columns=["Date", "Open","High","Low"],axis=1)

    # Ensure no missing values
    df_daily.dropna(inplace=True)

    df_daily.to_pickle("vectorization_process_files/list_files/merged_biz_document_vectors_tfdoc_test.pkl")

    print("stock price data processing done")

#train data preparation
# data_preprocessing()
stock_price_data_processing("HNB")

#test data preparation
data_preprocessing_test()
stock_price_data_processing_test("HNB")

