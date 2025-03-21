import re

import joblib
import nltk
import numpy as np
import pandas as pd
import torch
# from gensim.models.doc2vec import Doc2Vec, TaggedDocument
# from keras import Input
# from keras.src.callbacks import ReduceLROnPlateau, EarlyStopping
# from keras.src.layers import Conv1D, Flatten, LeakyReLU, BatchNormalization, Dropout
# from keras.src.optimizers import RMSprop, Adam, SGD
# from keras.src.regularizers import l2
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
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import keras_tuner as kt
from tensorflow.keras.losses import Huber


# from tensorflow.keras.optimizers import AdamW
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from stellargraph import StellarGraph
from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import chardet

# Initialize QuantileTransformer
# -----------------------------------------------------------------------------------------
qt = QuantileTransformer(output_distribution='uniform')
# model_name = "ProsusAI/finBERT"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name)

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
token = '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'

fingpt_tokenizer = AutoTokenizer.from_pretrained("cm309/distilroberta-base-finetuned-Financial-News-Superior", use_auth_token=token)
fingpt_model = AutoModel.from_pretrained("cm309/distilroberta-base-finetuned-Financial-News-Superior",use_auth_token=token)


finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone",use_auth_token=token)
finbert_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone",use_auth_token=token)



def data_preprocessing():

    def get_finbert_sentiment(text):
        inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = finbert_model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return scores.numpy().flatten()

    def get_fingpt_vector(text):
        inputs = fingpt_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = fingpt_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

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
        joblib.dump(tfidf_vectorizer, "workflow_files/model_files/tfidf_vectorizer.pkl")
        joblib.dump(tfidf_matrix, "workflow_files/model_files/tfidf_matrix.pkl")

        tfidf_vector = [tfidf_vectorizer.transform([row]).toarray()[0] for row in documents]
        dataframe["vector"] = tfidf_vector
        return tfidf_vector, dataframe

    def preprocessing():

        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('vader_lexicon')
        processed_data = []


        df = pd.read_csv("All_news_CSV_files/biz_news_articles_test_backup.csv",
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

        #sbert conversion
        processed_df['sbert_vectors'] = processed_df['cleaned_news'].apply(lambda x: sbert_model.encode(x))

        #fingpt coversion
        processed_df['fingpt_vectors'] = processed_df['cleaned_news'].apply(lambda x: get_fingpt_vector(x))

        #finbert sentiment
        processed_df[['neg_sent', 'neu_sent', 'pos_sent']] = processed_df['cleaned_news'].apply(
            lambda x: pd.Series(get_finbert_sentiment(x)))

        df_daily = processed_df.groupby('added_date').agg({
            'sbert_vectors': 'mean',
            'fingpt_vectors': 'mean',
            'neg_sent': 'mean',
            'neu_sent': 'mean',
            'pos_sent': 'mean'
        }).reset_index()

        df_daily.to_pickle("workflow_files/list_files/document_vectors_test.pkl")


    preprocessing()


# -----------------------------------------------------------------------------------------------------------------------

def stock_price_data_processing(stock_name):

    # df_daily = pd.read_pickle("workflow_files/list_files/document_vectors.pkl")
    df_daily = pd.read_pickle("workflow_files/list_files/document_vectors_test.pkl")
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
    df_daily = df_daily.drop(columns=["Date", "Open","High","Low","Vol."],axis=1)

    # Ensure no missing values
    df_daily.dropna(inplace=True)

    # df_daily.to_pickle("workflow_files/list_files/price_attached_vectors.pkl")
    df_daily.to_pickle("workflow_files/list_files/price_attached_vectors_test.pkl")


# -----------------------------------------------------------------------------------------------------------------------

def gnn_creation():


    # Load dataset
    df = pd.read_pickle("workflow_files/list_files/price_attached_vectors_test.pkl")

    # Normalize stock price and change
    scaler = StandardScaler()
    df[['price']] = scaler.fit_transform(df[['price']])  # Normalize only price

    return_scaler = StandardScaler()
    df[['return']] = return_scaler.fit_transform(df[['return']])  # Normalize return separately (not used for training)

    # Initialize graph
    graph = nx.Graph()

    # Add nodes (news articles and stock prices)
    for index, row in df.iterrows():
        news_vector = np.concatenate([row['sbert_vectors'], row['fingpt_vectors']])
        graph.add_node(f"news_{index}", type="news", features=news_vector)
        graph.add_node(f"stock_{row['added_date']}", type="stock", price=row['price'], change=row['return'])

    # Convert news vectors into a matrix for similarity computation
    news_vectors = np.stack(
        df.apply(lambda x: np.concatenate([x['sbert_vectors'], x['fingpt_vectors']]), axis=1).tolist())

    # Compute cosine similarity between news articles
    cosine_sim = cosine_similarity(news_vectors)

    # Add news-news edges based on similarity
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if cosine_sim[i, j] > 0.6:  # Similarity threshold
                graph.add_edge(f"news_{i}", f"news_{j}", weight=cosine_sim[i, j])

    # Add edges between news and stock (same date)
    for index, row in df.iterrows():
        stock_node = f"stock_{row['added_date']}"
        if stock_node in graph.nodes:
            graph.add_edge(f"news_{index}", stock_node, weight=1)

    # Add edges between consecutive stock prices (time-based relation)
    unique_dates = sorted(df['added_date'].unique())
    for i in range(len(unique_dates) - 1):
        graph.add_edge(f"stock_{unique_dates[i]}", f"stock_{unique_dates[i + 1]}", weight=1)

    # Convert to StellarGraph
    stellar_graph = StellarGraph.from_networkx(graph, node_features="features")

    # Train-test split: Select only stock nodes for prediction
    stock_nodes = [n for n in graph.nodes if n.startswith("stock_")]
    train_nodes, test_nodes = train_test_split(stock_nodes, test_size=0.2, random_state=42)

    # Extract corresponding target values (stock prices)
    train_targets = np.array([graph.nodes[node]['price'] for node in train_nodes])
    test_targets = np.array([graph.nodes[node]['price'] for node in test_nodes])

    # Create a GraphSAGE data generator
    generator = GraphSAGENodeGenerator(stellar_graph, batch_size=50, num_samples=[10, 10, 5])
    train_gen = generator.flow(train_nodes, train_targets, shuffle=True)
    test_gen = generator.flow(test_nodes, test_targets)

    # Build GraphSAGE Model
    graphsage = GraphSAGE(layer_sizes=[64, 32, 16], generator=generator, bias=True, dropout=0.5)
    x_inp, x_out = graphsage.in_out_tensors()
    predictions = Dense(1, activation='linear')(x_out)
    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(optimizer=Adam(lr=0.005), loss='mse')

    # Train the model
    model.fit(train_gen, epochs=40, verbose=1)

    # Predict stock prices on the test set
    y_pred = model.predict(test_gen)
    y_test = test_targets  # True stock prices

    y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()


    # Compute Evaluation Metrics
    mse = mean_squared_error(y_test_original, y_pred_original)
    mae = mean_absolute_error(y_test_original, y_pred_original)
    r2 = r2_score(y_test_original, y_pred_original)

    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

    # Ensure x_values (dates) is sorted
    test_dates = [node.replace("stock_", "") for node in test_nodes]
    sorted_indices = np.argsort(test_dates)  # Sort by date
    x_values = np.array(test_dates)[sorted_indices]  # Sorted date strings


    x_values = x_values[:100]
    y_test_original = y_test_original[:100]  # Ensure matching length
    y_pred_original = y_pred_original[:100]

    # Plot actual vs. predicted prices
    plt.figure(figsize=(20, 10))
    plt.plot(range(len(x_values)), y_test_original, marker='o', linestyle='-', color='b', label="Actual")
    plt.plot(range(len(x_values)), y_pred_original, marker='s', linestyle='-', color='r', label="Predicted")

    # Connect corresponding points with vertical dashed lines
    for i in range(len(y_test_original)):
        plt.plot([i, i], [y_test_original[i], y_pred_original[i]], 'k--', alpha=0.5)

    # Set X-axis labels as dates
    plt.xticks(range(len(x_values)), x_values, rotation=45, fontsize=12)

    # Labels and title
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.title("Stock Price Prediction Over Time")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()


# -----------------------------------------------------------------------------------------------------------------------
# data_preprocessing()
stock_price_data_processing("BIL")
# gnn_creation()