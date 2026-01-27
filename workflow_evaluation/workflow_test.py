import re

import joblib
import matplotlib.pyplot as plt
import networkx as nx
import nltk
import numpy as np
import pandas as pd
import torch
from keras.src.layers import BatchNormalization
from nltk import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import QuantileTransformer
from stellargraph import StellarGraph
from stellargraph.layer import GraphSAGE, MeanAggregator
from stellargraph.mapper import GraphSAGENodeGenerator
from tensorflow.keras.models import load_model
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

# Initialize QuantileTransformer
# -----------------------------------------------------------------------------------------
qt = QuantileTransformer(output_distribution='uniform')
# model_name = "ProsusAI/finBERT"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name)

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
token = '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'

fingpt_tokenizer = AutoTokenizer.from_pretrained("cm309/distilroberta-base-finetuned-Financial-News-Superior",
                                                 use_auth_token=token)
fingpt_model = AutoModel.from_pretrained("cm309/distilroberta-base-finetuned-Financial-News-Superior",
                                         use_auth_token=token)

finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone", use_auth_token=token)
finbert_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone", use_auth_token=token)


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

        df = pd.read_csv("../All_news_CSV_files/biz_news_articles_test_backup.csv"
                         , encoding="ISO-8859-1")

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
                processed_data.append(
                    {"added_date": row["added_date"], "filtered_tokens": words, "cleaned_news": sentence})

        processed_df = pd.DataFrame(processed_data)
        processed_df['added_date'] = pd.to_datetime(processed_df['added_date']).dt.date

        # sbert conversion
        processed_df['sbert_vectors'] = processed_df['cleaned_news'].apply(lambda x: sbert_model.encode(x))

        # fingpt coversion
        processed_df['fingpt_vectors'] = processed_df['cleaned_news'].apply(lambda x: get_fingpt_vector(x))

        # finbert sentiment
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
    df_daily = pd.read_pickle("../workflow_files/list_files/document_vectors_test.pkl")
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
    df_daily = df_daily.drop(columns=["Date", "Open", "High", "Low", "Vol."], axis=1)

    # Ensure no missing values
    df_daily.dropna(inplace=True)

    df_daily.to_pickle("workflow_files/list_files/price_attached_vectors_test.pkl")


# -----------------------------------------------------------------------------------------------------------------------

def gnn_creation():
    # Load unseen data
    df_test = pd.read_pickle("../workflow_files/list_files/price_attached_vectors_test.pkl")

    # Use the same scaler from training
    scaler = joblib.load("../workflow_files/model_files/min_max_scaler.pkl")
    # df_test[['price']] = scaler.transform(df_test[['price']])  # Apply same transformation
    # df_test[['return']] = return_scaler.transform(df_test[['return']])  # Apply return scaler

    # Initialize graph
    graph_test = nx.Graph()

    # Add nodes (news articles & stock prices)
    for index, row in df_test.iterrows():
        news_vector = np.concatenate([row['sbert_vectors'], row['fingpt_vectors']])
        graph_test.add_node(f"news_{index}", type="news", features=news_vector)
        graph_test.add_node(f"stock_{row['added_date']}", type="stock", price=row['price'], change=row['return'])

    # Add edges (news-stock relations)
    for index, row in df_test.iterrows():
        stock_node = f"stock_{row['added_date']}"
        if stock_node in graph_test.nodes:
            graph_test.add_edge(f"news_{index}", stock_node, weight=1)

    # Convert to StellarGraph
    stellar_graph_test = StellarGraph.from_networkx(graph_test, node_features="features")

    # Extract stock nodes (for prediction)
    test_nodes_unseen = [n for n in graph_test.nodes if n.startswith("stock_")]
    test_targets_unseen = np.array([graph_test.nodes[node]['price'] for node in test_nodes_unseen])

    # Normalize target values using the same scaler
    test_targets_unseen = scaler.transform(test_targets_unseen.reshape(-1, 1)).flatten()

    # Create GraphSAGE Data Generator for unseen data
    generator_test = GraphSAGENodeGenerator(stellar_graph_test, batch_size=50, num_samples=[10, 10, 5])
    test_gen_unseen = generator_test.flow(test_nodes_unseen, test_targets_unseen, shuffle=False)

    # Load the trained model
    custom_objects = {"GraphSAGE": GraphSAGE, "MeanAggregator": MeanAggregator,
                      "BatchNormalization": BatchNormalization}
    loaded_model = load_model("../workflow_files/model_files/graphsage_stock_model.h5", custom_objects=custom_objects)

    # Predict on unseen test set
    y_pred_unseen = loaded_model.predict(test_gen_unseen)
    y_test_unseen = test_targets_unseen  # True stock prices

    # Convert predictions back to original scale
    y_pred_unseen_original = scaler.inverse_transform(y_pred_unseen.reshape(-1, 1)).flatten()
    y_test_unseen_original = scaler.inverse_transform(y_test_unseen.reshape(-1, 1)).flatten()

    # Compute Evaluation Metrics for Unseen Data
    mse_unseen = mean_squared_error(y_test_unseen_original, y_pred_unseen_original)
    mae_unseen = mean_absolute_error(y_test_unseen_original, y_pred_unseen_original)
    r2_unseen = r2_score(y_test_unseen_original, y_pred_unseen_original)

    print(f"Unseen Data - Mean Squared Error (MSE): {mse_unseen:.4f}")
    print(f"Unseen Data - Mean Absolute Error (MAE): {mae_unseen:.4f}")
    print(f"Unseen Data - R² Score: {r2_unseen:.4f}")

    # Ensure x_values (dates) are sorted
    test_dates_unseen = [node.replace("stock_", "") for node in test_nodes_unseen]
    test_dates_unseen = pd.to_datetime(test_dates_unseen)  # Convert to datetime
    sorted_indices_unseen = np.argsort(test_dates_unseen)  # Sort by date

    x_values_unseen = test_dates_unseen[sorted_indices_unseen]  # Sorted dates
    y_test_unseen_original = y_test_unseen_original[sorted_indices_unseen]
    y_pred_unseen_original = y_pred_unseen_original[sorted_indices_unseen]

    # Plot Actual vs Predicted Prices
    plt.figure(figsize=(20, 10))
    plt.plot(range(len(x_values_unseen)), y_test_unseen_original, marker='o', linestyle='-', color='b', label="Actual")
    plt.plot(range(len(x_values_unseen)), y_pred_unseen_original, marker='s', linestyle='-', color='r',
             label="Predicted")

    # Connect points with dashed lines
    for i in range(len(y_test_unseen_original)):
        plt.plot([i, i], [y_test_unseen_original[i], y_pred_unseen_original[i]], 'k--', alpha=0.5)

    # Set X-axis labels as dates
    plt.xticks(range(len(x_values_unseen)), x_values_unseen.strftime('%Y-%m-%d'), rotation=45, fontsize=12)

    # Labels and title
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.title("Stock Price Prediction on Unseen Data")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()


# -----------------------------------------------------------------------------------------------------------------------
data_preprocessing()
stock_price_data_processing("HNB")
# gnn_creation()
