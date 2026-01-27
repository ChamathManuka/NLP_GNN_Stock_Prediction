import pickle
import re
from html.parser import HTMLParser

import joblib
import nltk
import numpy as np
import pandas as pd
import torch
from nltk import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
token = 'your_token'

fingpt_tokenizer = AutoTokenizer.from_pretrained("cm309/distilroberta-base-finetuned-Financial-News-Superior",
                                                 use_auth_token=token)
fingpt_model = AutoModel.from_pretrained("cm309/distilroberta-base-finetuned-Financial-News-Superior",
                                         use_auth_token=token)

finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone", use_auth_token=token)
finbert_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone", use_auth_token=token)


class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return ''.join(self.fed)


def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


def clean_article_text(full_text):
    # Remove JSON objects (basic pattern)
    full_text = re.sub(r'\{.*?\}', '', full_text)

    # Remove CSS-like style definitions
    full_text = re.sub(r'^\s*\/\*.*?\*\/\s*$|^\s*table\..*?\{.*?\}|body\s*\{.*?\}', '', full_text,
                       flags=re.DOTALL | re.MULTILINE)

    pattern = r"^\s*\/\*.*?\*\/\s*$|^\s*table\..*?\{.*?\}|MicrosoftInternetExplorer4|st1\:*{behavior:url\(#ieooui\)}|body\s*\{.*?\}"
    reg_text = re.sub(pattern, "", full_text, flags=re.DOTALL | re.MULTILINE)

    # Remove HTML/XML tags
    full_text = re.sub(r'<.*?>', '', reg_text)

    # Remove multiple newlines
    full_text = re.sub(r'\n+', '\n', full_text)

    # Remove excessive whitespace
    full_text = ' '.join(full_text.split())

    stripped = strip_tags(full_text)

    return stripped


def clean_text():
    df1 = pd.read_csv('../All_news_CSV_files/news_articles90k.csv')
    df2 = pd.read_csv('../All_news_CSV_files/all_news_2019to2010.csv')

    processed_data = []
    documents = []
    for index, row in df1.iterrows():
        full_text = row["full_text"]

        if (isinstance(full_text, str)):
            clean_article_text(full_text)
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

            processed_data.append({"added_date": row["added_date"], "filtered_tokens": sentence})
            documents.append(sentence)
            print(sentence)

    for index, row in df2.iterrows():
        full_text = row["full_text"]

        if (isinstance(full_text, str)):
            clean_article_text(full_text)
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

            processed_data.append({"added_date": row["added_date"], "filtered_tokens": sentence})
            documents.append(sentence)
            print(sentence)

    processed_df = pd.DataFrame(processed_data)
    processed_df.to_csv('All_news_CSV_files/processed_data.csv', index=False)

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

    kmeans = KMeans(n_clusters=8, random_state=42)  # Choose the number of clusters
    kmeans.fit(tfidf_matrix)

    # 4. Analyze Clusters
    cluster_labels = kmeans.labels_

    clustered_articles = {}
    for i, label in enumerate(cluster_labels):
        if label not in clustered_articles:
            clustered_articles[label] = []
        clustered_articles[label].append(processed_data[i])
    with open('../workflow_files/list_files/clustered_documents.pkl', 'wb') as f:
        pickle.dump(clustered_articles, f)
    print("Clustered documents saved process_9_files directory")


def data_filtering():
    filtered_news = []
    processed_data = []
    with open('../workflow_files/list_files/clustered_documents.pkl', 'rb') as f:
        clustered_news = pickle.load(f)

    for key, lst in clustered_news.items():
        if (key != 1):  # Assuming 0-based indexing for keys
            filtered_news.extend(lst)

    documents = []
    reg1 = 'mirror stock watch brief daili round market activ colombo stock exchang monfri check day top loser gainer fluctuat within market index'
    reg2 = 'normal tablemsonormalt msostylenamet normal msotstylerowbands msotstylecolbands msostylenoshowy msostylepar msopaddingaltin pt pt msoparamarginin msoparamarginbottompt msopaginationwidoworphan fontsizept fontfamilytim'

    # More efficient regex creation (compile once):
    pattern1 = re.compile(r"(?:\b" + r"\b\W*".join(re.escape(word) for word in reg1.split()) + r")+", re.IGNORECASE)
    pattern2 = re.compile(r"(?:\b" + r"\b\W*".join(re.escape(word) for word in reg2.split()) + r")+", re.IGNORECASE)

    for doc in filtered_news:
        sentence = list(doc.items())[1][1]

        # Use re.search() and check for a match:
        if pattern1.search(sentence) or pattern2.search(sentence):
            continue  # Skip if either pattern is found

        processed_data.append({"added_date": list(doc.items())[0][1], "filtered_tokens": list(doc.items())[1][1]})

    processed_df = pd.DataFrame(processed_data)
    processed_df.to_pickle("workflow_files/list_files/processed_documents.pkl")


def vectorizing():
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

    processed_df = pd.read_pickle("../workflow_files/list_files/processed_documents.pkl")
    processed_df['added_date'] = pd.to_datetime(processed_df['added_date'], infer_datetime_format=True, errors='coerce')

    processed_df['added_date'] = processed_df['added_date'].dt.strftime('%Y-%m-%d')
    processed_df['added_date'] = pd.to_datetime(processed_df['added_date']).dt.date

    # sbert conversion
    processed_df['sbert_vectors'] = processed_df['filtered_tokens'].apply(lambda x: sbert_model.encode(x))
    print("sbert vectors saved in workflow directory")

    # fingpt coversion
    processed_df['fingpt_vectors'] = processed_df['filtered_tokens'].apply(lambda x: get_fingpt_vector(x))
    print("fingpt vectors saved in workflow directory")

    # finbert sentiment
    processed_df[['neg_sent', 'neu_sent', 'pos_sent']] = processed_df['filtered_tokens'].apply(
        lambda x: pd.Series(get_finbert_sentiment(x)))
    print("finbert vectors saved in workflow directory")

    processed_df.to_pickle("workflow_files/list_files/document_vectors_not_concat.pkl")

    df_daily = processed_df.groupby('added_date').agg({
        'sbert_vectors': 'mean',
        'fingpt_vectors': 'mean',
        'neg_sent': 'mean',
        'neu_sent': 'mean',
        'pos_sent': 'mean'
    }).reset_index()

    print("daily vectors aggregation done")
    df_daily.to_pickle("workflow_files/list_files/document_vectors_all.pkl")
    print("all vectors saved in workflow directory")


# -----------------------------------------------------------------------------------------------------------------------
def stock_price_data_processing(stock_name):
    df_daily = pd.read_pickle("../workflow_files/list_files/document_vectors_all.pkl")
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

    df_daily.to_pickle("workflow_files/list_files/price_attached_vectors_all.pkl")


# -----------------------------------------------------------------------------------------------------------------------
def test():
    price_scaler = joblib.load("../workflow_files/model_files/min_max_scaler.pkl")
    return_scaler = joblib.load("../workflow_files/model_files/return_scaler.pkl")

    # 2️⃣ Load unseen dataset
    df_test = pd.read_pickle("../workflow_files/list_files/price_attached_vectors_test.pkl")

    # Normalize return values using the same return scaler
    df_test['return_scaled'] = return_scaler.transform(df_test[['return']])

    for index, row in df_test.iterrows():
        news_vector = np.concatenate([
            row['sbert_vectors'],
            row['fingpt_vectors'],
            [row['neu_sent'], row['pos_sent'], row['neg_sent']],
            [row['return_scaled']]
        ])
        print(f"News Node {index} Feature Shape: {news_vector.shape}")  # Should print (1156,)


# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------


# clean_text()
data_filtering()
# vectorizing()
# stock_price_data_processing("HNB")
# test()
