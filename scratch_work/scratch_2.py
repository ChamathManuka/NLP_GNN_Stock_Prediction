import csv

import joblib
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.feature_extraction.text import TfidfVectorizer


def vector_creation():
    # Sample documents
    documents = []
    with open('../process_2_files/old_news_articles_processed.csv', mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        # Read each line in the CSV file
        for document in csv_reader:
            if (isinstance(document[1], str)):
                documents.append(document[1])
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

    joblib.dump(tfidf_vectorizer, "../tf-idf_model_files/tfidf_vectorizer1024.joblib")
    joblib.dump(tfidf_matrix, "../tf-idf_model_files/tfidf_matrix1024.joblib")
    print("Model and matrix saved successfully.")


vector_creation()

vectorizer = joblib.load("../tf-idf_model_files/tfidf_vectorizer1024.joblib")
tfidf_matrix = joblib.load("../tf-idf_model_files/tfidf_matrix1024.joblib")

# Load your data (replace with your actual file paths)
news_df = pd.read_csv("../process_2_files/old_news_articles_processed.csv")
return_df = pd.read_csv("../stock_price_files/percentage_price_files/HNB_stock_data_with_returns.csv")

# Convert 'Date' columns to datetime
news_df['added_date'] = pd.to_datetime(news_df['added_date'])
return_df['Trade Date'] = pd.to_datetime(return_df['Trade Date'])

# Merge DataFrames
merged_df = news_df.merge(return_df[['Trade Date', 'Daily_Return']],
                          left_on='added_date',
                          right_on='Trade Date',
                          how='left')

# Create DataFrame from TF-IDF matrix (corrected)
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# Concatenate TF-IDF DataFrame with merged_df
merged_df = pd.concat([merged_df, tfidf_df], axis=1)

# Perform t-tests
for col in tfidf_df.columns:
    # Split data into two groups based on whether Daily_Return is above or below the median
    group1 = merged_df[merged_df['Daily_Return'] > merged_df['Daily_Return'].median()][col]
    group2 = merged_df[merged_df['Daily_Return'] <= merged_df['Daily_Return'].median()][col]

    # Perform independent t-test
    t_stat, p_value = ttest_ind(group1, group2)

    print(f"T-test for {col}:")
    print(f"T-statistic: {t_stat}")
    print(f"p-value: {p_value}")
    print("-" * 20)
