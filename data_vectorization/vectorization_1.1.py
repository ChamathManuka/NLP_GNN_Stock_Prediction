import pickle

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
import joblib
import numpy as np
# Sample documents
documents = []
all_list = []
with open('../All_news_CSV_files/news_articles90k_processed.csv', mode='r') as file:
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

with open('../All_news_CSV_files/more_news_articles_processed.csv', mode='r') as file:
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
        all_list.append([document[0],formatted_string])
file.close()

with open('../All_news_CSV_files/processed_documents_9K.pkl', 'wb') as f:
    pickle.dump(all_list, f)

# Apply TF-IDF
tfidf_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=9600,
    min_df=5,
    max_df=0.8,
    stop_words='english',
    sublinear_tf=True,
    use_idf=True
)
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

joblib.dump(tfidf_vectorizer, "tf-idf_model_files/tfidf_vectorizer9K.joblib")
joblib.dump(tfidf_matrix, "tf-idf_model_files/tfidf_matrix9K.joblib")
print("Model and matrix saved successfully.")
