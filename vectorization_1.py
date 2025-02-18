from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
import joblib
import numpy as np
# Sample documents
with open('All_news_CSV_files/news_articles90k_processed.csv', mode='r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)
    documents = []
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
file.close()

# Apply TF-IDF
tfidf_vectorizer = TfidfVectorizer(
    stop_words='english',         # Remove common stop words
    max_df=0.7,                   # Ignore terms that appear in more than 70% of documents
    min_df=2,                     # Ignore terms that appear in fewer than 2 documents
    sublinear_tf=True,            # Use logarithmic scale for term frequency
    norm='l2',                    # Apply L2 normalization to improve comparability
    ngram_range=(1, 2),           # Use unigrams and bigrams
    # max_features=99000             # Limit to 5000 most important features
    max_features=4800             # Limit to 5000 most important features
)
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

joblib.dump(tfidf_vectorizer, "tf-idf_model_files/tfidf_vectorizer5k_1.joblib")
joblib.dump(tfidf_matrix, "tf-idf_model_files/tfidf_matrix5k_1.joblib")
print("Model and matrix saved successfully.")
