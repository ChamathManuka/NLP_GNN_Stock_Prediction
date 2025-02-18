from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
import joblib
import numpy as np
# Sample documents
with open('processed_testing_news_articles.csv', mode='r') as file:
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
    max_features=100000             # Limit to 5000 most important features
)
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# joblib.dump(tfidf_vectorizer, "tfidf_vectorizer500.joblib")
# joblib.dump(tfidf_matrix, "tfidf_matrix500.joblib")
# print("Model and matrix saved successfully.")

# Perform LSA with TruncatedSVD
# svd = TruncatedSVD(n_components=2)  # n_components = desired number of topics
# lsa_matrix = svd.fit_transform(tfidf_matrix)

# Load the saved model and matrix
loaded_vectorizer = joblib.load("tfidf_vectorizer100k.joblib")
loaded_matrix = joblib.load("tfidf_matrix100k.joblib")

print("Loaded model and matrix successfully.")

# Example new document
doc1_str = ['jeopardi', 'follow', 'anoth', 'delay', 'editionth', 'futur', 'south', 'asian', 'game']
doc2_str = ['belov', 'region', 'event', 'like', 'discontinu', 'edit', 'wa', 'postpon', 'south', 'asian', 'game']
doc3_str = ['popular', 'south', 'asian', 'game', 'sag', 'like', 'banish', 'region', 'calendar', 'schedul', 'event', 'wa', 'put', 'authoritiessubstandard', 'sag', 'face', 'axe', 'year']
doc4_str = ['longstand', 'region', 'competit', 'could', 'discontinu', 'event', 'wa', 'postpon', 'yet', 'againth', 'south', 'asian', 'game']

document1 = ' '.join(doc1_str)
document2 = ' '.join(doc2_str)
document3 = ' '.join(doc3_str)
document4 = ' '.join(doc4_str)


vector1 = loaded_vectorizer.transform([document1]).toarray()[0]
vector2 = loaded_vectorizer.transform([document2]).toarray()[0]
vector3 = loaded_vectorizer.transform([document3]).toarray()[0]
vector4 = loaded_vectorizer.transform([document4]).toarray()[0]

print("TF-IDF vector for the new document1:", vector2)
print("TF-IDF vector for the new document2:", vector4)
# Check the shape of

# num_documents, vector_size = loaded_matrix.shape
# print("Number of documents:", num_documents)
# print("Size of each TF-IDF vector:", vector_size)

A = np.array(vector1)
B = np.array(vector2)

#Compute cosine similarity
cosine_similarity = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
print("Cosine Similarity:", cosine_similarity)

A_flat = A.flatten()
B_flat = B.flatten()

# Compute cosine similarity
cosine_similarity = np.dot(A_flat, B_flat) / (np.linalg.norm(A_flat) * np.linalg.norm(B_flat))
print("Flatten Cosine Similarity:", cosine_similarity)
