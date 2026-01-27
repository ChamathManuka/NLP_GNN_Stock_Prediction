import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Step 1: Read documents from a CSV file
input_file = '../test_simillar_vectors.csv'  # Path to your input CSV file
data = pd.read_csv(input_file)

# Step 2: Extract the documents into a list
documents = data['filtered_tokens'].tolist()

# Step 3: Create a TfidfVectorizer object
max_features = 10000  # Limit to top 10,000 features
tfidf_vectorizer = TfidfVectorizer(max_features=max_features)


# Step 4: Train (fit) the vectorizer on your corpus
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# Step 5: Convert the sparse matrix to a dense matrix for easy handling
dense_matrix = tfidf_matrix.todense()

# Step 6: Create a DataFrame to store the original indices and their corresponding TF-IDF vectors
# Convert the dense matrix to a DataFrame
tfidf_df = pd.DataFrame(dense_matrix, columns=tfidf_vectorizer.get_feature_names_out())

# Add the original index from the documents
#tfidf_df['index'] = data['index']

# Step 7: Save the DataFrame to a new CSV file
output_file = '../tfidf_vectors.csv'  # Path to your output CSV file
tfidf_df.to_csv(output_file, index=False)

print(f'TFIDF vectors saved to {output_file}.')
