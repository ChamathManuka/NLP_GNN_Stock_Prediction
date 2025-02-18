import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

loaded_vectorizer = joblib.load("tf-idf_model_files/tfidf_vectorizer10k.joblib")
loaded_matrix = joblib.load("tf-idf_model_files/tfidf_matrix10k.joblib")

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

tfidf_vectors = []
tfidf_vectors.append(vector1)
tfidf_vectors.append(vector2)
tfidf_vectors.append(vector3)
tfidf_vectors.append(vector4)



# Assuming tfidf_vectors is a 2D array of shape (n_samples, 500)
# Here, we simulate tfidf_vectors with random data for demonstration
# tfidf_vectors = np.random.rand(10, 500)  # Replace with actual TF-IDF vectors for your documents

# Step 1: Apply PCA or t-SNE to reduce the dimensionality to 3 components for RGB
# Using PCA here; replace with TSNE for nonlinear mapping if desired
pca = PCA(n_components=300)
reduced_vectors = pca.fit_transform(tfidf_vectors)  # Shape: (n_samples, 3)

# Optional: Use t-SNE for more nonlinear relationships
# tsne = TSNE(n_components=3)
# reduced_vectors = tsne.fit_transform(tfidf_vectors)

# Step 2: Normalize the reduced dimensions to fit RGB scale (0-255)
scaler = MinMaxScaler(feature_range=(0, 255))
rgb_scaled_vectors = scaler.fit_transform(reduced_vectors).astype(int)  # Convert to integer for RGB

# Step 3: Generate the RGB color image for each document
# Reshape the RGB vectors into a 2D format; here we'll make a simple square image grid

# Define the grid size, e.g., reshape (3D vector * 10 documents) to 10x10 pixels
grid_size = (10, 10)  # Adjust based on how many documents you have
image = rgb_scaled_vectors.reshape(grid_size[0], grid_size[1], 3)

# Display the image
plt.imshow(image)
plt.axis('off')
plt.show()
