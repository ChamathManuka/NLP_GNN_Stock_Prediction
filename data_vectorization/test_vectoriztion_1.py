import openai
import numpy as np

# Set your API key
openai.api_key = "your_openai_api_key"

# Request embeddings using the OpenAI client
response = openai.Embedding.create(
    model="text-embedding-ada-002",  # Use a valid embedding model
    input="Testing 123"
)

# Extract the embedding data
embedding = response['data'][0]['embedding']

# Optionally slice to the first 256 dimensions
cut_dim = embedding[:256]

# Normalize the embedding using L2 normalization
def normalize_l2(vector):
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector

# Normalize the cut dimensions
norm_dim = normalize_l2(cut_dim)

# Print the normalized embedding
print(norm_dim)
