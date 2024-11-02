
#model = AutoModelForSequenceClassification.from_pretrained("allenai/llama", use_auth_token="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")

import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
#from transformers import HfApi

model_name = "allenai/specter"  # Use this model as LLaMA is not available
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Assuming your news data is a list of words (preprocessed using stemming, lemmatization)
news_data1 = "['longstand', 'region', 'competit', 'could', 'discontinu', 'event', 'wa', 'postpon', 'yet', 'againth', 'south', 'asian', 'game']"
news_data2 = "['popular', 'south', 'asian', 'game', 'sag', 'like', 'banish', 'region', 'calendar', 'schedul', 'event', 'wa', 'put', 'authoritiessubstandard', 'sag', 'face', 'axe', 'year']"
news_data3 = "['jeopardi', 'follow', 'anoth', 'delay', 'editionth', 'futur', 'south', 'asian', 'game']"

encoded_inputs = tokenizer(news_data1, return_tensors="pt", padding=True)
vectorized_news1 = model.bert(**encoded_inputs).last_hidden_state[:, 0, :]
print(vectorized_news1)

encoded_inputs = tokenizer(news_data3, return_tensors="pt", padding=True)
vectorized_news2 = model.bert(**encoded_inputs).last_hidden_state[:, 0, :]
print(vectorized_news2)

#print(vectorized_news1.shape)
#api = HfApi(token="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")

# Detach the tensors from the computation graph
vectorized_news1_detached = vectorized_news1.detach()
vectorized_news2_detached = vectorized_news2.detach()

# Convert the detached tensors to NumPy arrays
vectorized_news1_np = vectorized_news1_detached.numpy()
vectorized_news2_np = vectorized_news2_detached.numpy()

# Calculate Euclidean distance
euclidean_distance = np.linalg.norm(vectorized_news1_np - vectorized_news2_np)
print("Euclidean distance:", euclidean_distance)

# Calculate cosine similarity
cosine_similarity = np.dot(vectorized_news1_np, vectorized_news2_np) / (np.linalg.norm(vectorized_news1_np) * np.linalg.norm(vectorized_news2_np))
print("Cosine similarity:", cosine_similarity)