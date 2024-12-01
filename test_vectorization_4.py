from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import csv
import numpy as np


with open('processed_news_articles.csv', mode='r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)
    documents = []
    # Read each line in the CSV file
    for document in csv_reader:
        documents.append(document[1])
file.close()

tagged_data = [TaggedDocument(words=doc.split(), tags=[str(i)]) for i, doc in enumerate(documents)]

# model = Doc2Vec.load("doc2vec.model")
model = Doc2Vec(tagged_data, vector_size=50, window=3, min_count=2, epochs=100)
model.save("doc2vec50.model")

# Infer vector for a new document
vector1 = model.infer_vector("['belov', 'region', 'event', 'like', 'discontinu', 'edit', 'wa', 'postpon', 'south', 'asian', 'game']".split())
vector2 = model.infer_vector("['longstand', 'region', 'competit', 'could', 'discontinu', 'event', 'wa', 'postpon', 'yet', 'againth', 'south', 'asian', 'game']".split())
vector3 = model.infer_vector("['jeopardi', 'follow', 'anoth', 'delay', 'editionth', 'futur', 'south', 'asian', 'game']".split())


A = np.array(vector1)
B = np.array(vector2)

# Compute cosine similarity
cosine_similarity = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
print("Cosine Similarity:", cosine_similarity)
