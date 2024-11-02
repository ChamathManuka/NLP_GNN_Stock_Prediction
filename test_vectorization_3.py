import gensim.downloader as api
from gensim.models import Word2Vec

import csv
# Load pre-trained Word2Vec model
model = api.load("word2vec-google-news-300")

# Open the CSV file
with open('test_simillar_data.csv', mode='r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)
    # Read each line in the CSV file
    for document in csv_reader:
        print(document[2])
        document_vector = sum(model[word] for word in document if word in model) / len(document)
        print(document_vector)
