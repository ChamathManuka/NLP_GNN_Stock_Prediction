import re

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import csv
from tqdm import tqdm

# Download NLTK data if not already downloaded
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')


# Read the CSV file into a DataFrame
# df = pd.read_csv("news_articles90k.csv")
df = pd.read_csv("news_articles_test.csv")
# Create a new DataFrame to store the processed data
processed_data = []
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Iterate through the rows of the DataFrame
for index, row in df.iterrows():
    # Access data from each row
    full_text = row["full_text"]
    title = row["title"]
    if (isinstance(full_text, str) and isinstance(title, str)):
        paragraph = full_text + title
        print(paragraph)
        cleared_tokens = [re.sub(r"[^a-zA-Z]", "", token) for token in word_tokenize(paragraph)]
        tokens = [re.sub(r"\s+", " ", token) for token in cleared_tokens if token]
        stemmed_tokens = [stemmer.stem(token) for token in tokens]
        lemmatized_tokens = [lemmatizer.lemmatize(stemmed_token) for stemmed_token in stemmed_tokens]
        filtered_tokens = [token for token in lemmatized_tokens if token not in stop_words]
        # Add the processed data to the new DataFrame
        processed_data.append({"added_date": row["added_date"], "filtered_tokens": filtered_tokens})

# Create a new DataFrame from the processed data
processed_df = pd.DataFrame(processed_data)

# Save the processed DataFrame to a new CSV file
processed_df.to_csv("processed_news_articles_test.csv", index=False)
# processed_df.to_csv("processed_news_articles.csv", index=False)

