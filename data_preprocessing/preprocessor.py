import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK data if not already downloaded
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

# Read the CSV file into a DataFrame
# df = pd.read_csv("news_articles90k.csv")
# df = pd.read_csv("Business_news_CSV_files/business_test_news_articles.csv")
df = pd.read_csv("../process_7.3_files/csv_files/old_news_articles_processed_concat.csv")
# Create a new DataFrame to store the processed data
processed_data = []
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Iterate through the rows of the DataFrame
for index, row in df.iterrows():
    # Access data from each row
    paragraph = row['filtered_tokens']
    if (isinstance(paragraph, str)):
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
processed_df.to_csv("process_7.3_files/csv_files/tokenized_news_articles_processed_concat.csv", index=False)
# processed_df.to_csv("processed_news_articles.csv", index=False)
