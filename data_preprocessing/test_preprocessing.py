
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from tqdm import tqdm

# Download NLTK data if not already downloaded
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')
import csv

# Read the CSV file into a DataFrame
df = pd.read_csv("../Business_news_CSV_files/business_news_articles_v1.csv")
# Create a new DataFrame to store the processed data
processed_data = []
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Iterate through the rows of the DataFrame
for index, row in df.iterrows():
    # Access data from each row
    paragraph = row["full_text"]
    print(paragraph)
    print(type(paragraph))
    if isinstance(paragraph, str):
        tokens = word_tokenize(paragraph)
        stemmed_tokens = [stemmer.stem(token) for token in tokens]
        lemmatized_tokens = [lemmatizer.lemmatize(stemmed_token) for stemmed_token in stemmed_tokens]
        filtered_tokens = [token for token in lemmatized_tokens if token not in stop_words]
        # Add the processed data to the new DataFrame
        processed_data.append({"added_date": row["added_date"], "filtered_tokens": filtered_tokens})

# Create a new DataFrame from the processed data
processed_df = pd.DataFrame(processed_data)

# Save the processed DataFrame to a new CSV file
processed_df.to_csv("processed_business_news_articles.csv", index=False)

