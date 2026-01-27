import pandas as pd

# Read the CSV file
# df = pd.read_csv('processed_news_articles.csv')
df = pd.read_csv('../processed_news_articles_test.csv')

# Ensure 'added_date' is parsed as datetime
df['added_date'] = pd.to_datetime(df['added_date'])

# Extract only the date part
df['added_date'] = df['added_date'].dt.date

# Sort the DataFrame by 'added_date' in descending order
# df = df.sort_values(by='added_date', ascending=False)

# Write the preprocessed dates back to a new CSV file
df.to_csv('output_file_test.csv', index=False)
# df.to_csv('output_file.csv', index=False)
