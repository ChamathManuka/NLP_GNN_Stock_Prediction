import csv
import json
import re
from datetime import date
from html.parser import HTMLParser

import requests
from bs4 import BeautifulSoup


class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return ''.join(self.fed)


headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36'
}

with open("All_news_CSV_files/biz_news_articles_test_test.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["added_date", "title", "full_text"])


    def strip_tags(html):
        s = MLStripper()
        s.feed(html)
        return s.get_data()


    def clean_article_text(full_text):
        # Remove JSON objects (basic pattern)
        full_text = re.sub(r'\{.*?\}', '', full_text)

        # Remove CSS-like style definitions
        full_text = re.sub(r'^\s*\/\*.*?\*\/\s*$|^\s*table\..*?\{.*?\}|body\s*\{.*?\}', '', full_text,
                           flags=re.DOTALL | re.MULTILINE)

        pattern = r"^\s*\/\*.*?\*\/\s*$|^\s*table\..*?\{.*?\}|MicrosoftInternetExplorer4|st1\:*{behavior:url\(#ieooui\)}|body\s*\{.*?\}"
        reg_text = re.sub(pattern, "", full_text, flags=re.DOTALL | re.MULTILINE)

        # Remove HTML/XML tags
        full_text = re.sub(r'<.*?>', '', reg_text)

        # Remove multiple newlines
        full_text = re.sub(r'\n+', '\n', full_text)

        # Remove excessive whitespace
        full_text = ' '.join(full_text.split())

        stripped = strip_tags(full_text)

        return stripped


    def extract_data(json_data):
        """Extracts added_date and full_text from the JSON data and returns a list of tuples."""
        data = []
        for article in json_data["articles"]:
            added_date = article["ADDED_DATE"]
            title = article["TITLE"]
            full_text = clean_article_text(article["INTRO_TEXT"])

            data.append((added_date, title, full_text))
        return data


    def save_to_csv(data, filename):
        """Saves the extracted data to a CSV file."""
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["added_date", "title", "full_text"])
            writer.writerows(data)


    for n in range(800):
        try:
            # Prepare payload with desired date range
            payload = {
                'category': 215,  # Assuming you want all categories (change if needed)
                'datefrom': '2024/01/01',
                'dateto': '2025/01/01',
                'start': 30 * n,  # Potentially for pagination (investigate further)
                'keywords': '',  # Empty for broad search, add keywords if needed
            }

            # Send POST request to the archive search URL
            response = requests.post(
                'https://www.dailymirror.lk/home/archivesearch', headers=headers, data=payload)
            # print(response.content)
            # Parse the response content
            soup = BeautifulSoup(response.content, 'html.parser')

            try:
                json_data = json.loads(response.content)  # Attempt to parse JSON
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                # Handle the error case (e.g., log the error or return a default value)
                exit(1)  # Exit with an error code (optional)
            else:
                print("JSON data parsed successfully.", n)

            plain_text = soup.get_text()

            data = extract_data(json_data)
            writer.writerows(data)
            print("now writing: ", data[-1][0])
            # Save the data to a CSV file
            # save_to_csv(data, "article_data.csv")





        except Exception as e:
            print(f"Error processing date {date}: {e}")
