import csv
import json
from datetime import date

import requests
from bs4 import BeautifulSoup

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36'
}

proxies = {
    "http": "http://35.183.5.23:11",
    "https": "https://35.183.5.23:11",
}

with open("All_news_CSV_files/test.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["added_date", "title"])


    def extract_data(json_data):
        """Extracts added_date and full_text from the JSON data and returns a list of tuples."""
        data = []
        for article in json_data["articles"]:
            added_date = article["ADDED_DATE"]
            title = article["TITLE"]

            data.append((added_date, title))
        return data


    def save_to_csv(data, filename):
        """Saves the extracted data to a CSV file."""
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["added_date", "title"])
            writer.writerows(data)


    for n in range(800):
        try:
            # Prepare payload with desired date range
            payload = {
                'category': 0,  # Assuming you want all categories (change if needed)
                'datefrom': '2024/01/01',
                'dateto': '2010/01/01',
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
