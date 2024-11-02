import csv
from datetime import date, timedelta
import requests
from bs4 import BeautifulSoup
from gensim.utils import simple_preprocess
import json

headers = {
    'User-Agent': 'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.1) Gecko/2008071615 Fedora/3.0.1-1.fc9 Firefox/3.0.1'
}


def save_to_csv(data, filename):
    """Saves the extracted data to a CSV file."""
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["added_date", "title", "full_text"])
        writer.writerows(data)


def extract_data(json_data):
    """Extracts added_date and full_text from the JSON data and returns a list of tuples."""
    data = []
    for article in json_data["articles"]:
        added_date = article["ADDED_DATE"]
        title = article["TITLE"]
        full_text = article["INTRO_TEXT"]

        data.append((added_date, title, full_text))
    return data


with open("news_articles90k.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["added_date", "title", "full_text"])

    for n in range(3000):
        try:
            # Prepare payload with desired date range
            payload = {
                'category': 0,  # Assuming you want all categories (change if needed)
                'datefrom': '2015/01/01',
                'dateto': '2024/01/01',
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
                print("JSON data parsed successfully.")

            plain_text = soup.get_text()

            data = extract_data(json_data)
            writer.writerows(data)

        except Exception as e:
            print(f"Error processing date {date}: {e}")
