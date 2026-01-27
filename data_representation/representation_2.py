import csv
import pickle

import joblib
import numpy as np
from PIL import Image
from keras import models, layers
from keras.applications import VGG16
from sklearn.preprocessing import MinMaxScaler

# Load pre-trained VGG16 model without the top layers
# base_model = VGG16(weights='imagenet', include_top=False, input_shape=(80, 40, 3))
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(33, 1000, 3))
model = models.Sequential([
    base_model,
    layers.Flatten()  # Flatten the output to get feature vectors
])


loaded_vectorizer = joblib.load("tf-idf_model_files/tfidf_vectorizer99k.joblib")
# loaded_matrix = joblib.load("tf-idf_model_files/tfidf_matrix99k.joblib")

# loaded_vectorizer = joblib.load("tf-idf_model_files/tfidf_vectorizer5k_1.joblib")
# loaded_matrix = joblib.load("tf-idf_model_files/tfidf_matrix5K_1.joblib")

print("Loaded model and matrix successfully.")

with open('../All_news_CSV_files/2024_09_26_test.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)

    date_image_list_testing = []
    count = 1
    # Process each line
    for row in csvreader:
        date = row[0]
        doc1_str = row[1]
        cleaned_string = doc1_str.strip('[]').replace(' ', '').split(',')
        leaned_words = [word.strip("'") for word in cleaned_string]
        document1 = ' '.join(leaned_words)
        vector1 = loaded_vectorizer.transform([document1]).toarray()[0]
        tfidf_vector = vector1
        # tfidf_vector = vector1[1:]
        # date_list.append(date)

        # Step 1: Calculate the sizes for each channel
        n_features = len(tfidf_vector)
        red_size = n_features // 3
        green_size = n_features // 3
        blue_size = n_features - red_size - green_size  # Ensure the total equals 10,000

        # Step 2: Split the 10,000-dimensional TF-IDF vector into three parts for RGB channels
        red_channel = tfidf_vector[:red_size]
        green_channel = tfidf_vector[red_size:red_size + green_size]
        blue_channel = tfidf_vector[red_size + green_size:]

        # If the blue channel has fewer features, pad it with zeros
        if len(blue_channel) < blue_size:
            blue_channel = np.pad(blue_channel, (0, blue_size - len(blue_channel)), mode='constant', constant_values=0)

        # Step 3: Normalize each channel independently to the 0-255 range
        red_scaled = MinMaxScaler((0, 255)).fit_transform(red_channel.reshape(-1, 1)).astype(int).flatten()
        green_scaled = MinMaxScaler((0, 255)).fit_transform(green_channel.reshape(-1, 1)).astype(int).flatten()
        blue_scaled = MinMaxScaler((0, 255)).fit_transform(blue_channel.reshape(-1, 1)).astype(int).flatten()

        # Step 4: Combine into an RGB image and reshape to a 100x100 grid
        rgb_image = np.stack([red_scaled, green_scaled, blue_scaled], axis=1)

        # Ensure the final RGB image has 10,000 pixels
        # if rgb_image.shape[0] != 10000:
        #     raise ValueError('Combined RGB channels do not sum to 10,000 pixels.')

        rgb_image = rgb_image.reshape(33, 1000, 3)
        # rgb_image = rgb_image.reshape(40, 40, 3)
        image1 = Image.fromarray(rgb_image.astype(np.uint8))
        img1_array = np.array(image1)
        img1_array = np.expand_dims(img1_array, axis=0)
        features1 = model.predict(img1_array)

        date_image = [date, features1]
        date_image_list_testing.append(date_image)
        count = count + 1
        print(count)

    # Save the list of (date, image) pairs
    with open('../date_feature_test_list_files/9K_2024_09_26.pkl', 'wb') as f:
        pickle.dump(date_image_list_testing, f)

    print("Date-Image list saved successfully!")
