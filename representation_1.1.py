import keras.src.applications
import numpy as np
from PIL import Image
import pickle
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import csv
import pickle
import imagehash
from PIL import Image
from itertools import chain
from tensorflow import keras
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras import models, layers
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from PIL import Image


# loaded_vectorizer = joblib.load("tf-idf_model_files/tfidf_vectorizer99k.joblib")
loaded_vectorizer = joblib.load("tf-idf_model_files/tfidf_vectorizer9K.joblib")
# loaded_matrix = joblib.load("tf-idf_model_files/tfidf_matrix99k.joblib")
loaded_matrix = joblib.load("tf-idf_model_files/tfidf_matrix9K.joblib")

print("Loaded model and matrix successfully.")

# Load pre-trained VGG16 model without the top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(80, 40, 3))
# base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 220, 3))
model = models.Sequential([
    base_model,
    layers.Flatten()  # Flatten the output to get feature vectors
])




def represent_data(csvreader, name):
    date_image_list = []
    count = 1
    # Process each line
    for row in csvreader:
        date = row[0]
        doc1_str = row[1]
        cleaned_string = doc1_str.strip('[]').replace(' ', '').split(',')
        leaned_words = [word.strip("'") for word in cleaned_string]
        document1 = ' '.join(leaned_words)
        vector1 = loaded_vectorizer.transform([document1]).toarray()[0]
        # tfidf_vector = vector1[1:]
        tfidf_vector = vector1
        # date_list.append(date)

        # Step 1: Calculate the sizes for each channel
        n_features = len(tfidf_vector)
        red_size = n_features // 3
        green_size = n_features // 3
        blue_size = n_features - red_size - green_size  # Ensure the total equals 100,000

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

        rgb_image = rgb_image.reshape(80, 40, 3)
        # rgb_image = rgb_image.reshape(150, 220, 3)
        image1 = Image.fromarray(rgb_image.astype(np.uint8))
        #new code change here

        img1_array = np.array(image1)
        img1_array = np.expand_dims(img1_array, axis=0)
        features1 = model.predict(img1_array)

        date_image = [date, features1]
        date_image_list.append(date_image)
        count = count + 1
        print(count)

    # Save the list of (date, image) pairs
    with open('9K_date_feature_list'+name+'.pkl', 'wb') as f:
        pickle.dump(date_image_list, f)
        print("Date-Feature list saved successfully! "+name)


with open('All_news_CSV_files/processed_documents_9K.pkl', 'rb') as f:
    loaded_list2 = pickle.load(f)

# loaded_list3 = loaded_list2[:40000]
# represent_data(loaded_list3, '1.1')
#
# loaded_list4 = loaded_list2[40000:80000]
# represent_data(loaded_list4, '1.2')

loaded_list5 = loaded_list2[80000:120000]
represent_data(loaded_list5, '1.3')

loaded_list6 = loaded_list2[120000:160000]
represent_data(loaded_list6, '1.4')

loaded_list7 = loaded_list2[160000:]
represent_data(loaded_list7, '1.5')
#

