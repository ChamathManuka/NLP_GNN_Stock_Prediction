import numpy as np
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

loaded_vectorizer = joblib.load("tf-idf_model_files/tfidf_vectorizer100k.joblib")
loaded_matrix = joblib.load("tf-idf_model_files/tfidf_matrix100k.joblib")

print("Loaded model and matrix successfully.")

# Example new document
doc1_str = ['jeopardi', 'follow', 'anoth', 'delay', 'editionth', 'futur', 'south', 'asian', 'game']
doc2_str = ['belov', 'region', 'event', 'like', 'discontinu', 'edit', 'wa', 'postpon', 'south', 'asian', 'game']
doc3_str = ['popular', 'south', 'asian', 'game', 'sag', 'like', 'banish', 'region', 'calendar', 'schedul', 'event', 'wa', 'put', 'authoritiessubstandard', 'sag', 'face', 'axe', 'year']
doc4_str = ['longstand', 'region', 'competit', 'could', 'discontinu', 'event', 'wa', 'postpon', 'yet', 'againth', 'south', 'asian', 'game']
doc5_str = ['approxim', 'complaint', 'lodg', 'annual', 'involv', 'health', 'staff', 'includ', 'doctor', 'health', 'minist', 'ramesh', 'pathirana', 'revealedhealth', 'ministri', 'receiv', 'complaint', 'regard', 'medic', 'neglig']
document1 = ' '.join(doc1_str)
document2 = ' '.join(doc2_str)
document3 = ' '.join(doc3_str)
document4 = ' '.join(doc4_str)
document5 = ' '.join(doc5_str)


vector1 = loaded_vectorizer.transform([document1]).toarray()[0]
vector2 = loaded_vectorizer.transform([document2]).toarray()[0]
vector3 = loaded_vectorizer.transform([document3]).toarray()[0]
vector4 = loaded_vectorizer.transform([document4]).toarray()[0]
vector5 = loaded_vectorizer.transform([document5]).toarray()[0]

# Example 10,000-dimensional TF-IDF vector (simulated with random data for illustration)
# tfidf_vector = vector1[1:]  # Single document, 10,000 features
rgb_image_list = []
vector_list = []
# vector_list.append(vector1[1:])
# vector_list.append(vector2[1:])
# vector_list.append(vector3[1:])
# vector_list.append(vector4[1:])
vector_list.append(vector5[1:])


for tfidf_vector in vector_list:
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

    rgb_image = rgb_image.reshape(123, 271, 3)
    rgb_image_list.append(rgb_image)
    # Display the image
    plt.imshow(rgb_image.astype(int))
    plt.axis('off')
    plt.show()

# image1 = rgb_image_list[0]
image1 = Image.fromarray(rgb_image_list[0].astype(np.uint8))
image1.save("saved_image100k_1.png")
image2 = Image.fromarray(rgb_image_list[1].astype(np.uint8))
image2.save("saved_image100k_2.png")
image1 = Image.fromarray(rgb_image_list[2].astype(np.uint8))
image1.save("saved_image100k_3.png")
image2 = Image.fromarray(rgb_image_list[3].astype(np.uint8))
image2.save("saved_image100k_4.png")

