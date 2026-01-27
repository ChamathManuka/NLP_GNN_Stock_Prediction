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

# Load pre-trained VGG16 model without the top layers
# base_model = VGG16(weights='imagenet', include_top=False, input_shape=(45, 200, 3))
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(40, 40, 3))
model = models.Sequential([
    base_model,
    layers.Flatten()  # Flatten the output to get feature vectors
])


# # Load the pickled list
# # with open('date_feature_list_files/date_feature_list_21.2.pkl', 'rb') as f:
# with open('date_feature_list_files/date_feature_list_41.2.pkl', 'rb') as f:
#     loaded_list1 = pickle.load(f)
#
# # with open('date_feature_list_files/date_feature_list_22.2.pkl', 'rb') as f:
# with open('date_feature_list_files/date_feature_list_42.2.pkl', 'rb') as f:
#     loaded_list2 = pickle.load(f)

# concatenated_list = list(chain(loaded_list1[1:], loaded_list2))

with open('date_feature_list_files/all_date_feature_list1.1.pkl', 'rb') as f:
    loaded_list1 = pickle.load(f)

with open('date_feature_list_files/all_date_feature_list1.2.pkl', 'rb') as g:
    loaded_list2 = pickle.load(g)

with open('date_feature_list_files/all_date_feature_list1.3.pkl', 'rb') as g:
    loaded_list3 = pickle.load(g)

with open('date_feature_list_files/all_date_feature_list1.4.pkl', 'rb') as g:
    loaded_list4 = pickle.load(g)

with open('date_feature_list_files/all_date_feature_list1.5.pkl', 'rb') as g:
    loaded_list5 = pickle.load(g)

concatenated_list = list(chain(loaded_list1, loaded_list2, loaded_list3, loaded_list4, loaded_list5))







with open('../date_feature_test_list_files/2024_09_30_5K_ALL_test.pkl', 'rb') as f:
    loaded_list_test = pickle.load(f)
    # loaded_list_test = loaded_list_test[:1500]
# Now you can use the loaded list
# print(loaded_list)

def find_most_similar(features1, image_list):
    """
    Finds the most similar image in a list to a given image.

    Args:
        image1: The reference image.
        image_list: A list of images to compare against.

    Returns:
        The most similar image and its hash distance.
    """

    # Extract features for each image
    # img1_array = np.array(image1)
    # img1_array = np.expand_dims(img1_array, axis=0)
    # features1 = model.predict(img1_array)
    min_distance = .85
    sorted_list = []
    # hash1 = imagehash.average_hash(Image.fromarray(image1))
    # min_distance = float('inf')
    # most_similar = None

    for row in image_list:
        # img2_array = np.array(image2[1])
        # img2_array = np.expand_dims(img2_array, axis=0)
        # features2 = model.predict(img2_array)
        distance = cosine_similarity(features1, row[1])[0][0]
        if distance > min_distance:
            sorted_list.append(row[0])

    return sorted_list

# Assuming your lists are named list1 and list



similar_pairs = []

for item2 in loaded_list_test:
    similar_pairs.append((item2, find_most_similar(item2[1], concatenated_list)))
    print("now running: "+str(item2))
        # Save the list of (date, image) pairs
with open('../selected_list_files/2024_09_30_5K_ALL_test.pkl', 'wb') as f:
    pickle.dump(similar_pairs, f)
    print(" list has been saved ##########################################################################################################################################################################")



