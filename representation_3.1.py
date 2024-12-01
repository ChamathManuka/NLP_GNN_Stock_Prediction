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

# Load the pickled list
with open('date_feature_list1.1.pkl', 'rb') as f:
    loaded_list1 = pickle.load(f)

with open('date_feature_list2.1.pkl', 'rb') as f:
    loaded_list2 = pickle.load(f)

concatenated_list = list(chain(loaded_list1, loaded_list2))

with open('date_image_list_test.pkl', 'rb') as f:
    loaded_list_test = pickle.load(f)

def find_most_similar(features1, image_list):

    min_distance = .8
    sorted_list = []

    for row in image_list:
        distance = cosine_similarity(features1, row[1])[0][0]
        if distance > min_distance:
            sorted_list.append(row[0])

    return sorted_list

for item2 in loaded_list_test:
    similar_pairs = []
    total_out_list = []
    start = 0
    end = 2000
    for x in range(0, 40):
        print("now running " +item2[0]+ " lot #########################################################################################################################################################################")

        out_list = find_most_similar(item2[1], concatenated_list[start:end])
        total_out_list.append(out_list)
        start = start + 2000
        end = end + 2000


    similar_pairs.append((item2[0], total_out_list))
        # Save the list of (date, image) pairs
    with open('selected_list_1'+item2[0]+'.pkl', 'wb') as f:
        pickle.dump(similar_pairs, f)
        print(item2[0]+" list has been saved ##########################################################################################################################################################################")



