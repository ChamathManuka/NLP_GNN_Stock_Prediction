from tensorflow import keras
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras import models, layers
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from PIL import Image

# Load pre-trained VGG16 model without the top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = models.Sequential([
    base_model,
    layers.Flatten()  # Flatten the output to get feature vectors
])

# Load and preprocess the images to the required shape
def load_and_preprocess_image(img_path):
    # Open the image and resize to match the VGG16 expected input size
    img = Image.open(img_path).resize((224, 224))  # Resize to (224, 224)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return preprocess_input(img_array)  # Preprocess for VGG16

# Load your custom images
img_path1 = 'saved_image100k_1.png'
img_path2 = 'saved_image100k_2.png'

# Load and preprocess images
img1 = load_and_preprocess_image(img_path1)
img2 = load_and_preprocess_image(img_path2)

# Extract features for each image
features1 = model.predict(img1)
features2 = model.predict(img2)

# Calculate cosine similarity
cosine_sim = cosine_similarity(features1, features2)
print("Cosine Similarity:", cosine_sim[0][0])
