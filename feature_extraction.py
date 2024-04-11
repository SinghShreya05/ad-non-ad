import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications.efficientnet import preprocess_input
from pathlib import Path


def create_model():
    base_model = EfficientNetB7(
        weights="imagenet", include_top=False, input_shape=(600, 600, 3)
    )
    model = tf.keras.Sequential([base_model, tf.keras.layers.GlobalAveragePooling2D()])
    return model

model = create_model()

def preprocess_and_extract_features(img_path):
    img = image.load_img(img_path, target_size=(600, 600))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features


def process_directory_and_save(directory_path):
    path_list = Path(directory_path).rglob("*.png")
    i = 0
    for img_path in path_list:
        features = preprocess_and_extract_features(str(img_path))
        feature_filename = f"ads_features/{i}.npy"
        np.save(feature_filename, features)
        i += 1

directory_path = "Ads/"
os.makedirs('ads_features', exist_ok=True)

process_directory_and_save(directory_path)

print(
    f"Processed all .png files from {directory_path} and saved features individually."
)
