import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB7

def create_model():
    base_model = tf.keras.applications.EfficientNetB7(
        include_top=False,
        weights='imagenet',
        input_shape=(600, 600, 3)
    )
    # Freeze the base model
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation='sigmoid')  # binary classification layer
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
