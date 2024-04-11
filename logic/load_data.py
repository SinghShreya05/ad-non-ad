import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_data_generators(base_dir, batch_size):
    # train_datagen = ImageDataGenerator(
    #     rescale=1.0 / 255, validation_split=0.2  # using 20% of the data for validation
    # )
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255  
    )

    train_generator = train_datagen.flow_from_directory(
        "../dataset/train",
        target_size=(600, 600), 
        batch_size=batch_size,
        class_mode="binary",  # binary labels
        # subset="training",  # set as training data
    )

    validation_generator = train_datagen.flow_from_directory(
        "../dataset/validation",
        target_size=(600, 600),
        batch_size=batch_size,
        class_mode="binary",
        # subset="validation",  # set as validation data
    )

    return train_generator, validation_generator
