from load_model import create_model
from load_data import create_data_generators
import os

def train_model(model, train_gen, val_gen, epochs):
    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // train_gen.batch_size,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=val_gen.samples // val_gen.batch_size,
    )
    return history

batch_size = 16
epochs = 10
train_gen, val_gen = create_data_generators(batch_size)

model = create_model()
history = train_model(model, train_gen, val_gen, epochs)

file_path = "../models/efficientnetb7_binary_classifier.h5"
model_dir = "../models"
os.makedirs(model_dir, exist_ok=True)
if os.path.exists(file_path):
    os.remove(file_path)

model.save(file_path)
