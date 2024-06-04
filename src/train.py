from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from src.compare import compare_models
from src.model import build_model, save_model, load_custom_model
import os
import numpy as np
import sys

from src.util import AUG_MODEL_FILE_NAME, NO_AUG_MODEL_FILE_NAME



def get_data_generators(x_train, y_train, x_test, y_test):
    # Data augmentation for the training data
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    # No data augmentation for the test data
    test_datagen = ImageDataGenerator()

    # Create data generators
    train_generator = train_datagen.flow(x_train, y_train, batch_size=32)
    test_generator = test_datagen.flow(x_test, y_test, batch_size=32)

    return train_generator, test_generator

def load_normalized_dataset():
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Normalize the input data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    return x_train, y_train, x_test, y_test


def train_model(use_data_augmentation=True):
    x_train, y_train, x_test, y_test = load_normalized_dataset()

    if use_data_augmentation:
      # Get data generators
      train_generator, test_generator = get_data_generators(x_train, y_train, x_test, y_test)
      print(f'train_generator is type {type(train_generator)}')

    # Build the model
    model = build_model()

    fit_args = {'epochs': 10}

    if use_data_augmentation:
       fit_args.update({
          'x': train_generator,
          'validation_data': test_generator
       })
    else:
       fit_args.update({
          'x': x_train,
          'y': y_train,
          'validation_data': (x_test, y_test)
       })


    history = model.fit(**fit_args)

    # Save the model
    file_save_path = AUG_MODEL_FILE_NAME if use_data_augmentation else NO_AUG_MODEL_FILE_NAME
    save_model(model, file_path=file_save_path)

    return model, history

def train_model_with_augmentation():
   return train_model(use_data_augmentation=True)


def train_model_without_augmentation():
   return train_model(use_data_augmentation=False)



if __name__ == "__main__":
    if '-f' in sys.argv or not os.path.isfile(NO_AUG_MODEL_FILE_NAME):
      print("Training model without data augmentation...")
      train_model_without_augmentation()
      pass

    if '-f' in sys.argv or not os.path.isfile(AUG_MODEL_FILE_NAME):
      print("Training model with data augmentation...")
      train_model(use_data_augmentation=True)

    if '-c' in sys.argv:
      print("Comparing models...")
      compare_models()
