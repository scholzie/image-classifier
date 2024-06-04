import os
import time
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

from src.model import load_custom_model
from src.util import AUG_MODEL_FILE_NAME, NO_AUG_MODEL_FILE_NAME, CIFAR10_CLASS_NAMES as class_names


def image_to_file(image_data, base_dir, file_name, title=""):
    os.makedirs(base_dir, exist_ok=True)

    # TODO: add table of contextual information (true label, prediction)
    plt.imshow(image_data)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.savefig(f'{base_dir}/{file_name}.png')
    plt.close()

def compare_models():
    # Load the CIFAR-10 dataset
    (_, _), (x_test, y_test) = cifar10.load_data()
    y_test = to_categorical(y_test, 10)

    # Normalize the input data
    x_test = x_test.astype('float32') / 255.0

    # Load models
    model_no_aug = load_custom_model(NO_AUG_MODEL_FILE_NAME)
    model_aug = load_custom_model(AUG_MODEL_FILE_NAME)

    # Evaluate models on test data
    loss_no_aug, accuracy_no_aug = model_no_aug.evaluate(x_test, y_test)
    loss_aug, accuracy_aug = model_aug.evaluate(x_test, y_test)

    # Print comparison
    print(f"Model without Data Augmentation: Loss = {loss_no_aug:.4f}, Accuracy = {accuracy_no_aug:.4f}")
    print(f"Model with Data Augmentation:    Loss = {loss_aug:.4f}, Accuracy = {accuracy_aug:.4f}")

    # Make predictions on a few test images
    # TODO: make number of predictions configurable
    indices = np.random.choice(len(x_test), 10, replace=False)
    x_sample = x_test[indices]
    y_sample = y_test[indices]
    
    predictions_no_aug = model_no_aug.predict(x_sample)
    predictions_aug = model_aug.predict(x_sample)

    # Compare predictions
    timestr = time.strftime("%Y%m%d-%H%M%S")
    print("\nSample Predictions:")
    for i, index in enumerate(indices):
        true_label = np.argmax(y_sample[i])
        no_aug_label = np.argmax(predictions_no_aug[i])
        aug_label = np.argmax(predictions_aug[i])
        print(f"\nImage {i+1} (#{index}):")
        print(f"True Label:          {true_label} ({class_names[true_label]})")
        print(f"Prediction No Aug:   {no_aug_label} ({class_names[no_aug_label]})")
        print(f"Prediction Aug:      {aug_label} ({class_names[aug_label]})")
        image_to_file(x_sample[i], base_dir=f'results/output/{timestr}', file_name=f'{i+1}_{index}_{class_names[true_label]}')

if __name__ == "__main__":    
    print("Comparing models...")
    compare_models()