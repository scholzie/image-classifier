import numpy as np
from keras.models import load_model
from .data_preprocessing import load_and_preprocess_image

def predict(image_path, model_path='results/model/cifar10_model.keras', maintain_aspect_ratio=False):
    model = load_model(model_path)
    img = load_and_preprocess_image(image_path, maintain_aspect_ratio)
    predictions = model.predict(img)[0]
    return predictions

def get_class_probabilities(predictions, class_names):
    class_probabilities = [(class_names[i], predictions[i]) for i in range(len(class_names))]
    class_probabilities.sort(key=lambda x: x[1], reverse=True)
    return class_probabilities

def print_class_probabilities(class_probabilities):
    print(f"{'Class Name':<15} {'Probability':<10}")
    print("-" * 25)
    for class_name, probability in class_probabilities:
        print(f"{class_name:<15} {probability:<10.4f}")
