from src.model import build_model, save_model, load_model
from src.predict import predict, get_class_probabilities, print_class_probabilities
from src.util import CIFAR10_CLASS_NAMES as class_names
import sys

def main():
    # Example usage
    # TODO: accept filename as argument
    image_path = './working-dir/747-400.jpg'  # Replace with the path to your image

    # Predict an image
    predictions = predict(image_path, maintain_aspect_ratio=True)
    class_probabilities = get_class_probabilities(predictions, class_names)
    print_class_probabilities(class_probabilities)

if __name__ == "__main__":
    main()
