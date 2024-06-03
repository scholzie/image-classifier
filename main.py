from src.model import build_model, save_model, load_model
from src.predict import predict, get_class_probabilities, print_class_probabilities

def main():
    # Example usage
    image_path = 'path_to_your_image.jpg'  # Replace with the path to your image
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Predict an image
    predictions = predict(image_path, maintain_aspect_ratio=True)
    class_probabilities = get_class_probabilities(predictions, class_names)
    print_class_probabilities(class_probabilities)

if __name__ == "__main__":
    main()
