# Image Classification

## Description

This project is an image classification application using Convolutional Neural Networks (CNNs). It is a learning platform for me to get deep hands-on experience working with production-ready AI platforms. Check the TODO section for what's planned.

## Project Structure

- `data/`: Contains datasets and images.
- `src/`: Source code for data preprocessing, model building, and prediction.
- `results/`: Stores image output, models, and logs.

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/ImageClassificationProject.git
   cd ImageClassificationProject
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

- Train the model:

  ```bash
  python -m src.train
  ```

  This will create two models - one using data augmentation, and one without.

  Note: If the model(s) already exist(s), delete/rename them from `/results/model/` or run

  ```bash
  python -m src.train -f
  ```

- Compare the models:

  ```bash
  python -m src.compare
  ```

  This runs an evaluation of the two models and reports Loss and Accuracy:

  ```
  Model without Data Augmentation: Loss = 0.9609, Accuracy = 0.6886
  Model with Data Augmentation:    Loss = 0.8786, Accuracy = 0.7005
  ```

  The program will then run a prediction on a number of test images using both models. The test images can be found in `/results/output/<Ymd-HMS>/`

  Note: comparison can be run during the above training step by using `python -m src.train -c`

- Predict an image:

  ```python
  from src.predict import predict, get_class_probabilities, print_class_probabilities

  image_path = 'path_to_your_image.jpg'
  predictions = predict(image_path)
  class_probabilities = get_class_probabilities(predictions, class_names)
  print_class_probabilities(class_probabilities)
  ```

## TODO

- **Model Improvement:**

  - ~~Apply data augmentation techniques~~
  - Implement regularization techniques (dropout, batch normalization)
  - Perform hyperparameter tuning

- **Advanced Model Architectures:**

  - Use transfer learning with pre-trained models (e.g., VGG16, ResNet50)
  - Experiment with deeper networks

- **Expanding the Dataset:**

  - Create and label a custom dataset
  - Fine-tune the model on specific classes

- **Real-Time Object Classification:**

  - Integrate the model with a webcam for real-time classification
  - Deploy the model on a mobile device

- **Visualization and Interpretation:**

  - Visualize feature maps from convolutional layers
  - Generate Class Activation Maps (CAMs)

- **Evaluation and Reporting:**

  - Create a confusion matrix
  - Calculate detailed metrics (precision, recall, F1-score)

- **Application Development:**
  - Implement a CLI to run the project
  - Develop a web application for image upload and classification
  - Create an API for the image classification model

## Contributing

Contributions are welcome! Please submit a pull request.

## License

This project is licensed under the MIT License.
