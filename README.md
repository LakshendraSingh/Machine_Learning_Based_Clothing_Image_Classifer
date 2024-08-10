# Fashion MNIST Classification with TensorFlow

This project demonstrates a simple image classification model using TensorFlow and Keras to classify images from the Fashion MNIST dataset. The model is a basic neural network with a single dense layer that predicts the category of clothing items.

## Project Overview

The Fashion MNIST dataset contains 70,000 grayscale images of 10 different types of clothing items. Each image is 28x28 pixels. This project includes:

- Loading and visualizing the dataset.
- Preprocessing the data (normalization).
- Building and training a neural network model.
- Making predictions and visualizing the results.

## Requirements

- Python 3.x
- TensorFlow 2.x
- Matplotlib
- GPU

You can install the required packages using pip:

```
pip install tensorflow matplotlib
```
# Dataset

 The dataset is loaded from TensorFlow's dataset library and consists of the following categories:

- 0 T-shirt/top
- 1 Trouser
- 2 Pullover
- 3 Dress
- 4 Coat
- 5 Sandal
- 6 Shirt
- 7 Sneaker
- 8 Bag
- 9 Ankle boot

# Code Overview
1. Data Loading and Visualization: Loads the Fashion MNIST dataset, visualizes some training and validation images.
2. Data Normalization: Normalizes the pixel values to be between 0 and 1.
3. Model Definition: Defines a simple neural network with one dense layer.
4. Model Training: Compiles and trains the model on the training data, validates on the validation data.
5. Predictions and Visualization: Makes predictions on a subset of validation images and visualizes the predictions.

# Notes

1. Ensure TensorFlow is correctly installed and configured to use GPU if available.
2. The model architecture and training parameters are kept simple for demonstration purposes. For more accurate results, consider experimenting with more complex architectures and hyperparameters.

# Acknowledgments

1. The creators of the Fashion MNIST dataset.
2. Matplotlib, TensorFlow and Keras are developed by the TensorFlow team.
3. NVIDIA's "Building a Brain in 10 Minutes" course.
