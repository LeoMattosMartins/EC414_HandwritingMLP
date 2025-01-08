# EC414 HandwritingMLP

Neural Net MLP to distinguish between handwritten digits 0-9.

## Overview

This project implements a Multi-Layer Perceptron (MLP) to classify handwritten digits from the MNIST dataset using Python. The implementation includes data preprocessing, model initialization, training, and evaluation.

## Installation

To run this project, you need to have Python and the following libraries installed:

 - numpy
 - tqdm
 - keras
 - scikit-learn
 - matplotlib
 - You can install the required libraries using pip:

  ```bash
    pip install numpy tqdm keras scikit-learn matplotlib
  ```

## Data Preprocessing

The MNIST dataset is loaded and preprocessed in the following steps:

 - Load the MNIST dataset.
 - Normalize the pixel values between 0 and 1.
 - Flatten the images into 1D arrays.
 - Convert the class labels into one-hot encoded vectors.
 - Split the training data into training and validation sets.
  ```Python
  from keras.datasets import mnist
  from sklearn.model_selection import train_test_split
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train = x_train.astype('float32') / 255
  x_test = x_test.astype('float32') / 255
  x_train = x_train.reshape((len(x_train), 784))
  x_test = x_test.reshape((len(x_test), 784))
  y_train = np.eye(10)[y_train]
  y_test = np.eye(10)[y_test]
  x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.16666, random_state=42)
  ```

## Model Initialization

The MLP class is defined to initialize and train the neural network. The weights are initialized using He initialization for better performance.

  ```Python
class MLP:
    def __init__(self, batch=64, lr=1e-2, epochs=50):
        self.batch = batch
        self.epochs = epochs
        self.lr = lr
        self.loss = []
        self.acc = []
        self.init_weights()

    def init_weights(self):
        self.W1 = np.random.randn(784, 256) / (0.5 * np.sqrt(784))
        self.W2 = np.random.randn(256, 128) / (0.5 * np.sqrt(256))
        self.W3 = np.random.randn(128, 10) / np.sqrt(128)
  ```
## Activation Function

The ReLU activation function is used in the hidden layers to introduce non-linearity.

```Python
def ReLU(self, z):
    return np.maximum(0, z)
```
## Training and Evaluation

The training and evaluation process involves forward propagation, loss calculation, backpropagation, and accuracy measurement. The model is trained for a specified number of epochs with a given batch size and learning rate.

For more details, you can refer to the full mlp.py file.
