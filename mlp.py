# Importing Libraries

import numpy as np
import tqdm
import random
import sys
from keras.datasets import mnist
from  sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
%matplotlib inline

# Data Preprocessing

# Load the MNIST Dataset with variable names as given above
(x_train, y_train), (x_test, y_test) = mnist.load_data()

orig_train = x_train

# Normalize the pixel values between 0 and 1
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Flatten the images into 1D arrays
x_train = x_train.reshape((len(x_train), 784))
x_test = x_test.reshape((len(x_test), 784))

# Convert the class labels into one-hot encoded vectors
y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]

# Split the train data into training and validation data
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.16666, random_state=42)


# Print the shapes of the data arrays
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_valid shape:', x_valid.shape)
print('y_valid shape:', y_valid.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

# Plot the first 9 images in the dataset
for i in range(9):
 plt.subplot(330 + 1 + i)
 plt.imshow(orig_train[i], cmap=plt.get_cmap('gray'))
plt.show()

class MLP:
    def __init__(self, batch = 64, lr = 1e-2, epochs = 50):

        '''

        Initialization Function

        '''
        # DO NOT CHANGE
        self.batch = batch
        self.epochs = epochs
        self.lr = lr

        self.loss = []
        self.acc = []

        # Add here any other variable intitalization you want (if any - you do not need to modify this function

        # END

        # DO NOT CHANGE
        self.init_weights()

    def init_weights(self):

        '''

        This function initializes the weights for the neural network

        '''

        # We initialize the weights with appropriate dimensions (based on the input and output dimension) using np.random() function
        # There should be 256 neurons in the first hidden layer, and 128 neurons in the second hidden layer
        # The scalings are magical constants that produce "He initialization". You can google "Xavier He initialization" to learn more about different initialization schemes.
        # These scalings make SGD work better, and also help avoid numerical overflow issues in the final sigmoid activation.

        self.W1 = np.random.randn(784, 256) / (0.5 * np.sqrt(784))
        self.W2 = np.random.randn(256, 128) / (0.5 * np.sqrt(256))
        self.W3 = np.random.randn(128, 10)  / np.sqrt(128)

    def ReLU(self, z):

        '''

        Computes the Rectified Linear Unit (ReLU) activation on the given input and returns the output.
        The ReLU activation function sets all negative values in the input to 0, and leaves all non-negative values unchanged.

        arguments:
        z: The input to the ReLU activation function.

        returns:
        The output of the ReLU activation function, which is the element-wise maximum of 0 and the input. Should have the same shape as z
        ReLU(z) = max(0, z)

        '''
        # Compute ReLU activation on the given input and return the output


        return np.maximum(0,z)

    def dReLU(self, z):

        '''

        Computes the derivative of the Rectified Linear Unit (ReLU) activation on the given input and returns the output.
        The derivative of the ReLU activation function is 1 for all non-negative input values, and 0 for all negative input values.

        arguments:
        z: The input to the dReLU function.

        returns:
        The output of the dReLU function, which is the element-wise derivative of the ReLU activation function. Should have the same shape as z
        ReLU(z) = max(0, z)

        '''

        # Compute the derivative of ReLU activation on the given input and return the output
        # Ignore any points of non-differentiability.

        return (z > 0).astype(float)

    def sigmoid(self, z):

        '''

        Computes the Sigmoid activation on the given input and returns the output.

        arguments:
        z: The input to the Sigmoid function

        returns:
        The output of the Sigmoid function, should have the same shape as x
        sigmoid(z) = 1/(1 + e^-z)

        '''

        # Compute Sigmoid activation on the given input and return the output

        return 1 / (1 + np.exp(-z))

    def dsigmoid(self, z):

        '''

        Computes the derivative of the Sigmoid activation on the given input and returns the output.

        arguments:
        z: The input to the dsigmoid function.

        returns:
        The output of the dsigmoid function, which is the element-wise derivative of the sigmoid activation function.
        sigmoid(z) = 1/(1 + e^z)

        '''
        # Compute the derivative of Sigmoid activation on the given input and return the output

        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def forward(self, X):

        '''
        Computes the forward pass on the network.

        Performs the mathematical operations on the weights andinputs, and makes a prediction using a neural network with three hidden layers.
        The forward pass of the neural network is implemented by performing the following operations:


        arguments:
        X: The input data on which the model has to make predictions. Should have shape [B, 784] for some B

        returns:
        a3: the output layer. Should have shape [B, 10]

        This function should use the weight matrices W1, W2, W3 to compute the output of the network.
        The first two layers (the hidden layers) should use the ReLU activation function.
        The last layer (the output layer) should use the sigmoid activation function.

        So, if v1 = X W1, then the first hidden layer is a1 = ReLU(v1)

        Similarly, if v3 = a3 W3, then the output layer is a3 = sigmoid(v3)

        You must store a3 in self.a3 for access in later functions.

        You will also probably find it helpful to store some intermediate values you compute during the forward pass (like hidden layer activations)
        as attributes of self. For example, you will likely want to store self.a1.
        These stored values will not be checked in our checking code, but you will probably need them when you implement the backward pass.

        You MUST NOT store intermediate values in global scope (this would be somewhat difficult to do, but do not try to do it).

        '''

        # Perform the mathematical operations on the weights, inputs, and bias and make a prediction.
        # The prediction should be stored in self.a3, and also returned from this function.

        self.X = X
        self.v1 = np.dot(X, self.W1)
        self.a1 = self.ReLU(self.v1)
        self.v2 = np.dot(self.a1, self.W2)
        self.a2 = self.ReLU(self.v2)
        self.v3 = np.dot(self.a2, self.W3)
        self.a3 = self.sigmoid(self.v3)

        return self.a3

    def squared_loss(self, y, y_preds):

        '''
        Computes the squared loss of the neural network's prediction.

        arguments:
        y: The true label(s) for the input(s) to the neural network.
        y_preds: The predicted label(s) for the input(s) to the neural network

        returns:
        error: The squared loss between the predicted output of the neural network and the true label(s).

        '''
        loss = 0.5 * np.sum((y - y_preds)**2)/np.shape(y)[0]

        return loss

    def backprop(self, x, y):

        '''

        Computes the gradient of all weights and biases of the neural network based on the errors made during training.
        This function performs backpropagation on the neural network to update the weights and biases.
        It first computes the error of the model on all training batches.
        Then, it finds the derivative of the loss with respect to each weight vector and bias.

        arguments:
        x: The input data on which the model made predictions during the forward pass. Shape = [B, 784] for batch size B
        y: The true label(s) for the input(s) to the neural network.  Shape = [B, 10] for batch size B

        This function need not return anything.
        However, it must set the following variables:

        self.dLdW3: shape [128, 10]
        self.dLdW2: shape [256, 128]
        self.dLdW1: shape [784, 256]

        Each of these should hold the gradient of the loss computed by squared_loss(forward(x), y) with respect to Wi.
        '''


        m = x.shape[0]

        # Compute derivatives for layer 3
        dA3 = self.a3 - y
        dV3 = dA3 * self.dsigmoid(self.v3)
        self.dLdW3 = np.dot(self.a2.T, dV3) / m

        # Compute derivatives for layer 2
        dA2 = np.dot(dV3, self.W3.T)
        dV2 = dA2 * self.dReLU(self.v2)
        self.dLdW2 = np.dot(self.a1.T, dV2) / m

        # Compute derivatives for layer 1
        dA1 = np.dot(dV2, self.W2.T)
        dV1 = dA1 * self.dReLU(self.v1)
        self.dLdW1 = np.dot(x.T, dV1) / m

        # # END CODE

        # DO NOT CHANGE
        assert self.dLdW3.shape == self.W3.shape
        assert self.dLdW2.shape == self.W2.shape
        assert self.dLdW1.shape == self.W1.shape

    def apply_sgd(self):

        '''

        Updates the weights and biases of the neural network based on the gradient found in the backpropogation step.
        The updates should be performed by subtracting from the weight the learning rate times the corresponding derivative of the weight/bias.

        Using the learning rate self.lr, this function should implement

        W = W - lr * gradient of loss with respect to W

        for all parameters W.

        After this function is finished,
        self.W1, self.W2 and self.W3 should all contain the updated values for W1, W2, W3.
        '''

        # Update the weights using the derivatives calculated above
        # DO NOT call `backprop` in this function: this function should ASSUME that `backprop` has already been called!

        self.W3 -= self.lr * self.dLdW3
        self.W2 -= self.lr * self.dLdW2
        self.W1 -= self.lr * self.dLdW1

        # END CODE

    def shuffle(self, x, y):

        '''

        This function shuffles the input data and corresponding labels in order to ensure randomness in the order of training examples.

        arguments:

        x: The input data on which the model has to make predictions (shape [B, 784] for some integer B)
        y: The true label(s) for the input(s) to the neural network. (shape [B, 10] for some integer B)

        returns:
        shuffled input data and corresponding labels

        '''

        # DO NOT CHANGE
        idx = [i for i in range(x.shape[0])]
        np.random.shuffle(idx)
        x = x[idx]
        y = y[idx]
        return x, y

    def train(self, xtrain, ytrain, xvalid, yvalid):

        '''

        This function performs the training of the neural network using all the functions defined above.

        arguments:
        xtrain: The training data (shape [B, 784])
        ytrain: The labels for the training data (shape [B, 10])
        xvalid: The validation data (shape [C, 784])
        yvalid: The labels for the validation data (shape [C, 10])

        '''
        for epoch in range(self.epochs):
            l = 0
            acc = 0
            xtrain, ytrain = self.shuffle(xtrain, ytrain)

            for batch in range(xtrain.shape[0]//self.batch-1):
                start = batch*self.batch
                end = (batch + 1)*self.batch
                x = np.array(xtrain[start:end])
                y = np.array(ytrain[start:end])
                self.forward(x)
                l += self.squared_loss(y, self.a3)

                acc += np.count_nonzero(np.argmax(self.a3, axis = 1) == np.argmax(y, axis = 1)) / self.batch
                self.backprop(x, y)
                self.apply_sgd()

            valid_acc = self.test(xvalid, yvalid)
            print("Epoch " + str(epoch + 1) + ":")
            print("Training Accuracy: %.2f%%" % (acc*100/(xtrain.shape[0]//self.batch)))
            print("Validation Accuracy: %.2f%% \n" % (valid_acc*100))
            self.loss.append(l/(xtrain.shape[0]//self.batch))
            self.acc.append(acc*100/(xtrain.shape[0]//self.batch))

    def plot(self):

        '''

        This function will be used to plot the overall training accuracy and loss graphs for the trained Neural Network

        '''

        fig, axs = plt.subplots(2)
        fig.tight_layout(pad = 3.0)
        plt.title("Model Loss and Accuracy")
        axs[0].set_title("Loss Plot")
        axs[0].set_xlabel("Epochs")
        axs[0].set_ylabel("Loss")
        axs[0].plot(self.loss)
        axs[1].set_title("Accuracy Plot")
        axs[1].set_xlabel("Epochs")
        axs[1].set_ylabel("Accuracy")
        axs[1].plot(self.acc)

    def test(self, xtest, ytest):

        '''

        This function takes in the test data and respective labels and evaluates the accuracy of the neural network on this data.

        arguments:
        xtest: The test data
        ytest: The labels for the test data

        returns:
        acc: accuracy of the model on the given test data

        '''

        self.forward(xtest)
        acc = np.count_nonzero(np.argmax(self.a3, axis = 1) == np.argmax(ytest, axis = 1)) / xtest.shape[0]
        return acc

    def predict(self, x):

        '''

        Functions exactly like the test() function, except it returns the predictions instead of accuracy

        arguments:
        x: The data

        returns:
        predictions: Predictions of the model on the input data

        '''

        predictions = []
        for i in x:
          image = i.reshape(1, -1)
          self.forward(image)
          predictions.append(np.argmax(self.a3, axis = 1))
        return np.array(predictions)

# DO NOT CHANGE
def backprop_test():
  count = 0
  for i in range(5):
    print(f"Running backprop test replicate {i+1} of 5...")
    test = MLP(64, 5e-3, 1)
    check_no_globals = MLP(64, 5e-3, 1)
    x, y = test.shuffle(x_train, y_train)
    yhat = test.forward(x)
    computed_loss = test.squared_loss(y, yhat)

    test.backprop(x, y)
    first_dW1 = test.dLdW1.copy()
    check_no_globals = MLP(64, 5e-3, 1)
    test.backprop(x, y)
    if np.linalg.norm(first_dW1 - test.dLdW1) > 1e-7:
      print(f"The backprop output changed after running a forward pass on an independent model... did you store some intermediate values in global scope?")
      count += 1

    param_count = (784 * 256) + (256 * 128) + (128 * 10)# + 256 + 128 + 10
    v_scale = 1e-6 /np.sqrt(param_count)

    v_W1 = np.random.normal(0, v_scale, (784 * 256)).reshape(test.W1.shape)
    v_W2 = np.random.normal(0, v_scale, (256 * 128)).reshape(test.W2.shape)
    v_W3 = np.random.normal(0, v_scale, (128 * 10)).reshape(test.W3.shape)

    dot_prod = 0
    dot_prod += np.sum(test.dLdW1 * v_W1)
    dot_prod += np.sum(test.dLdW2 * v_W2)
    dot_prod += np.sum(test.dLdW3 * v_W3)

    Wv1 = test.W1 + v_W1.reshape(784, 256)
    Wv2 = test.W2 + v_W2.reshape(256, 128)
    Wv3 = test.W3 + v_W3.reshape(128, 10)

    l1 = test.squared_loss(y, test.sigmoid(test.ReLU(test.ReLU(x.dot(test.W1)).dot(test.W2)).dot(test.W3)))
    l2 = test.squared_loss(y, test.sigmoid(test.ReLU(test.ReLU(x.dot(    Wv1)).dot(    Wv2)).dot(    Wv3)))

    if np.abs(computed_loss - l1)/(np.abs(computed_loss) + np.abs(l1)) > 1e-5:
      print("there is an error in forward: expected value: {l1}, implemented value: {computed_loss}")
      count += 1

    norm = 0
    norm += np.linalg.norm(v_W1)**2
    norm += np.linalg.norm(v_W2)**2
    norm += np.linalg.norm(v_W3)**2
    norm = np.sqrt(norm)
    t1 = dot_prod/norm
    t2 = (l2 - l1)/norm
    error = abs(t1 - t2)/(abs(t1) +  abs(t2))
    if error > 1e-4:
      print(f"backprop check failure with (computed grad - numerical grad)/ (computed_grad + numerical grad): {error}, computed grad: {t1}, numerical grad: {t2}")
      count+=1
  if count >= 1:
    print("Something is wrong...")
  else:
    print("Great Job, backprop check passed!")

backprop_test()

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.to_numpy() / 255.0
y = mnist.target.astype(int).to_numpy()

# Manually create one-hot encoded labels
num_classes = 10
y_one_hot = np.zeros((y.size, num_classes))
y_one_hot[np.arange(y.size), y] = 1

# Split the dataset into training, validation, and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)
xtrain, xvalid, ytrain, yvalid = train_test_split(xtrain, ytrain, test_size=0.2, random_state=42)

# Initialize the MLP model
nn = MLP(batch=64, lr=1.0, epochs=20)

# Train the model
nn.train(xtrain, ytrain, xvalid, yvalid)

# Plot training accuracy and loss
nn.plot()

# Test the model
test_accuracy = nn.test(xtest, ytest)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

fig, axs = plt.subplots(3, 3, figsize=(10, 10))
plt.subplots_adjust(hspace=0.5, wspace=0.5)

for i in range(9):
    row = i // 3
    col = i % 3
    x = random.randint(0, len(y_test))
    plot_input = x_test[x].reshape(28, 28)
    pred = nn.predict(x_test[x].reshape(1, -1))
    axs[row, col].imshow(plot_input, cmap=plt.get_cmap('gray'))
    axs[row, col].set_title("Model Prediction: %d, True Label: %d" % (pred, np.argmax(y_test[x])))

plt.show()

preds = nn.predict(x_test)
cm = confusion_matrix(np.argmax(y_test, axis = 1), preds)
cmdisplay = ConfusionMatrixDisplay(cm)
cmdisplay.plot()
plt.show()
