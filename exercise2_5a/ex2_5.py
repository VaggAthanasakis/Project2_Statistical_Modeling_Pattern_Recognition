## Vasilis Diakoloukas 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

import sys
import os

# Change the stdout encoding to UTF-8
#sys.stdout.reconfigure(encoding='utf-8')

# Neural Network Dense (Fully Connected) Layer without activation
class Dense():
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    #Forward Propagation on a Dense Layer
    def forward(self, input):
        self.input = input
        # Add Code Here
        #fwd =  np.dot(self.input , self.weights) + self.bias
        fwd =  np.dot(self.weights, self.input) + self.bias
        return fwd

    #Backward Propagation on a Dense Layer
    # dE_dY is dE/dY Gradient
    # dE_dW is dE/dW Gradient
    # dE_dB is dE/dB Gradient
    # dE_dX is dE/dX Gradient
    def backward(self, dE_dY, learning_rate):
        dZ_dW = self.input.T                    # dZ_dW = Xi
        dE_dW = np.dot(dE_dY,dZ_dW)/ self.input.shape[1]# dE_dW =  dE_dY / dZ_dW /// Normalizing by Batch Size so that the dot product can be calculated
                                                                                ##// Ensures that the gradient computed is averaged over all samples in the batch.
        dY_dX = self.weights                    # dY_dX = W since we have no activation function
        dE_dX = np.dot(dY_dX.T,dE_dY)           # dE_dX = dE_dY * dY_dX = dE_dY * W
        
        # Since b is added to each neuron's output we need sum for each neuron across the batch
        # dE_dB = sum(dE_dZ * dZ_db) = sum(dE_dZ * 1) = sum(dE_dZ) = sum(dE_dY) => dE_dB = sum(dE_dY)
        # keepdims = True ensures that the resulting gradient has the same shape as B
        dE_dB = np.sum(dE_dY, axis=1, keepdims=True)/ self.input.shape[1]   

        self.update_weights(dE_dW, dE_dB, learning_rate)
        return dE_dX
    

    # Update Layer Weights and bias
    def update_weights(self, dE_dW, dE_dB, learning_rate):
        # Add Code Here
        self.weights -= learning_rate * dE_dW
        self.bias -= learning_rate * dE_dB


# Neural Network Activation Layer Abstract Class
# Consider Activation as a separate layer for more flexibility
# Properties and methods will be inherited into the specific activation function classes
class Activation():
    def __init__(self, activation, activation_grad):
        self.activation = activation
        self.activation_grad = activation_grad

    # input: is the input to the activation function
    # Y: is the output of the activation function
    def forward(self, input):
        self.input = input
        Y = self.activation(input)    # perform the forward pass
        return Y

    # Backward estimation of dE/dX using the activation prime (derivative)
    def backward(self, dE_dY , learning_rate = 0): #learning rate is not used for this scenario 
        dE_dX = dE_dY * self.activation_grad(self.input) 
        return dE_dX
 

# Softmax Activation Function
# Should be used in the output layer especially when Cross-Entropy is considered
class Softmax():
    def forward(self, input):
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output

    def backward(self, gradient_output,learning_rate = 0):
        n = np.size(self.output)
        # J = np.diag(self.output.flatten()) - np.outer(self.output, self.output.T)
        # return np.dot(J,gradient_output)
        return np.dot((np.identity(n) - self.output.T) * self.output, gradient_output)


# Tanh Activation Function
class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            act = np.tanh(x)
            return act

        def tanh_grad(x):
            # Add Code Here
            actGrad = 1 - np.tanh(x) ** 2
            return actGrad

        super().__init__(tanh, tanh_grad)


# Sigmoid (Logistic) Activation Function
class Sigmoid(Activation):
    def __init__(self):
        # Logistic Activation Function
        def sigmoid(x):
            act = 1 / (1 + np.exp(-x))
            return act

        # Activation Function Gradient (Derivative)
        def sigmoid_grad(x):
            s = sigmoid(x)
            actGrad = s * (1 - s)
            return actGrad

        super().__init__(sigmoid, sigmoid_grad)

        
# Return the the cross entropy loss
def loss_cross_entropy(y_true, y_pred):
    B = len(y_true)
    # Using the cross entropy formula to calculate the loss
    # We have added a small value in every ln operation so that we avoid ln(0)
    loss = (1/B)*np.sum(-y_true*np.log(y_pred+ 1e-15)-(1-y_true)*np.log(1-y_pred+1e-15))
    return loss

# Return the derivative of the cross entropy loss
def loss_cross_entropy_grad(y_true, y_pred):
    lossGrad =  -(y_true / y_pred - (1 - y_true) / (1 - y_pred))
    return lossGrad

def mse(y_true, y_pred):
    # Add Code Here
    loss = np.mean((y_true - y_pred) ** 2)
    return loss

def mse_grad(y_true, y_pred):
    # Add Code Here
    n = y_true.shape[0]
    lossGrad = (2 / n) * (y_pred - y_true)
    return lossGrad

def preprocess_data(x, y, limit):
    # reshape and normalize input data
    x = x.reshape(x.shape[0], 28 * 28, 1)
    x = x.astype("float32") / 255
    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y = to_categorical(y)
    y = y.reshape(y.shape[0], 10, 1)

    return x[:limit], y[:limit]

def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

def reshuffle(X, Y):
    NData = len(X)
    perm_indices = np.arange(NData)
    np.random.shuffle(perm_indices)
    X = X[perm_indices]
    Y = Y[perm_indices]
    return X, Y

def read_next_batch(X, Y, batch_size, batch_idx=0):
    NData = len(X)
    if batch_idx + batch_size < NData:
        X_batch = X[batch_idx:batch_idx+batch_size]
        Y_batch = Y[batch_idx:batch_idx+batch_size]
        batch_idx = batch_idx + batch_size
        return X_batch, Y_batch, batch_idx
    else:
        return None, None, None

def train(network, x_train, y_train, epochs = 100, learning_rate = 0.01, batch_size = 128, verbose = True):
    num_batches = 0
    for epoch in range(epochs):
        epoch_loss = 0
        x_train, y_train = reshuffle(x_train, y_train)
        batch_idx = 0
        x_batch, y_batch, batch_idx = read_next_batch(x_train, y_train, batch_size, batch_idx)
        while x_batch is not None:
            num_batches += 1
            for x, y in zip(x_batch, y_batch):
                # forward pass
                output = predict(network, x)
                # Epoch Loss/Error based on prediction
                # Use cross_entropy
                epoch_loss += loss_cross_entropy(y, output)
                # Use MSE
                #epoch_loss += mse(y, output)

                # backward Error Propagation
                grad = loss_cross_entropy_grad(y, output)
                #grad = mse_grad(y, output)

                for layer in reversed(network):
                    grad = layer.backward(grad, learning_rate)

            x_batch, y_batch, batch_idx = read_next_batch(x_train, y_train, batch_size, batch_idx)

            #epoch_loss /= len(x_train)
            epoch_loss /= num_batches
        if verbose:
            print(f"{epoch + 1}/{epochs}, error={epoch_loss}")
    print("\nEnd Of Training...")

##################################################################
###### Start Running the code from here

# load MNIST using Keras
# Select 1000 training samples and 20 test samples and appropriate preprocess them
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 1000)
x_test, y_test = preprocess_data(x_test, y_test, 20)

# Build the Neural Network Architecture
# Change the layer name and parameters appropriately to experiment with
network = [
    Dense(28 * 28, 50),
    #Tanh(),
    Sigmoid(),
    Dense(50, 10),
    #Tanh()
    Softmax()
]

# network = [
#     Dense(28 * 28, 128),
#     Sigmoid(),
#     Dense(128, 64),
#     Sigmoid(),
#     Dense(64, 10),
#     Softmax()
# ]

# network = [
#     Dense(28 * 28, 256),
#     Sigmoid(),
#     Dense(256, 128),
#     Sigmoid(),
#     Dense(128, 64),
#     Sigmoid(),
#     Dense(64, 10),
#     Softmax()
# ]



# train the network using the input data and stochastic Gradient Descent
# Define different learning rates, epochs and batch_size to experiment with
train(network, x_train, y_train, epochs=100, learning_rate=0.1, batch_size = 128)

# Evaluate performance on test data
print("\nStart Of Evaluation\n")
for x, y in zip(x_test, y_test):
    output = predict(network, x)
    print('pred:', np.argmax(output), '\ttrue:', np.argmax(y))

ratio = sum([np.argmax(y) == np.argmax(predict(network, x)) for x, y in zip(x_test, y_test)]) / len(x_test)
toterror = sum([mse(y, predict(network, x)) for x, y in zip(x_test, y_test)]) / len(x_test)
print('ratio: %.2f' % ratio)
print('mse: %.4f' % toterror)


# Plot 10 samples with their corresponding network prediction and true label
samples = 10
for test, true in zip(x_test[:samples], y_test[:samples]):
    image = np.reshape(test, (28, 28))
    plt.imshow(image, cmap='binary')
    pred = predict(network, test)
    idx = np.argmax(pred)
    idx_true = np.argmax(true)
    plt.title('pred: %s, prob: %.2f, true: %d' % (idx, pred[idx], idx_true))
    plt.show()
    #print('pred: %s, prob: %.2f, true: %d' % (idx, pred[idx], idx_true))