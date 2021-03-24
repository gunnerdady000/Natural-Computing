import numpy as np
import matplotlib.pyplot as plt
import time


class ANN(object):

    # Initialize class
    def __init__(self, inputs=2, outputs=2, layers=None, niter=10, learning_rate=0.1):
        # create empty layer and checker
        self.is_empty = False

        if layers is None:
            layers = np.array([])
            self.is_empty = True

        # holds number of inputs
        self.inputs = inputs

        # holds number of outputs
        self.outputs = outputs

        # ndarray of integers each value being the number of nodes in a hidden layer
        self.layers = layers

        # learning rate
        self.learning_rate = learning_rate

        # number of iterations
        self.nither = niter

    # Train the net by finding weights
    def train(self, X, Y):
        # get the input values
        self.X = X
        # get the output values
        self.Y = Y

    # Create weights based off of the layers ndarray
    def weight_creation(self):
        # array of arrays to hold the weights
        self.weights = []

        # loop is 1 to end of list
        if not self.is_empty:
            # first weight is inputs and layers[0]
            weight_n = np.random.rand(self.layers[0], self.inputs)

            # append to list
            self.weights.append(weight_n)

            # 1 until the one before the end of the array
            for ind in range(1, self.layers.size):
                # current list will n-1 and n
                weight_n = np.random.rand(self.layers[ind], self.layers[ind-1])
                self.weights.append(weight_n)

            # last layer is n-layers by outputs
            weight_n = np.random.rand(self.outputs, self.layers[self.layers.size - 1])
            self.weights.append(weight_n)

            # array of biases that correspond to their respective weights
            self.b = np.random.rand(self.layers.size, 1)

            # hold A's
            self.A = np.zeros(self.layers.size - 1)

            # hold the Z's
            self.Z = np.zeros(self.layers.size)
        else:
            # only one layer results in a single weight input by output
            self.weights = np.random.rand(self.inputs, self.outputs)

            # only one bias point
            self.b = np.random.rand(1)

            # only one A
            self.A = np.zeros(1)

            # two Z's
            self.Z = np.zeros(2)

    # Predict using trained weights
    def predict(self, X):
        pass

    # Forward Propagation
    def forward_prop(self):
        # start with n = 0
        self.Z[0] = np.dot(self.X.T, self.weights[0]) + self.b[0]

        # Zn+1 = X dot Wn+1 + bn+1
        if not self.is_empty:
            # A0 is relu of Z0
            self.A[0] = self.relu(self.Z[0])
            # Zn = An-1 dot W[n] + b[n]
            Zn = np.dot(self.A[0], self.weights[1])

            for n in range(1, self.layers.size):
                Zn = self.X.dot(self.weights[n]) + self.b[n]
                # relu

        # end with n = size - 1

    # Calculate summation
    def summation(self, x, w, b):
        # Zn = X dot Wn + bn
        return np.dot(x, w) + b

    # Calculate using sigmoid
    def sigmoid(self, Z):
        return 1.0 / (1 + np.exp(-Z))

    # Derivative of sigmoid
    def def_sigmoid(self, Z):
        return self.sigmoid(Z) * (1 - self.sigmoid(Z))

    # Calculate using relu
    def relu(self, Z):
        return np.maximum(0, Z)

    # Derivative of the relu
    def back_relu(self, x):
        return np.where(x <= 0, 0, 1)

    # Cost function
    def cost(self, y, y_predict):
        pass
