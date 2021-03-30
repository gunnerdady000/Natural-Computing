import numpy as np
import matplotlib.pyplot as plt
import time


class ANN(object):

    # Initialize class
    def __init__(self, inputs=2, outputs=2, layers=None, niter=10, learning_rate=0.1, eta=0.0000000001):
        # add input and output to layers array
        if layers is None:
            self.layers = np.array(inputs)
            np.append(self.layers, outputs)
        else:
            self.layers = np.array(inputs)
            self.layers = np.append(self.layers, layers)
            self.layers = np.append(self.layers, outputs)


        # n = layers - input and output
        self.n = self.layers.size - 1

        # holds number of inputs
        self.inputs = inputs

        # holds number of outputs
        self.outputs = outputs

        # learning rate
        self.learning_rate = learning_rate

        # number of iterations
        self.nither = niter

        # eta value
        self.eta = eta

    # Train the net to destroy the human race
    def train(self, X, Y):
        # get the input values
        self.X = X

        # get the output values
        self.Y = Y

        # create weights
        self.weight_creation()

        # create error/loss array
        self.error = []

        # create a variable to see if we are done training
        check = 0

        for i in range(self.nither):
            # forward prop
            self.forward_prop()

            # back prop
            self.backward_prop(self.ypredict)

            # collect losses and check to see if loss is acceptable
            self.error.append(self.loss)

            # maybe make this user input for accuracy
            if self.loss < 0.05:
                # double check that we did not get lucky
                check +=1
                # exit if we meet the accuracy twice as that should be good to go
                if check == 2:
                    return


    # Create weights based off of the layers ndarray
    def weight_creation(self):
        # array of arrays to hold the weights, Z, and A
        self.weights = []
        self.b = []
        self.Z = []
        self.A = []

        # weights
        for n in range(self.n):
            # Wn = rand[layer[n], layer[n+1]]
            Wn = np.random.rand(self.layers[n], self.layers[n+1])
            # bn random value
            bn = np.random.rand(self.layers[n+1],)

            # add to array of arrays
            self.weights.append(Wn)
            self.b.append(bn)

        # Z
        for n in range(self.n):
            # Zn = zeros[layers[n], layers[n+1]]
            Zn = np.zeros((self.layers[n], self.layers[n+1]))

            # add to array of arrays
            self.Z.append(Zn)

        # A
        for n in range(self.n):
            # An = zeros[layers[n]]
            An = np.zeros(self.layers[n])

            # add to array of arrays
            self.A.append(An)

    # Predict using trained weights
    def predict(self, X):
        # will have to change the values or edit forward prop
        self.X = X

        # forward prop
        self.forward_prop()

        # figure out rounding

    # Forward Propagation, should have made the X's the first weight...
    def forward_prop(self):
        # Z[0] = X dot W[0] + b[0]
        self.Z[0] = np.dot(self.X, self.weights[0]) + self.b[0]
        self.A[0] = self.relu(self.Z[0])

    # Zn = An dot W[n] + b[n]
        for n in range(1, self.n):
            self.Z[n] = np.dot(self.A[n-1], self.weights[n]) + self.b[n]

            # relu
            self.A[n] = self.relu(self.Z[n])

        # end with n - 1
        self.ypredict = self.A[self.n - 1]

        # calculate loss
        self.loss = self.cost(self.ypredict)
        print(self.loss)

    # Backward Propagation
    def backward_prop(self, y_predict):
        # why me?
        inv_y = 1.0 - self.Y

        # invert output prediction
        inv_ypredict = 1.0 - y_predict

        # delta of whys
        dl_ypred = np.divide(inv_y, self.eta_calc(inv_ypredict)) - np.divide(self.Y, self.eta_calc(y_predict))

        # delta sig
        dl_sig = y_predict * (inv_ypredict)

        # Z[n] =  dl_ypred * dl_sig
        dl_Zn = dl_ypred * dl_sig

        # now for n - count, going down in index hence backwards... this will break
        for n in reversed(range(self.n)):
            # find dlA = dl_Zn dot Wn.T
            dl_An = np.dot(dl_Zn, self.weights[n].T)

            # dl_Wn and update Wn, dl_Wn = An-1 dot Wn.T ,Wn = Wn - rate * dl_Wn
            dl_Wn = np.dot(self.A[n-1].T, dl_Zn)
            self.weights[n] = self.weights[n] - self.learning_rate * dl_Wn

            # find dl_bn and update bn, dl_bn = sum(dl_zn), bn = bn - rate * dl_bn
            dl_bn = np.sum(dl_Zn, axis=0, keepdims=True)
            self.b[n] = self.b[n] - self.learning_rate * dl_bn

            # dl_Zn = dl_An * dl_relu(Zn)
            dl_Zn = dl_An * self.back_relu(self.Z[n-1])

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

    # ETA calc
    def eta_calc(self, Z):
        return np.maximum(Z, self.eta)

    # Cost function
    def cost(self, y_predict):
        # invert test output
        inv_y = 1.0 - self.Y

        # invert output prediction
        inv_ypredict = 1.0 - y_predict

        # update the prediction
        y_predict = self.eta_calc(y_predict)

        # update the inverted prediction
        inv_ypredict = self.eta_calc(inv_ypredict)

        # return the loss
        return -1/np.size(self.Y, 0) * (np.sum(np.multiply(np.log(y_predict), self.Y) + np.multiply(inv_y, np.log(inv_ypredict))))