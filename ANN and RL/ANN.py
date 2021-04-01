import numpy as np
import matplotlib.pyplot as plt
import time

class ANN2(object):

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

    # Gave up do it the easy way
    def train2(self, X, Y):
            # Gather X
            self.X = X

            # Gather Y
            self.Y = Y

            # Create error array
            self.error = []

            # create weights
            self.weight_2()

            # create check variable
            check = 0

            self.loss = 1
            # run through iterations
            for i in range(self.nither):
                # forward prop
                self.forward_prop2()

                # compute loss sum((a2 - y)^2) / N
                self.loss = (np.square(self.a2 - self.Y).sum()) / np.size(self.Y, 0)

                # back prop
                self.back_prop2()

                # error list add
                self.error.append(self.loss)

                # check to see if we have mastered the data set
                if self.loss < 0.01:
                    check += 1
                    if check == 2:
                        return

    # Here we Go :(
    def forward_prop2(self):
        # z1 = X, w1
        self.z1 = np.dot(self.X, self.w1)

        # use sigmoid function
        self.a1 = self.sigmoid(self.z1)

        # z2 = w2, a1
        self.z2 = np.dot(self.a1, self.w2)

        # sigmoid function
        self.a2 = self.sigmoid(self.z2)

        # collect the predicted y
        self.ypredict = self.a2

    # just for the homework
    def back_prop2(self):
        err = 2.0 * (self.a2 - self.Y)
        grad_z2 = self.def_sigmoid(self.a2) * err

        grad_a1 = np.dot(grad_z2, self.w1)
        grad_z1 = self.def_sigmoid(self.a1) * grad_a1

        grad_w2 = np.dot(self.a1.T, grad_z2)
        grad_w1 = np.dot(self.X.T, grad_z1)

        # update weights
        self.w1 -= self.learning_rate * grad_w1
        self.w2 -= self.learning_rate * grad_w2

    # weight 2 function
    def weight_2(self):
        # create random weights
        self.w1 = np.random.randn(self.layers[0], self.layers[1])
        self.w2 = np.random.rand(self.layers[1], self.layers[2])

    # Predict 2
    def predict2(self, X):
        self.X = X
        self.forward_prop2()
        return np.round(self.ypredict)

    # Calculate summation
    def summation(self, x, w, b):
        # Zn = X dot Wn + bn
        return np.dot(x, w) + b

    # Calculate using sigmoid
    def sigmoid(self, Z):
        return 1.0 / (1 + np.exp(-Z))

    # Derivative of sigmoid
    def def_sigmoid(self, Z):
        return Z * (1.0 - Z)

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

    def accuracy(self, Y, yhat):
        return round((Y == yhat).sum()/float(Y.size) * 100, 3)

    def convert(self, Y):
        output = Y[:, 1]
        return output.astype('int')


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
                Wn = np.random.rand(self.layers[n], self.layers[n + 1])
                # bn random value
                bn = np.random.rand(self.layers[n + 1], )

                # add to array of arrays
                self.weights.append(Wn)
                self.b.append(bn)

            # Z
            for n in range(self.n):
                # Zn = zeros[layers[n], layers[n+1]]
                Zn = np.zeros((self.layers[n], self.layers[n + 1]))

                # add to array of arrays
                self.Z.append(Zn)

            # A
            for n in range(self.n):
                # An = zeros[layers[n]]
                An = np.zeros(self.layers[n])

                # add to array of arrays
                self.A.append(An)

        # Predict using trained weights for general

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
                self.Z[n] = np.dot(self.A[n - 1], self.weights[n]) + self.b[n]

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
                dl_Wn = np.dot(self.A[n - 1].T, dl_Zn)
                self.weights[n] = self.weights[n] - self.learning_rate * dl_Wn

                # find dl_bn and update bn, dl_bn = sum(dl_zn), bn = bn - rate * dl_bn
                dl_bn = np.sum(dl_Zn, axis=0, keepdims=True)
                self.b[n] = self.b[n] - self.learning_rate * dl_bn

                # dl_Zn = dl_An * dl_relu(Zn)
                dl_Zn = dl_An * self.back_relu(self.Z[n - 1])

        # Calculate summation

    def summation(self, x, w, b):
            # Zn = X dot Wn + bn
            return np.dot(x, w) + b

        # Calculate using sigmoid
    def sigmoid(self, Z):
            return 1.0 / (1 + np.exp(-Z))

        # Derivative of sigmoid
    def def_sigmoid(self, Z):
            return Z * (1.0 - Z)

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
            return -1 / np.size(self.Y, 0) * (
                np.sum(np.multiply(np.log(y_predict), self.Y) + np.multiply(inv_y, np.log(inv_ypredict))))

    def accuracy(self, Y, yhat):
            return round((Y == yhat).sum() / float(Y.size) * 100, 3)

    def convert(self, Y):
            output = Y[:, 1]
            return output.astype('int')