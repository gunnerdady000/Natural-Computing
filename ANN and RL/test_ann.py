import pandas as pd, numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from ANN import ANN2


def main():
    print("For testing the ANN")
    # For testing trained net
    data = pd.read_csv("testdata.dat")
    # put [0:1] into X array
    x = data.iloc[0:38, [0, 1]].values
    # change 1 or 2 into (0,1) or (1,0) into Y array
    y = data.iloc[0:38, 2].values

    # For training
    data2 = pd.read_csv("datafile.dat")
    x2 = data2.iloc[0:238, [0, 1]].values
    y2 = data2.iloc[0:238, 2].values

    # plot the data sets
    # plot data sets
    plot(x2, y2)
    plot(x, y)
    # subtract down to 0 or 1
    y = y - 1
    y2 = y2 - 1

    # one hot trick from https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-1-hot-encoded-numpy-array
    y = np.eye(np.max(y) + 1, dtype=float)[y]
    y2 = np.eye(np.max(y2) + 1, dtype=float)[y2]


    # number of nodes for hidden layers
    layers = np.array([22])

    # number of iterations
    niter = [100, 1000, 2000, 3000]
    # number of learning rates
    rl = [0.1, 0.01, 0.001, 0.0001]
    # empty list of the accuracy
    accuracy = []
    for n in niter:
        for learn in rl:
            net = ANN2(2, 2, layers, n, learn)
            net.train2(x2, y2)
            yhat = net.predict2(x)
            print("Using: ", n, learn)
            print(net.accuracy(y, yhat))
            accuracy.append(net.accuracy(y, yhat))
    # print all of the accuracies
    print(accuracy)
    # testing prediction of the training data set
    net = ANN2(2, 2, layers, 5000, 0.01)
    net.train2(x2, y2)
    yhat = net.predict2(x2)
    print(net.accuracy(y2, yhat))
    plot(x2, net.convert(yhat))
    # testing prediction of the testing data set
    net.train2(x, y)
    yhat = net.predict2(x)
    print(net.accuracy(y, yhat))
    plot(x, net.convert(yhat))
    # proper testing method
    net.train2(x2, y2)
    yhat = net.predict2(x)
    print(net.accuracy(y, yhat))
    plot(x, net.convert(yhat))
    # inverted testing method
    net.train2(x, y)
    yhat = net.predict2(x2)
    print(net.accuracy(y2, yhat))
    plot(x2, net.convert(yhat))

def plot(x, y):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')

    # create a np array
    cmap = ListedColormap(colors[:len(np.unique(y))])
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=x[y == cl, 0], y=x[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)
    plt.show()


if __name__ == "__main__":
    main()

