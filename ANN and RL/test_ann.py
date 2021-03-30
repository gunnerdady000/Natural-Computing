import pandas as pd, numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from ANN import ANN

def main():
    print("For testing the ANN")
    data = pd.read_csv("testdata.dat")
    # put [0:1] into X array
    x = data.iloc[0:38, [0, 1]].values
    # change 1 or 2 into (0,1) or (1,0) into Y array
    y = data.iloc[0:38, 2].values

    # subtract down to 0 or 1
    y = y - 1
    # plot
    #plot(x,y)

    # one hot trick from https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-1-hot-encoded-numpy-array
    y = np.eye(np.max(y) + 1, dtype=float)[y]

    layers = np.array([2, 3, 4])
    net = ANN(2, 2, layers, 10, 0.1)
    net.train(x, y)
    print(net.weights[0])

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
