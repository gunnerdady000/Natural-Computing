import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from CA import cellar_automa


class CaNn(object):
    def __init__(self, size):
        # The grid we are training to predict
        self.size = size

        # Keras stuff for the neural net
        self.neural_net = None

        self.loss_history = None

        # Scale factor for values
        self.val_scale = 100

    """
    Initialize the neural net.
    """
    def init(self):
        # Create the Neural net
        self.neural_net = keras.models.Sequential()
        self.neural_net.add(layers.Dense(8, input_dim=2, kernel_initializer='normal', activation='relu'))
        self.neural_net.add(layers.Dense(20, kernel_initializer='normal', activation='softmax'))
        self.neural_net.add(layers.Dense(8, kernel_initializer='normal', activation='softmax'))
        self.neural_net.add(layers.Dense(1, kernel_initializer='normal', activation='relu'))

        # Compile model
        self.neural_net.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

    """
    Train the neural net using the given points and known values.
    """
    def train(self, points, targets, verbose=2):
        targets = targets / self.val_scale
        self.loss_history = self.neural_net.fit(
            points,
            targets,
            validation_data=(points, targets),
            batch_size=20,
            epochs=400,
            shuffle=True,
            verbose=verbose)

    """
    Get the predicted grid value at the location from the NN.
    """
    def get_val(self, x, y):
        loc = tf.expand_dims(tf.convert_to_tensor([x,y]),0)
        return tf.squeeze(self.neural_net(loc)).numpy() * self.val_scale

    """
    Generate a grid using the NN.
    """
    def generate_grid(self):
        # Generate the list of points
        points = []
        for x in range(self.size):
            for y in range(self.size):
                points.append([x,y])

        # Predict the values at the points using the NN
        grid = tf.squeeze(self.neural_net.predict(tf.convert_to_tensor(np.array(points)))).numpy()
        return grid.reshape(self.size,self.size) * self.val_scale

    """
    Generate and plot a grid using the NN.
    """
    def graph_grid(self, wait=True):
        # Construct the grid
        grid = self.generate_grid()

        # Plot the results
        plt.clf()
        plt.title("NN Modeled Results")
        plt.imshow(grid, cmap=plt.cm.get_cmap("gray"), interpolation='nearest')
        if (not wait):
            plt.draw()
            plt.pause(0.25)  # brief wait
        else:
            plt.show()

    """
    Plot the loss value as a function of epoch during training.
    """
    def graph_loss(self):
        # plot the loss function over the epochs
        plt.plot(self.loss_history.history['loss'])
        plt.title('Model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.show()

"""
Return a grid of points from a grid of the given size. The number of points will be point_count^2.
"""
def dist_points(grid_size, point_count):
    locs = np.round(np.linspace(0,grid_size-1,point_count))
    pts = []
    for x in locs:
        for y in locs:
            pts.append([x,y])

    return np.array(pts).astype(int)

def main():
    grid_size = 20

    # Create the grid and solve with CA
    known_grid = cellar_automa(grid_size, 150)
    known_grid.laplace()
    known_grid.graph()
    known_grid.graph_converge("CA Convergence")

    # Generate a distribution of points and corresponding values
    points = dist_points(grid_size, int(grid_size/2))
    values = known_grid.image[points[:,0], points[:,1]]

    # Plot the points
    plt.imshow(known_grid.image, cmap=plt.cm.get_cmap("gray"), interpolation='nearest')
    plt.scatter(points[:,0], points[:,1], marker='o', color="blue", )
    plt.title('Training Data Locations')
    plt.show()

    # Create and train the NN
    neural_net = CaNn(grid_size)
    neural_net.init()
    neural_net.train(points, values)

    # Plot the loss function
    neural_net.graph_loss()

    # Plot the results
    neural_net.graph_grid()

    # Plot the error
    plt.clf()
    plt.title("NN Model Error")
    error = np.abs(known_grid.image-neural_net.generate_grid())
    plt.imshow(error, cmap=plt.cm.get_cmap("Reds"), interpolation='nearest')
    plt.show()


if __name__ == '__main__':
    main()