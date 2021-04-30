import numpy as np
import matplotlib.pyplot as plt


class cellar_automa(object):

    def __init__(self, number_cells=25, niter=125):
        # number of cells within bounds
        self.number_cells = number_cells
        # number of iterations
        self.niter = niter
        # creates output image
        self.image = np.zeros((self.number_cells, self.number_cells))
        # used as a buffer image as everything changes
        self.buffer = np.zeros((self.number_cells, self.number_cells))
        # Track change amounts across the grid
        self.adjustments = np.zeros(niter)

    def laplace(self, images=False):
        # loop through and create initial conditions
        for i in range(1, self.number_cells - 1):
            self.image[i, self.number_cells - 1] = i * (self.number_cells - 1 - i)

        # check for multiple image output
        if images:
            plt.show()
            plt.ion()
            plt.figure()

        self.buffer = np.copy(self.image)

        print("\rRunning CA: 0%", end="")

        # begin partial diff
        for t in range(self.niter):
            for i in range(1, self.number_cells - 1):
                for j in range(1, self.number_cells - 1):
                    self.buffer[i, j] = (self.image[i - 1, j] + self.image[i + 1, j] + self.image[i, j - 1] +
                                         self.image[i, j + 1]) / 4.0

            # Calculate the average delta for the cells
            self.adjustments[t] = np.abs(np.average(self.image - self.buffer))

            # move buffer state into image state
            self.image = np.copy(self.buffer)

            # maybe pictures
            if images:
                self.graph(images)

            print('\rRunning CA: {:.2f}%  '.format(t / self.niter * 100), end="")

        print('\rRunning CA: 100%  ')

        # close multiple images
        if images:
            plt.ioff()
            plt.close()

    def graph(self, images=False, title="CA Results"):
        if images:
            plt.clf()

        plt.title(title)
        plt.imshow(self.image, cmap=plt.cm.get_cmap("gray"), interpolation='nearest')
        if images:
            plt.pause(0.01)
        else:
            plt.show()

    def graph_converge(self, title="Convergence"):
        plt.clf()
        plt.title(title)
        plt.xlabel("Iteration")
        plt.ylabel("Cell Change")
        plt.plot(self.adjustments)
        plt.show()
