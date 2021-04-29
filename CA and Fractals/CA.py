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

    def laplace(self, images=False):
        # loop through and create initial conditions
        for i in range(1, self.number_cells - 1):
            self.image[i, self.number_cells - 1] = i * (self.number_cells - 1 - i)

        # check for multiple image output
        if images:
            plt.show()
            plt.ion()
            plt.figure()

        # begin partial diff
        for t in range(self.niter):
            for i in range(1, self.number_cells - 1):
                for j in range(1, self.number_cells - 1):
                    self.buffer[i, j] = (self.image[i - 1, j] + self.image[i + 1, j] + self.image[i, j - 1] +
                                         self.image[i, j + 1]) / 4.0

            # move buffer state into image state
            for i in range(1, self.number_cells - 1):
                for j in range(1, self.number_cells - 1):
                    self.image[i, j] = self.buffer[i, j]

            # maybe pictures
            if images:
                self.graph(images)

        # close multiple images
        if images:
            plt.ioff()
            plt.close()

    def graph(self, images=False):
        if images:
            plt.clf()

        plt.imshow(self.image, cmap=plt.cm.get_cmap("gray"), interpolation='nearest')
        if images:
            plt.pause(0.01)
        else:
            plt.show()
