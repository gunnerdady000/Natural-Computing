import numpy as np
import matplotlib.pyplot as plt
import copy
from matplotlib.animation import FuncAnimation


class HillClimbing (object):

    # Initialize Class
    def __init__(self, rate=0.01, niter=100):
        self.niter = niter # number of iterations
        self.rate = rate # used to hold the learning rate

    # Climbing for problem 1
    def fit(self):
        # create the loop variable k and x setting both to zero
        k = 0
        x = 0

        # graph initial point
        self.k = k
        self.maximum = x
        self.graph_max(x, 0)

        # use animation if short enough
        if self.niter < 30000:
            plt.show()
            plt.ion()
            plt.figure()

        # work our way through the climb until we hit max amount of user iterations
        for k in range(self.niter):
            self.cur_iter = k

            x_prime = x + np.random.random() * self.rate

            # if f(x_prime) > f(x)
            if self.evaulate(x_prime) > self.evaulate(x):
                x = x_prime
                # get the iteration that it took to get the value
                self.k = k
                if self.niter < 30000:
                    self.graph_max(x, 1)

        # end animation
        if self.niter < 30000:
            plt.ioff()
            plt.close()

        # get the maximum that was found
        self.maximum = x

        # Final graph
        self.graph_max(x, 0)

    def evaulate(self, x):
        return np.power(2, -2 * np.power((x-0.1)/0.9, 2)) * np.power(np.sin(5*np.pi*x), 6)

    def graph_max(self, max, skip):
        x = np.arange(0, 1, 0.0001)
        y = np.power(2, -2 * np.power((x-0.1)/0.9, 2)) * np.power(np.sin(5*np.pi*x), 6)

        # used ot plot animation
        if skip == 1:
            plt.clf()

        plt.plot(x, y, label=f'Current Iteration {self.k} and maximum {max} ')
        plt.plot(max, self.evaulate(max), 'rx')
        plt.xlabel('X-value')
        plt.ylabel('f(x)-value')
        plt.title('Hill Climbing')
        plt.legend(loc='best')

        if skip == 1:
            plt.pause(0.01)
        else:
            plt.show()



class SimulatedAnnealing (object):

    # Initialize Class
    def __init__(self, rate=0.01, niter=1000, Trate=0.999, Tint=3000, Tfin=0.001):
        self.rate = rate # learning rate
        self.niter = niter  # number of iterations
        self.Trate = Trate # the are at which T is augmented
        self.Tint = Tint  # used to hold the initial temperature
        self.Tfin = Tfin # used to hold the final temperature value
        self.k = 0 # used to hold the current iteration

    def maximize(self):
        # pick random starting location
        x = np.random.uniform(0, 1)
        # used to show animation
        graphing = False
        if self.niter < 3000:
            graphing = True
        # need to change x if it is equal to 1 to stay within bounds
        if x > 0.9:
            x = x / 2 # just decided to cut it in half

        # set k = 0
        k = 0

        # graph initial point
        self.k = k
        self.maximum = x
        self.Tcurr = self.Tint
        self.graph_max(x, 0)

        # create empty arrays that will hold the values k-iterations and the corresponding x-values
        x_list = np.empty(0)
        k_list = np.empty(0)


        if graphing == True:
            plt.show()
            plt.ion()
            plt.figure()

        while (k < self.niter) and (self.Tcurr < self.Tfin):
            # use normal distribution to "randomly" move around
            x_prime = x + np.random.normal() * self.rate

            # Could not get rid of run time error without this -_-
            xp = ((self.evaluate(x) - self.evaluate(x_prime))/self.Tcurr)
            xp = np.round(xp, 6)
            uni = np.round(np.random.uniform(0, 1), 6)

            if np.exp(-xp) > uni:
                if 1 > x_prime > 0:
                    # x = x_prime
                    x = x_prime

                    # put graph here
                    self.k = k
                    if graphing == True:
                        self.graph_max(x, 1)

            if self.niter < 3000:
                x_list = np.append(x_list, x)
                k_list = np.append(k_list, k)

            # k++
            k += 1

            # change the rate of the T slowly
            self.Tcurr *= self.Trate

        if graphing == True:
            plt.ioff()
            plt.close()

        self.maximum = x

        self.graph_max(x, 0)

        if self.niter < 3000:
            self.graph_iterations(x_list, k_list)

    def minimize(self):
        # set k to 0
        self.k = 0
        graphing = False
        # set T to initial Temperature
        self.Tcurr = self.Tint

        # create array of random points
        x = np.random.randint(50, size=(50, 2))
        #x = np.vstack((x,x[0]))
        self.graph_TPS(x, 0)

        # Used to get show animation
        if graphing == True:
            plt.show()
            plt.ion()
            plt.figure()

        while self.k < self.niter and not(self.Tcurr < self.Tfin):
            # select the best of the different random tours
            x_prime = self.rand_select(x, int(x.size/4))

            # This is the minimizing function used correctly :)
            if np.random.uniform(0, 1) < np.exp((self.distance(x)-self.distance(x_prime))/self.Tcurr):
                if self.distance(x_prime) < self.distance(x):
                    x = x_prime
                    min = self.distance(x)
                    min_k = self.k
                    if graphing == True:
                        self.graph_TPS(x, 1)

            #print(self.k)
            # k++
            self.k += 1

            # T *= rate
            self.Tcurr *= self.Trate
        if graphing == True:
            plt.ioff()
            plt.close()

        print("Current Temp: ", self.Tcurr)
        print("Number of max iterations: ", self.k)
        print("Best distance: ", self.distance(x))
        self.graph_TPS(x, 0)
        self.k = min_k
        self.graph_TPS(x, 0)

    def evaluate(self, x):
        return np.power(2, -2 * np.power((x - 0.1) / 0.9, 2)) * np.power(np.sin(5 * np.pi * x), 6)

    def rand_select(self, x, half):
        # shuffle x randomly :)
        x_a = x[np.random.choice(x.shape[0], half*2, replace=False), :]
        # swap two columns
        x_b = x.copy()
        indone = np.random.randint(0, half)
        indtwo = np.random.randint(0, half)
        temp = x_b[indone]
        x_b[indone] = x[indtwo]
        x_b[indtwo] = x[indone]
        # invert back half of list
        x_c = x.copy()
        x_c[half:] = x_c[half:][::-1]
        # invert random sub lists
        x_d = x.copy()
        start = np.random.randint(0, half)
        end = half + start
        if end >= x_d.size:
            end -= 1
        x_d[start:end] = x_d[start:end][::-1]

        distances = np.array([self.distance(x_a), self.distance(x_b), self.distance(x_c), self.distance(x_d)])

        min_distance = np.where(distances == np.amin(distances))
        min_distance = min_distance[0][0]
        #min_distance = 1
        #print(min_distance)
        if min_distance == 0:
            x_prime = x_a
            #print("Picked A")
        elif min_distance == 1:
            x_prime = x_b
            #print("Picked B")
        elif min_distance == 2:
            x_prime = x_c
            #print("Picked C")
        else:
            x_prime = x_d
            #print("Picked D")
        return x_prime

    def distance(self, city_list):
        # matrix of matrices. Numpy is great :)
        return np.sum(np.sqrt(np.square(city_list[1:, 0] - city_list[:int(city_list.size/2)-1, 0]) + np.square(city_list[1:, 1] - city_list[:int(city_list.size/2)-1, 1])))

    def graph_TPS(self, city_list, skip):
        # Fix matrix mess
        x_array, y_array = np.hsplit(city_list, 2)
        x_array = np.concatenate(x_array, axis=0)
        x_array = np.append(x_array, x_array[0]).tolist()
        y_array = np.concatenate(y_array, axis=0)
        y_array = np.append(y_array, y_array[0]).tolist()

        # turn on animation
        if skip == 1:
            plt.clf()

        # plot cities
        plt.plot(x_array, y_array, color='red', label=f'Current Iteration {self.k} and min {self.distance(city_list)}')
        plt.legend(loc='best')
        plt.scatter(x_array, y_array, marker='o')

        # plot numbers for cities
        for i in range(len(x_array)-1):
            plt.text(x_array[i] * (1+0.02), y_array[i] * (1+0.02), i, fontsize=12)
        plt.xlabel("City Location in x-direction")
        plt.ylabel("City Location in y-direction")
        plt.title('TSP using SA')
        if skip == 1:
            plt.pause(0.01)
        else:
            plt.show()

    def graph_max(self, max, skip):
        # arrays used to show results
        x = np.arange(0, 1, 0.0001)
        y = np.power(2, -2 * np.power((x - 0.1) / 0.9, 2)) * np.power(np.sin(5 * np.pi * x), 6)

        # used ot plot animation
        if skip == 1:
            plt.clf()

        # plotting stuff
        plt.plot(x, y, label=f'Current Iteration {self.k} and maximum {round(max, 5)} at Temp {round(self.Tcurr, 5)}')
        plt.plot(max, self.evaluate(max), 'rx')
        plt.legend(loc='best')

        if skip == 1:
            plt.pause(0.01)
        else:
            plt.show()

    def graph_iterations(self, x_list, k_list):
        plt.plot(k_list, x_list)
        plt.title('Number of Iterations vs X values')
        plt.ylabel('X Values')
        plt.xlabel("Number of k-iterations")
        plt.show()
        plt.close()