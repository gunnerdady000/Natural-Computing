import numpy as np
import matplotlib.pyplot as plt
import time


class Mutation (object):

    # Initialize class
    def __init__(self, popsize=10, itemsize=10, generation=100):

        # holds the population per generation
        self.popsize = popsize

        # holds the size of how many items are within the knapsack
        self.itemsize = itemsize

        # holds the number of generations
        self.generation = generation

        # holds an empty ndarray for the fitted population
        self.fitted_list = np.array([])

    def int_population(self, min_weight, max_weight, min_value, max_value, animation=False):
        # create the population size using either 0 or 1 based off of population size and the number of items
        self.population = np.random.randint(2, size=(self.popsize, self.itemsize))

        # creates the item wights
        self.w = np.random.randint(min_weight, max_weight, size=self.itemsize)

        # creates the item values
        self.v = np.random.randint(min_value, max_value, size=self.itemsize)

        # make sure that the W is less than sum of w
        self.W = int(np.sum(self.w) / 2)

        # used to animate the bar graph
        self.animation = animation

        # used only to compare actual value verses the GA
        self.max_value = 0

    def weightfit(self, index):
        # had a feeling this works and it does... I think I did something like this in ML or I have transcend spacetime
        return np.sum(np.where(self.population[index] == 1, self.w, 0))

    def valuefit(self, index):
        # returns the value of the total value of the items
        return np.sum(np.where(self.population[index] == 1, self.v, 0))

    def popfit(self, current_generation):
        # create empty lists for weights and values
        weight_list = np.array([], dtype=int)
        value_list = np.array([], dtype=int)

        for i in range(0, self.popsize):
            # get the populations total value
            weight = self.weightfit(i)
            value = self.valuefit(i)

            # gather the wights if the total if less than W else 0
            if weight < self.W:
                weight_list = np.append(weight_list, weight)
                value_list = np.append(value_list, value)
            else:
                weight_list = np.append(weight_list, 0)
                value_list = np.append(value_list, 0)

        # create the fitted population array
        index = np.arange(weight_list.size)
        self.fitted_list = np.vstack((weight_list, value_list, index)).T

        # to animate or not to animate
        if self.animation:
            self.barchart(current_generation, self.animation)

        # figured out how to change the argsort from max to min given this website
        # https://www.kite.com/python/answers/how-to-use-numpy-argsort-in-descending-order-in-python
        # column 1 has the value of the objects and it what we want to sort by
        self.fitted_list = self.fitted_list[np.argsort(-1*self.fitted_list[:, 1])]

    def generate(self):
        max = np.array([])
        tic1 = time.perf_counter()
        self.max_value = self.dynamic(self.W, self.w, self.v, self.itemsize)
        toc1 = time.perf_counter()

        if self.animation:
            plt.show()
            plt.ion()
            plt.figure()
            plt.ylim(top=np.sum(self.w))

        tic2 = time.perf_counter()

        for i in range (self.generation):
            # create an array that keeps the fitness of each iteration
            # calculate the fitness of the current population
            self.popfit(i)
            if i == 0:
                max = self.fitted_list[0]
            else:
                max = np.vstack((max, self.fitted_list[0]))
            # mutate the children given the parents
            self.mutate()
        toc2 = time.perf_counter()
        if self.animation:
            plt.ioff()
            plt.close()

        self.barchart(self.generation, False)

        # graph
        self.graph(max)
        print("Dnyamic time: ", toc1 - tic1)
        print("Evolution time: ", toc2 - tic2)
        final_value = self.fitted_list[0][1]
        final_value = int(final_value)
        print(f"EA percent error: {round((final_value/self.max_value*100),4)} %")

    def roulette(self, fitted):
        # start at 0
        n = 0

        # get the total value
        total = np.sum(fitted)

        # get a random value between 0 and 1
        prob = np.random.rand() #* self.mutation_rate

        # start finding
        sum = fitted[0][1] / total
        index = 0
        # while sum < prob, increase n and sum
        while sum < prob:
            n += 1
            index = n % self.popsize
            wait = fitted[index][1] / total
            sum += wait

        # return the
        return index

    def mutate(self):
        # find parents
        dad = 0 # this seems to yields the best results
        mom = self.roulette(self.fitted_list)

        # combine mom and dad for parents
        self.recomb(dad, mom)

        # loop through each of the population, while keeping parents
        for i in range(2, self.popsize):
            # use random integer and take the mod using the item size, or number of bits
            bit_location = np.random.randint(0, self.itemsize) % self.itemsize

            # switch random child's bit
            self.population[i][bit_location] = not(self.population[i][bit_location])

    def recomb(self, dad, mom):
        # move dad and mom to the top of the population list [0, 1]
        dad_sel = self.fitted_list[dad][2]
        dad_pop = self.population[dad_sel]
        mom_sel = self.fitted_list[mom][2]
        mom_pop = self.population[mom_sel]
        self.population[0] = dad_pop
        self.population[1] = mom_pop

        # find random location for children
        for i in range (2, self.popsize):
            # random integer % number of bits
            location = np.random.randint(0, self.itemsize) % self.itemsize

            # a child is born into pain
            child = np.hstack((self.population[0][:location], self.population[1][location:]))

            # set child to the combination of the parents
            self.population[i] = child

    def graph(self, max):
        weight, value, postion = np.hsplit(max, 3)
        x = np.concatenate(value, axis=0)
        y = np.arange(self.generation)
        plt.plot(y, x)
        plt.ylabel('Largest Value')
        plt.xlabel('Number of Generations')
        plt.title('Best choice over generational change')
        plt.show()

    def barchart(self, current_generation, animation):
        # make labels the only way I know how, but it took like 20 minutes to figure this simple thing out
        labels = [str(i) for i in range(self.popsize)]
        labels = ['P' + i for i in labels]

        # empty lists for weights and values
        weight = []
        value = []

        # graph stuff
        if animation:
            plt.clf()

        # limit the height of the graph and make a max weight line
        limit = np.sum(self.w)
        plt.ylim(0, limit)
        plt.axhline(y=self.W, color='b', linestyle='--', alpha=0.5, label=f'Weight Limit {self.W}')

        # graph regions, red = bad, green = good
        plt.axhspan(0, self.W, facecolor='g', alpha=0.3)
        plt.axhspan(self.W, limit, facecolor='r', alpha=0.3)

        # graph each population
        for i in range(self.popsize):
            # find the weights for each population
            weight = np.where(self.population[i] == 1, self.w, 0)

            # find the values for each population
            value = np.where(self.population[i] == 1, self.v, 0)

            # find the total height of the weight and value
            total_value = np.sum(value)

            # check to see if the weight invalidates the item value
            if np.sum(weight) > self.W:
                total_value = 0

            # add in the total value for each
            labels[i] = labels[i] + '\n' + str(total_value)

            # output each of the item as a stacked bar
            for idx in range(weight.size):
                plt.bar(labels[i], weight[idx], bottom=np.sum(weight[:idx]))

        # label stuff
        plt.ylabel('Weight')
        plt.xlabel('Total Value per population')
        title = 'Current Generation ' + str(current_generation) + ', Max Value: ' + str(self.max_value)
        plt.title(title)
        plt.legend()

        if animation:
            plt.pause(0.01)
        else:
            plt.show()

    # code from https://www.geeksforgeeks.org/python-program-for-dynamic-programming-set-10-0-1-knapsack-problem/ by
    # Bhavya Jain, this code is only used to provide the exact answer and for comparison against the genetic algorithm
    def dynamic(self, w, wt, val, n):
        k = [[0 for x in range(w +1)] for x in range (n + 1)]

        for i in range(n+1):
            for W in range(w+1):
                if i == 0 or W == 0:
                    k[i][W] = 0
                elif wt[i-1] <= W:
                    k[i][W] = max(val[i-1] + k[i-1][W-wt[i-1]], k[i-1][W])
                else:
                     k[i][W] = k[i-1][W]
        return k[n][w]
