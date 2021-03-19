import numpy as np, matplotlib.pyplot as plt
from Homework1 import HillClimbing, SimulatedAnnealing


def main():
    #hill()
    #maxim()
    min()


def hill():
    hill = HillClimbing(0.01, 10000)
    hill.fit()
    print("Max of : ", hill.evaulate(hill.maximum))
    print("Given the X value of: ", hill.maximum)
    print("Found at iteration: ", hill.k)


def maxim():
    akneel = SimulatedAnnealing(0.1, 2999, 0.999, 0.001, 3000)
    akneel.maximize()
    print("Max of : ", akneel.evaluate(akneel.maximum))
    print("Given the X value of: ", akneel.maximum)
    print("Found at iteration: ", akneel.k)
    print("tempature max is: ", akneel.Tcurr)

def min():
    akneel = SimulatedAnnealing(0.01, 300000, 0.9999, 300000, 0.00001)
    akneel.minimize()
    print("Done")

if __name__ == "__main__":
    main()