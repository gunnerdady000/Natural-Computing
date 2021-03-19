import numpy as np, matplotlib.pyplot as plt
from Homework1 import HillClimbing, SimulatedAnnealing
from Homework2 import Mutation

def main():
    mut = Mutation(20, 150, 300)
    mut.int_population(1, 50, 1, 50, False)
    mut.generate()

if __name__ == "__main__":
    main()