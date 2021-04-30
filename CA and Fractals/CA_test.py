import numpy as np, matplotlib as plt
from CA import cellar_automa


def main():
    # test with 25 cells and 150 iterations
    cell = cellar_automa(20, 150)
    cell.laplace()
    cell.graph()
    cell.graph_converge()
    # test with 125 cells and 800 iterations
    cell = cellar_automa(125, 800)
    cell.laplace()
    cell.graph()
    cell.graph_converge()

if __name__ == "__main__":
    main()
