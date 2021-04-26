import numpy as np, matplotlib as plt
from CA import cellar_automa


def main():
    # test with 25 cells and 125 iterations
    cell = cellar_automa()
    cell.laplace(True)
    cell.graph()
    # test with 125 cells and 1000 iterations
    cell = cellar_automa(125, 1000)
    cell.laplace()
    cell.graph()

if __name__ == "__main__":
    main()
