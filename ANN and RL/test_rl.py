from unittest import TestCase
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import RL as RL


def main():
    print("For testing the RL")


if __name__ == "__main__":
    main()


class TestOUActionNoise(TestCase):
    def setUp(self):
        pass

    def test_looks_good(self):
        noise_gen = RL.OUActionNoise(mean=np.zeros(1), std_deviation=float(0.25) * np.ones(1))
        values = [noise_gen.generate() for i in range(10000)]
        sns.distplot(values)
        plt.show()
