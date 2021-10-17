# Pattern Recognition 2020: Assignment 1    |      Kalliri Aggeliki     A.M. 2446 #
from numpy import *
import numpy as np

# Read the data from the .data file
def read_file(path):
    dataset = np.genfromtxt(path, delimiter=',', dtype=int)

    for col in range(len(dataset[0]) - 1):
        max_value = max(dataset[:, col])
        for i in range(0, len(dataset)):
            if max_value > 100:
                dataset[i, col] = dataset[i, col] / max_value

    np.random.shuffle(dataset)

    return dataset
