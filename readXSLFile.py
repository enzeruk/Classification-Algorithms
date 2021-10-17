# Pattern Recognition 2020: Assignment 1    |      Kalliri Aggeliki     A.M. 2446 #
from numpy import *
import pandas as pd
import numpy as np


# Read the data from the excel file
def read_cv(path):
    skip_col = [0]    # read the excel file skipping the columns that are not needed for calculation (ID column)
    columns = [i for i in range(25) if i not in skip_col]
    info = pd.read_excel(path, usecols=columns, skiprows=1, skipfooter=20000)   # 10000/30000 data will be read

    dataset = array(info)  # cast the data we read from the excel as array

    for col in range(len(dataset[0]) - 1):
        max_value = max(dataset[:, col])
        for i in range(0, len(dataset)):
            if dataset[i, col] < 0:
                dataset[i, col] = np.abs(dataset[i, col])
            if max_value > 100:
                dataset[i, col] = dataset[i, col] / max_value

    return dataset
