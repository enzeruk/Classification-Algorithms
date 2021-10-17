# Pattern Recognition 2020: Assignment 1    |      Kalliri Angeliki     A.M. 2446 #
# TODO: Change the paths @ lines 10 & 16
from readDataFile import read_file
from readXSLFile import read_cv
from kfold import k_fold


# Experiment 1: spambase (4.000 data) #
print("Experiment 1.")
path1 = r'C:\Users\Desktop\spambase.data'
dataset1 = read_file(path1)
k_fold(dataset1)

# Experiment 2: credit card clients (10.000 out of 30.000 data) #
print("Experiment 2.")
path2 = r'C:\Users\Desktop\default of credit card clients.xls'
dataset2 = read_cv(path2)
k_fold(dataset2)

# ------------------------------------------------------------------------------------- #


