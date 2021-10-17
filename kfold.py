# Pattern Recognition 2020: Assignment 1    |      Kalliri Aggeliki     A.M. 2446 #
from numpy import *
from sklearn.model_selection import KFold
import numpy as np
from gaussianLVQ import gaussian_lvq
from kNN import knn_algorithm
from oneLayerGD import nn1_gd_algorithm
from twoLayersGD import nn2_gd_algorithm
from oneLayerSGD import nn1_sgd_algorithm
from twoLayersSGD import nn2_sgd_algorithm
from svm import svm_algorithm
from naiveBayes import bayes_algorithm

# 10-folds cross validation
def k_fold(dataset):
    i = 1
    splits = 10
    kfold = KFold(n_splits=splits)  # KFold(n_splits=splits)
    lvq_results_list = []  # These lists will store the metrics of each algorithm for each fold
    knn_results_list = []
    nn1_gd_results_list = []
    nn2_gd_results_list = []
    nn1_sgd_results_list = []
    nn2_sgd_results_list = []
    svm_results_list = []
    bayes_results_list = []

    neighbors = int(input("k-NN algorithm implementation: Select the number of neighbors: "))
    k = int(input("Neural Network with 1 hidden layer (using GD or SGD): Select the number of neurons: "))
    k1 = int(input("Neural Network with 2 hidden layers (using GD or SGD): Select the number of neurons @ 1st layer: "))
    k2 = int(input("Neural Network with 2 hidden layers (using GD or SGD): Select the number of neurons @ 2d layer: "))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    for train_set, test_set in kfold.split(dataset):
        print("Creating the fold #", i, sep="")

        print("LVQ (Gaussian) processing...")
        lvq_metrics = gaussian_lvq(dataset[train_set], dataset[test_set])
        lvq_results_list.append(lvq_metrics)

        print("k-NN (Euclidean distance) processing...")
        knn_metrics = knn_algorithm(dataset[train_set], dataset[test_set], neighbors)
        knn_results_list.append(knn_metrics)

        print("Neural Network (1 layer & Gradient Descent) processing...")
        nn1_gd_metrics = nn1_gd_algorithm(dataset[train_set], dataset[test_set], k)
        nn1_gd_results_list.append(nn1_gd_metrics)

        print("Neural Network (2 layers & Gradient Descent) processing...")
        nn2_gd_metrics = nn2_gd_algorithm(dataset[train_set], dataset[test_set], k1, k2)
        nn2_gd_results_list.append(nn2_gd_metrics)

        print("Neural Network (1 layer & Stohastic Gradient Descent) processing...")
        nn1_sgd_metrics = nn1_sgd_algorithm(dataset[train_set], dataset[test_set], k)
        nn1_sgd_results_list.append(nn1_sgd_metrics)

        print("Neural Network (2 layers & Stohastic Gradient Descent) processing...")
        nn2_sgd_metrics = nn2_sgd_algorithm(dataset[train_set], dataset[test_set], k1, k2)
        nn2_sgd_results_list.append(nn2_sgd_metrics)

        print("SVM processing...")
        svm_metrics = svm_algorithm(dataset[train_set], dataset[test_set])
        svm_results_list.append(svm_metrics)

        print("Naive Bayes processing...")
        bayes_metrics = bayes_algorithm(dataset[train_set], dataset[test_set])
        bayes_results_list.append(bayes_metrics)

        i = i + 1
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # --- Start of prints --- #
    # LVQ (with Gauss) algorithm results
    lvq_results_array = np.array(lvq_results_list)
    lvq_overall_array = np.sum(lvq_results_array, axis=0)
    lvq_overall_accuracy = lvq_overall_array[0] / len(lvq_results_list)
    lvq_overall_f1score = lvq_overall_array[1] / len(lvq_results_list)
    print("\n\nLVQ @ ", splits, "-folds metrics:", sep="")
    print("Accuracy = ", '%.2f' % (lvq_overall_accuracy * 100), "%", sep="")
    print("F1 Score = ", '%.2f' % (lvq_overall_f1score * 100), "%", sep="")

    # k-NN classification algorithm results
    knn_results_array = np.array(knn_results_list)
    knn_overall_array = np.sum(knn_results_array, axis=0)
    knn_overall_accuracy = knn_overall_array[0] / len(knn_results_list)
    knn_overall_f1score = knn_overall_array[1] / len(knn_results_list)
    print("\n\nK-NN @ ", splits, "-folds metrics:", sep="")
    print("Accuracy = ", '%.2f' % (knn_overall_accuracy * 100), "%", sep="")
    print("F1 Score = ", '%.2f' % (knn_overall_f1score * 100), "%", sep="")

    # Neural Network (1 layer & GD) algorithm results
    nn1_gd_results_array = np.array(nn1_gd_results_list)
    nn1_gd_overall_array = np.sum(nn1_gd_results_array, axis=0)
    nn1_gd_overall_accuracy = nn1_gd_overall_array[0] / len(nn1_gd_results_list)
    nn1_gd_overall_f1score = nn1_gd_overall_array[1] / len(nn1_gd_results_list)
    print("\n\nNeural Network (1 layer & GD) @ ", splits, "-folds metrics:", sep="")
    print("Accuracy = ", '%.2f' % (nn1_gd_overall_accuracy * 100), "%", sep="")
    print("F1 Score = ", '%.2f' % (nn1_gd_overall_f1score * 100), "%", sep="")

    # Neural Network (2 layers & GD) algorithm results
    nn2_gd_results_array = np.array(nn2_gd_results_list)
    nn2_gd_overall_array = np.sum(nn2_gd_results_array, axis=0)
    nn2_gd_overall_accuracy = nn2_gd_overall_array[0] / len(nn2_gd_results_list)
    nn2_gd_overall_f1score = nn2_gd_overall_array[1] / len(nn2_gd_results_list)
    print("\n\nNeural Network (2 layers & GD) @ ", splits, "-folds metrics:", sep="")
    print("Accuracy = ", '%.2f' % (nn2_gd_overall_accuracy * 100), "%", sep="")
    print("F1 Score = ", '%.2f' % (nn2_gd_overall_f1score * 100), "%", sep="")

    # Neural Network (1 layer & SGD) algorithm results
    nn1_sgd_results_array = np.array(nn1_sgd_results_list)
    nn1_sgd_overall_array = np.sum(nn1_sgd_results_array, axis=0)
    nn1_sgd_overall_accuracy = nn1_sgd_overall_array[0] / len(nn1_sgd_results_list)
    nn1_sgd_overall_f1score = nn1_sgd_overall_array[1] / len(nn1_sgd_results_list)
    print("\n\nNeural Network (1 layer & SGD) @ ", splits, "-folds metrics:", sep="")
    print("Accuracy = ", '%.2f' % (nn1_sgd_overall_accuracy * 100), "%", sep="")
    print("F1 Score = ", '%.2f' % (nn1_sgd_overall_f1score * 100), "%", sep="")

    # Neural Network (2 layers & SGD) algorithm results
    nn2_sgd_results_array = np.array(nn2_sgd_results_list)
    nn2_sgd_overall_array = np.sum(nn2_sgd_results_array, axis=0)
    nn2_sgd_overall_accuracy = nn2_sgd_overall_array[0] / len(nn2_sgd_results_list)
    nn2_sgd_overall_f1score = nn2_sgd_overall_array[1] / len(nn2_sgd_results_list)
    print("\n\nNeural Network (2 layers & SGD) @ ", splits, "-folds metrics:", sep="")
    print("Accuracy = ", '%.2f' % (nn2_sgd_overall_accuracy * 100), "%", sep="")
    print("F1 Score = ", '%.2f' % (nn2_sgd_overall_f1score * 100), "%", sep="")

    # SVM classification algorithm results
    svm_results_array = np.array(svm_results_list)
    svm_overall_array = np.sum(svm_results_array, axis=0)
    svm_overall_accuracy = svm_overall_array[0] / len(svm_results_list)
    svm_overall_f1score = svm_overall_array[1] / len(svm_results_list)
    print("\n\nSVM @ ", splits, "-folds metrics:", sep="")
    print("Accuracy = ", '%.2f' % (svm_overall_accuracy * 100), "%", sep="")
    print("F1 Score = ", '%.2f' % (svm_overall_f1score * 100), "%", sep="")

    # Naive Bayes classification algorithm results
    bayes_results_array = np.array(bayes_results_list)
    bayes_overall_array = np.sum(bayes_results_array, axis=0)
    bayes_overall_accuracy = bayes_overall_array[0] / len(bayes_results_list)
    bayes_overall_f1score = bayes_overall_array[1] / len(bayes_results_list)
    print("\n\nNaive Bayes @ ", splits, "-folds metrics:", sep="")
    print("Accuracy = ", '%.2f' % (bayes_overall_accuracy * 100), "%", sep="")
    print("F1 Score = ", '%.2f' % (bayes_overall_f1score * 100), "%", sep="")
    # --- End of prints --- #
