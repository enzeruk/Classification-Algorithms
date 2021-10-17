# Pattern Recognition 2020: Assignment 1    |      Kalliri Aggeliki     A.M. 2446 #
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier


# Classification algorithm 2a: Neural Network with 1 hidden layer & Stohastic Gradient Descent (SGD) #
# train_set & test_set, that are passed, keep the Y category info
def nn1_sgd_algorithm(train_set, test_set, k):
    nn1_sgd_metrics = []
    # Store the last column (Y category)
    y_train = train_set[:, len(train_set[0]) - 1]
    y_test = test_set[:, len(test_set[0]) - 1]

    # Delete the columns from the sets
    x_train = np.delete(train_set, len(train_set[0]) - 1, 1)
    x_test = np.delete(test_set, len(test_set[0]) - 1, 1)

    mlp = MLPClassifier(activation='logistic', solver='sgd', hidden_layer_sizes=(k, ), batch_size=1)
    mlp.fit(x_train, y_train)
    y_pred = mlp.predict(x_test)

    # Compute confusion_matrix = [TP FP]
    #                            [FN TN]
    matrix = np.array(confusion_matrix(y_test, y_pred))

    # Metrics calculation
    accuracy = (matrix[0][0] + matrix[1][1]) / np.sum(matrix)
    precision = matrix[0][0] / (matrix[0][0] + matrix[0][1])    # Precision = TP / TP + FP
    recall = matrix[0][0] / (matrix[0][0] + matrix[1][0])       # Recall = TP / TP + FN
    f1score = 2 * ((precision * recall) / (precision + recall))
    nn1_sgd_metrics.append(accuracy)
    nn1_sgd_metrics.append(f1score)

    return nn1_sgd_metrics
