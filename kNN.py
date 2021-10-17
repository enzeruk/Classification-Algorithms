# Pattern Recognition 2020: Assignment 1    |      Kalliri Angeliki     A.M. 2446 #
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


# Classification algorithm 1: Nearest Neighbor k-NN with Euclidean distance #
# train_set & test_set, that are passed, keep the Y category info
def knn_algorithm(train_set, test_set, k):
    knn_metrics = []
    # Store the last column (Y category)
    y_train = train_set[:, len(train_set[0]) - 1]
    y_test = test_set[:, len(test_set[0]) - 1]

    # Delete the columns from the sets
    x_train = np.delete(train_set, len(train_set[0]) - 1, 1)
    x_test = np.delete(test_set, len(test_set[0]) - 1, 1)

    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn.fit(x_train, y_train)               # Train the model to train_set

    y_pred = knn.predict(x_test)            # Compute the prediction y on the test_set (x samples)

    # Compute confusion_matrix = [TP FP]
    #                            [FN TN]
    matrix = np.array(confusion_matrix(y_test, y_pred))
    # print("Confusion matrix (knn):\n", matrix)

    # Metrics calculation
    accuracy = (matrix[0][0] + matrix[1][1]) / np.sum(matrix)
    precision = matrix[0][0] / (matrix[0][0] + matrix[0][1])        # Precision = TP / TP + FP
    recall = matrix[0][0] / (matrix[0][0] + matrix[1][0])           # Recall = TP / TP + FN
    f1score = 2 * ((precision * recall) / (precision + recall))
    knn_metrics.append(accuracy)
    knn_metrics.append(f1score)

    return knn_metrics
