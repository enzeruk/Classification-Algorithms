# Pattern Recognition 2020: Assignment 1    |      Kalliri Aggeliki     A.M. 2446 #
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC


# Classification algorithm 3: Support Vector Machines (SVM) #
# train_set & test_set, that are passed, keep the Y category info
def svm_algorithm(train_set, test_set):
    svm_metrics = []
    # Store the last column (Y category)
    y_train = train_set[:, len(train_set[0]) - 1]
    y_test = test_set[:, len(test_set[0]) - 1]

    # Delete the columns from the sets
    x_train = np.delete(train_set, len(train_set[0]) - 1, 1)
    x_test = np.delete(test_set, len(test_set[0]) - 1, 1)

    svc = SVC(kernel='linear')
    svc.fit(x_train, y_train)
    y_pred = svc.predict(x_test)

    # Compute confusion_matrix = [TP FP]
    #                            [FN TN]
    matrix = np.array(confusion_matrix(y_test, y_pred))

    # Metrics calculation
    accuracy = (matrix[0][0] + matrix[1][1]) / np.sum(matrix)
    precision = matrix[0][0] / (matrix[0][0] + matrix[0][1])    # Precision = TP / TP + FP
    recall = matrix[0][0] / (matrix[0][0] + matrix[1][0])       # Recall = TP / TP + FN
    f1score = 2 * ((precision * recall) / (precision + recall))
    svm_metrics.append(accuracy)
    svm_metrics.append(f1score)

    return svm_metrics
