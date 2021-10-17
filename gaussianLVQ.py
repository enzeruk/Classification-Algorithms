# Pattern Recognition 2020: Assignment 1    |      Kalliri Angeliki     A.M. 2446 #
from numpy import *
import numpy as np


# Euclidean distance calculation
def euclidean_dist(x, y):
    return np.linalg.norm(x - y)


# We have a 2-category (areas) experiment, we calculate the initial values for their sample mean & variance
# The function returns the list with the info of each of the 2 areas, without the category column Y to be considered
# from the data set
def initial_areas(train_set):
    # Keep 2 lists with the examples that originally belong to each of the 2 categories
    list_category0 = []
    list_category1 = []
    areas_info = []         # Will store 3 info for each area: area_info = [mean, variance, category]
    lines = len(train_set)
    columns = len(train_set[0])
    for i in range(lines):
        # examples that belong to category 0 are added to the category0 list
        if train_set[i][columns - 1] == 0:
            list_category0.append(train_set[i])
        # examples that belong to category 1 are added to the category1 list
        else:
            list_category1.append(train_set[i])

    # Delete the Y column
    list_category0 = np.delete(list_category0, len(list_category0[0]) - 1, 1)
    list_category1 = np.delete(list_category1, len(list_category1[0]) - 1, 1)

    mean0 = np.mean(list_category0, axis=0)
    var0 = np.var(list_category0)
    y0 = 0
    area0 = [mean0, var0, y0]

    mean1 = np.mean(list_category1, axis=0)
    var1 = np.var(list_category1)
    y1 = 1
    area1 = [mean1, var1, y1]

    areas_info.append(area0)
    areas_info.append(area1)

    return areas_info


# ------------------------------------------------------------------------------------- #
# LVQ implementation (Gaussian areas) #
def gaussian_lvq(train_set, test_set):
    m = 2                   # a default value for the loop (given equal to 10, here M = 2)
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0

    lvq_metrics = []                                                # will keep the accuracy & f1 score
    y_test = test_set[:, len(test_set[0]) - 1]                      # keep the y column of the test set in an array
    test_data = np.delete(test_set, len(test_set[0]) - 1, 1)        # delete y column from the test_set

    # Initialization of the area list
    areas_list = initial_areas(train_set)
    areas_list_row = len(areas_list)
    areas_list_col = len(areas_list[0])

    y_train = train_set[:, len(train_set[0]) - 1]                   # keep the y column of the train set in an array
    train_data = np.delete(train_set, len(train_set[0])-1, 1)       # delete y column from the train_set
    variance = np.var(train_data)                                   # calculate the train's variance

    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    for iteration in range(m * len(train_data)):
        line_x = np.random.randint(len(train_data), size=1)     # choose a random line
        x = train_data[line_x, :]                               # choose the random example from the above chosen line
        # print("Random example =", x, "@ line:", line_x)

        results = []        # keep the results of the similarity function for each random example each time

        for i in range(0, areas_list_row):
            res_vector = np.square(x - areas_list[i][0])      # (|x - mj|)^2 step by step, math operations in python
            vector = np.sum(res_vector)                       # with numpy don't work as they should mathematically
            similarity_function = np.exp(-(vector / (2*areas_list[i][1])))
            results.append(similarity_function)

        win_area_index = results.index(max(results))     # the winner area position in areas_list (line)
        # similarity = max(results)
        # print("Winner area:", win_area_index, "with similarity =", similarity)

        real_category_x = y_train[line_x]          # find the real category that the random example belongs to
        # print("x belongs to category:", real_category_x)
        # print("C of the area is:", areas_list[win_area_index][areas_list_col - 1])

        # The winner's area category equals with the original category of the random example (success)
        # areas_list[win_area_index][areas_list_col - 1] : the 3d info in the areas_list, the category of the area, Cj
        if (areas_list[win_area_index][areas_list_col - 1]) == real_category_x:
            # print("SUCCESS.")
            areas_list[win_area_index][0] = (1 - 0.001) * areas_list[win_area_index][0] + 0.001 * x
            areas_list[win_area_index][1] = np.power(np.sqrt(areas_list[win_area_index][1]) +
                                                     + 0.001 * euclidean_dist(x, areas_list[win_area_index][0]), 2)

        # The winner's area category doesn't equal with the original category of the random example (failure)
        else:
            # print("FAILURE.")
            areas_list[win_area_index][0] = (1 - 0.001) * areas_list[win_area_index][0] - 0.001 * x
            areas_list[win_area_index][1] = np.power(np.sqrt(areas_list[win_area_index][1]) -
                                                     - 0.001 * euclidean_dist(x, areas_list[win_area_index][0]), 2)

            new_area = [x, variance * 0.1, real_category_x]
            # New area is created, areas_list must be updated
            areas_list.append(new_area)
            areas_list_row = len(areas_list)
            areas_list_col = len(areas_list[0])
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # Confusion matrix creation (using the test set with the given areas that were created from the training set)
    for line in range(0, len(test_data)):
        test_results = []  # keep the results of the similarity function for each example in the test set each time
        x_test = test_data[line, :]         # x_test: an example from the test set

        for j in range(0, areas_list_row):
            test_res_vector = np.square(x_test - areas_list[j][0])      # (|x - mj|)^2 step by step,math operations
            test_vector = np.sum(test_res_vector)               # with numpy don't work as they should mathematically
            test_similarity_function = np.exp(-(test_vector / (2 * areas_list[j][1])))
            test_results.append(test_similarity_function)

        test_win_area_index = test_results.index(max(test_results))  # the winner area position in areas_list (line)
        test_similarity = max(test_results)
        # print("Winner area:", test_win_area_index, "with similarity =", test_similarity)

        test_real_category_x = y_test[line]  # find the real category that the example x, from the test set, belongs to
        # print("x belongs to category:", test_real_category_x)
        # print("C of the area is:", areas_list[test_win_area_index][areas_list_col - 1])

        # test_real_category_x : the true category of x in the test set
        # areas_list[test_win_area_index][areas_list_col - 1] : the 'predicted' category for x, of the winner region
        if (test_real_category_x == 0) and (areas_list[test_win_area_index][areas_list_col - 1] == 0):
            true_positive = true_positive + 1
        elif (test_real_category_x == 0) and (areas_list[test_win_area_index][areas_list_col - 1] == 1):
            false_positive = false_positive + 1
        elif (test_real_category_x == 1) and (areas_list[test_win_area_index][areas_list_col - 1] == 0):
            false_negative = false_negative + 1
        else:
            true_negative = true_negative + 1

    positives_negatives = true_positive + true_negative + false_positive + false_negative

    # Metrics calculation
    accuracy = (true_positive + true_negative) / positives_negatives
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1score = 2 * ((precision * recall) / (precision + recall))
    lvq_metrics.append(accuracy)
    lvq_metrics.append(f1score)

    areas_list = []         # empty the area list for the next fold

    return lvq_metrics
