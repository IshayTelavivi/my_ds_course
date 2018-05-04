"""
The purpose of the this code is to generate the K-NN classifier algorithm from scratch.
For simplicity, I built it that it supports only "euclidean" and "manhattan" metrics, as well as an external
function. (since I used only these metrics, It supports only numeric values)
Bottom line - the performance is similar to the sklearn KNeighborsClassifier, however much slower (so I added the
tqdm indication).
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from pandas.api.types import is_numeric_dtype
from tqdm import tqdm

# First there are the two distance functions

def euclidean_distance(point_A, point_B):
    """
    The function calculates the euclidean distance
    :param point_A: an iterable object with numeric values
    :param point_B: an iterable object with numeric values
    :return: a float of the euclidean distance
    """
    # The following line in the input validation
    assert len(point_A) == len(point_B), "Point do not have the same number of elements."
    sum_of_squares = 0 # This list stores the squares of each value couple
    for i in range(len(point_A)):
        squared = (point_A[i] - point_B[i])**2
        sum_of_squares += squared
    return np.sqrt(sum_of_squares)


def manhattan_distance(point_A, point_B):
    """
    The function calculates the manhattan distance
    :param point_A: an iterable object with numeric values
    :param point_B: an iterable object with numeric values
    :return: a float of the manhattan distance
    """
    # The following line in the input validation
    assert len(point_A) == len(point_B), "Points do not have the same number of elements."
    sum_of_delta = 0 # This variable collects the delta of each value couple
    for i in range(len(point_A)):
        feature_delta = point_A[i] - point_B[i]
        sum_of_delta += feature_delta
    return sum_of_delta

# The following dictionary allows the choice of the metric used
DISTANCE_FUNCTION_DICT = {'euclidean' : euclidean_distance, 'manhattan': manhattan_distance}

class K_nearest_neighbor():
    def __init__(self, n_neighbors=5, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric

        self.X = None # Stores the training X data
        self.y = None # Stores the training y data
        self.classes = None # a series with the unique classes that are in the training y data

    # The algorithm is lazy, so fit do not do too much, but I did find the need to read and keep the data in the
    # class object
    def fit(self, X_train, y_train):
        """
        The algorithm is lazy, so the fit function do not do too much, but I did find the need to read and keep the
        data in the class object, and get the classes (for the probability)
        :param X_train: a dataframe with the independent variables
        :param y_train: a series with the dependent variable (classes)
        :return: nothing
        """
        # Input validation 1: make sure the datasets are in the same length
        assert len(X_train) == len(y_train), "Datasets do not have the same length."
        # Input validation 2: make sure the values are numeric
        for column in X_train.columns:
            assert is_numeric_dtype(X_train[column]), "{} in not numeric.".format(X_train[column].name)
        self.X = X_train
        self.y = y_train
        self.classes = y_train.unique() # Makes the series of the unique classes


    def predict_basic(self, X_test):
        """
        The method takes data-set of X independent variable, and for each row runs over the training data, calculates
        the distance between the rows, and takes the closest k rows. Then it takes the class of these rows and generates
        a series.
        In addition it creates an array of the probability of each sample per class
        :param X_test: a dataframe with the X independent variables to be tested
        :return: a series of the predicted class (to be used in self.predict) and an array of probabilities (to be
        used in self.predict_proba)
        """
        prediction_list = [] # This is where the predicted class per sample is stored
        probability_list = [] # This is where the nested lists of probability per class are stored
        for index, test_row in tqdm(X_test.iterrows()):
            distances_and_classes_per_test_row_list = [] # Here the tuples of distance and class per train_row is
                                                         # stored
            for index, train_row in self.X.iterrows():
                # The following if statement is to allow using an external metric function
                # In case the metric is one of the built-in:
                if self.metric in DISTANCE_FUNCTION_DICT.keys():
                    distance_between_points = DISTANCE_FUNCTION_DICT[self.metric](test_row, train_row)
                # In case an external function is used:
                else:
                    distance_between_points = self.metric(test_row, train_row)
                row_class = self.y[index] # takes the matching class from the y data
                distances_and_classes_per_test_row_list.append((distance_between_points, row_class))
            # The list of tuples is sorted to allow taking the smallest distances
            distances_and_classes_per_test_row_list.sort(key=lambda tup: tup[0])
            class_list_for_test_row = [] # For each row this list collects the k class values
            for i in range(self.n_neighbors):
                class_list_for_test_row.append(distances_and_classes_per_test_row_list[i][1])
            # The following line takes the most prevalent class from the list
            predicted_class = max(set(class_list_for_test_row), key=class_list_for_test_row.count)
            prediction_list.append(predicted_class)
            # The following line compute the probability per class per row (sample)
            proba_for_test_row = [class_list_for_test_row.count(cls) / len(class_list_for_test_row) for cls in self.classes]
            probability_list.append(proba_for_test_row)

        probability_array = np.array(probability_list)
        return pd.Series(prediction_list), probability_array


    def predict(self, X_test):
        """
        The method takes the series of prediction from the self.predict_basic method
        :param X_test: a dataframe with the X independent variables to be tested
        :return: a series of the predicted class
        """
        return self.predict_basic(X_test)[0]


    def predict_proba(self, X_test):
        """
        The method takes the array of probabilities from the self.predict_basic method
        :param X_test: a dataframe with the X independent variables to be tested
        :return: an array of probabilities
        """
        return self.predict_basic(X_test)[1]


# # This is the code I tested my algorithm
# knn = K_nearest_neighbor()
# knn.fit(X_train, y_train)
# my_pred = knn.predict(X_test)
# cr = classification_report(y_test, my_pred)
# print(cr)
# prob_knn = pd.DataFrame(knn.predict_proba(X_test))
# print(prob_knn.head())
#
# # This is the sklearn for comparison
# classifier = KNeighborsClassifier()
# classifier.fit(X_train, y_train)
# pred = classifier.predict(X_test)
# cr1 = classification_report(y_test, pred)
# print(cr1)