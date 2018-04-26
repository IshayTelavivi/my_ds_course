# -*- coding: utf-8 -*-
"""
The purpose of the this code is to generate the decision trees algorithm from scratch.it was given as an assignment
in our course. Recursion functions were not used.
For simplicity, I built it that it supports only continuous data (does not support categorical data or
numeric with is not continuous). This can be added.
Also It supports only  Gini impurity and not information gain. It can be added on later phase as a parameter.
Also, I did not separate the code into the decision trees algorithm and the actual testing (both in the same file).

Bottom line - the weighted f1 I got for the tested dataset was as good as the sklearn algorithm
"""
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

NUMERIC_CONDITION_DICT = {"a": "lower", "b": "higher"} # These are required for sorting numeric variabales
NUMBER_OF_POTENTIAL_THRESHOLDS_PER_NUMERIC_COLUMN = 5

class Node():
    """
    Node is any part of the tree (the classifier), whether it is the root Node, any split Node or a leaf Node
    """
    def __init__(self, name, n_observations, n_per_class_list, depth):
        self.name = name
        self.n_observations = n_observations # Total samples/observation in this node
        self.n_per_class_list = n_per_class_list # A list with the number of observation from each class
        self.depth = depth # The level of depth of the node (the 'root' node is level zero)


        self.gini = gini_calculation(n_per_class_list)
        self.children = {} # Here are the details of the child nodes (if relevant. leaf nodes do not have children)
        self.feature_for_split = None # If the node is split (if not a leaf node), it is split by this feature
        self.threshold = None # If the node is split (if not a leaf node), this is the threshold in the relevant feature
        self.leaf_class = None # If the node is a leaf node, the class of the leaf is stored here
        
    def __str__(self):
        print_line = 'Node name: {}\ngini = {}\nsamples = {}\nvalues = {}'
        return print_line.format(self.name, self.gini, self.n_observations, self.n_per_class_list)

    def add_child(self, name, feature, threshold, n_samples, n_per_class, condition):
        """
        The method adds the details of the next nodes (the nodes after split) to the seld.children dictionary
        at this point these nodes do not exist, only the details in the dictionary. These details are used to create the
        nodes later on
        :param name: composed of the feature, threshold and the condition
        :param feature: by which column the split was done to create the node
        :param threshold: by which threshold the split was done to create the node
        :param n_samples: number of samples (observations) from all classes
        :param n_per_class: a list with the number of samples from each class
        :param condition: whether the values in the feature are lower, higher (or equal for categorical) than the
                threshold
        :return: returns nothing
        """
        self.children[name] = {"n_observations" : n_samples,
                               "n_per_class" : n_per_class,
                               "col" : feature,
                               "thresh" : threshold, 
                               "condition" : condition} # condition is whether the child is lower than threshold or
                                                        # higher. May be also equal for categorical (not applicable now)

    def choose_feature_and_rule_and_return_data_for_children(self, X_df, y_series, list_of_unique):
        """
        The method takes the data of features and classed, iterate over the features with various thresholds
        and picks the best feature-threshold that gives the lowest gini impurity.
        Then it adds the details of the added nodes (children), and prepares the data for them
        :param X_df: a dataframe with the X independent variables
        :param y_series: a series with the dependent variable (classes)
        :param list_of_unique: all classes in the whole dataset
        :return: a dictionary whose keys are the child nodes' names and the values are the sub data-sets
        """
        logger.debug("len X_df is {}. len y_s is {}".format(len(X_df), len(y_series)))
        delta_gini = 0
        feature = None # Here the feature to split by is stored
        threshold = None # Here the threshold to split by is stored
        n_observation_in_samples = None # a list stores the no. of samples for the child nodes
        count_per_class = None # A list that takes two inner lists, each contains the number of samples from each class
        data_a = None # Data to be used by the child node, after filtering the current node with the relevant feature
        data_b = None # Data to be used by the child node, after filtering the current node with the relevant feature
        for col in X_df.columns:
            if is_numeric_dtype(X_df[col]):
                # create a df of both X columns and class column, to enable correct filtering
                df_temp = pd.concat([X_df, y_series], axis=1)
                # The follwing three lines creates pre-defined potential thresholds to be checked
                min_value = min(df_temp[col])
                max_value = max(df_temp[col])
                threshold_array = np.linspace(min_value, max_value, num=NUMBER_OF_POTENTIAL_THRESHOLDS_PER_NUMERIC_COLUMN)
                #will always be < first (a), >= second (b)
                for num in threshold_array[1:-1]: # For each one of the potential thresholds
                    sample_a = df_temp[df_temp[col] < num] # Only the rows where the feature value lower than threshold
                    sample_b = df_temp[df_temp[col] >= num] # Only the rows where the feature value higher than threshold
                    samples = (sample_a, sample_b)
                    list_of_gini_weight_of_samples = [] # Stores the gini for each potential node (out of two)
                    count_of_sample_list = [] # Number of samples per potential node (out of two)
                    count_per_class_per_sample = [] # I want to have the n_samples per class for each group
                    # By the terminology I used, "sample" means a group of samples. A potential node
                    for sample in samples:
                        sample_count = sample.shape[0] # The length of the DF
                        count_of_sample_list.append(sample_count)
                        count_per_class_list = count_values_per_class(sample[y_series.name], list_of_unique)
                        count_per_class_per_sample.append(count_per_class_list)
                        sample_gini = gini_calculation(count_per_class_list) # Calculates gini impurity
                        gini_weight = sample_count * sample_gini
                        list_of_gini_weight_of_samples.append(gini_weight)
                    weighted_gini = sum(list_of_gini_weight_of_samples) / self.n_observations
                    # If weighted gini is the lowest, then it captures all relevant information
                    if self.gini - weighted_gini > delta_gini:
                        delta_gini = self.gini - weighted_gini
                        feature = col
                        threshold = num
                        n_observation_in_samples = count_of_sample_list
                        count_per_class = count_per_class_per_sample
                        data_a = sample_a
                        data_b = sample_b
            else: # non-numeric
                pass
        self.feature_for_split = feature
        self.threshold = threshold
        # The following 4 lines create the sub-datasets for the child nodes. X needs to be separated from y
        X_df_a = data_a[X_df.columns]
        y_a = data_a[y_series.name]
        X_df_b = data_b[X_df.columns]
        y_b = data_b[y_series.name]
        child_names = [] # This to allow creating the dictionary with the sub-data
        # The following for loop adds details of children to the self.children dictionary (nodes are created yet)
        for child in ("a","b"):
            child_name = "{}-{}-{}".format(str(feature), str(threshold), NUMERIC_CONDITION_DICT[child])
            child_names.append(child_name)
            self.add_child(child_name, feature, threshold, n_observation_in_samples[("a","b").index(child)],
                           count_per_class[("a","b").index(child)],
                           NUMERIC_CONDITION_DICT[child])
        data_for_a = (X_df_a, y_a)
        data_for_b = (X_df_b, y_b)

        return {child_names[0]: data_for_a, child_names[1]: data_for_b}



class TreeClassifier():
    """
    The class is a decision tree classifier
    """
    def __init__(self, max_depth=5, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_sample_split = min_samples_split
        self.min_sample_leaf = min_samples_leaf

        self.nodes = {} # A dictionary of all the nodes. Key are names and values are the nodes
        self.leaf_node_list = [] # Stores only the leaf nodes (their names)
        self.depth = 0 # The maximum depth of the classifier (will be the the self.max_depth unless stops earlier)


    def fit(self, X_df, y_series):
        """
        The Method generates the tree for the specific data-set
        :param X_df: a dataframe with the X independent variables
        :param y_series: a series with the dependent variable (classes)
        :return: returns nothing
        """

        # Input validation conditions
        condition_list_for_validation = [len(X_df) == len(y_series), type(X_df) == pd.DataFrame,
                                         type(y_series) == pd.Series]
        condition_message_dict = {len(X_df) == len(y_series): "X_df and y_series have different length",
                                  type(X_df) == pd.DataFrame: "X_df is not a pd.DataFrame",
                                  type(y_series) == pd.Series: "y_series is not a pd.Series"}
        if all(condition_list_for_validation):
            # I need to get the number of each class for the whole dataset to instantiate the root node (4 next lines)
            class_value_count = y_series.value_counts()
            n_per_class_list = []
            for value in class_value_count:
                n_per_class_list.append(value)
            # Then instantiate the root node
            root_node = Node('root', len(X_df), n_per_class_list, 0)
            self.nodes[root_node.name] = root_node # Adding the root node to the nodes dictionary
            unique_class_values = list(y_series.unique())  # Stores the original class values in a list before all the
                                                           #  filtering. Some may be lost after filtering
            # The following lists stores tuples with the information for each node.
            # As long as there nodes in the list the splitting continues
            queue_of_nodes_to_classify = [(root_node, X_df, y_series)]
            while len(queue_of_nodes_to_classify) > 0:
                # Next lines are only for the next log
                names_in_queue = []
                for tup in queue_of_nodes_to_classify:
                    name = tup[0].name
                    names_in_queue.append(name)
                logger.debug("Queue of nodes: {}".format(names_in_queue))
                node_in_work = queue_of_nodes_to_classify[0]
                logger.info("node_in_work is {} with depth of {}".format(node_in_work[0].name, node_in_work[0].depth))
                condition_list_for_splitting = [self.min_sample_leaf < min(node_in_work[0].n_per_class_list),
                                                self.min_sample_split < node_in_work[0].n_observations,
                                                self.max_depth > node_in_work[0].depth]
                if all(condition_list_for_splitting):
                    # The following line uses the method of the Node class to find the feature
                    # and rule (threshold) by which the splitting is done, and returns the data for the child nodes
                    data_for_children = node_in_work[0].choose_feature_and_rule_and_return_data_for_children\
                        (node_in_work[1], node_in_work[2], unique_class_values)
                    logger.info("choose_feature_and_rule completed for {}".format(node_in_work[0].name))
                    for child_name, values in node_in_work[0].children.items():
                        # The Node.children dictionary contains the required information for creating the nodes
                        # Now instantiating child nodes
                        new_node = Node(child_name, values['n_observations'], values["n_per_class"],
                                        node_in_work[0].depth + 1)
                        if new_node.depth > self.depth:
                            self.depth = new_node.depth # Updating the classifier's depth
                        self.nodes[child_name] = new_node # Adding the new node to the self.nodes dictionary
                        logger.info("{} wes created".format(new_node.name))
                        tuple_of_node = (new_node, data_for_children[child_name][0], data_for_children[child_name][1])
                        queue_of_nodes_to_classify.append(tuple_of_node) # Appending the node's tuple to the queue
                else: # If did not meat one or more of the conditions
                    # The following line checking which class has the largest samples, and stores it's index
                    index_of_class = node_in_work[0].n_per_class_list.index(max(node_in_work[0].n_per_class_list))
                    # The using this index the get the correct class from the unique_class_values_list, where the
                    # original number and order of the class is stored
                    node_in_work[0].leaf_class = unique_class_values[index_of_class]
                    self.leaf_node_list.append(node_in_work[0].name) # Adds the node the the list of leaf nodes
                    #print(node_in_work[0].leaf_class)
                    logger.info("{} is a leaf with {}".format(node_in_work[0].name, node_in_work[0].leaf_class))
                queue_of_nodes_to_classify.pop(0) # Takes out the node's tuple, to allow ending the while loop
        else: # If does not meet one of the validation conditions, prints relevant message
            for condition in condition_list_for_validation:
                if condition is False:
                    print(condition_message_dict[condition])


    def predict(self, X_test):
        """
        The method examines a data-set of X independent variable, and for each row runs over the tree to find the
        predicted class (category) for that row
        :param X_test: a dataframe with the X independent variables to be tested
        :return: a series of predicted classes (categories) matching to the X_test rows
        """
        predicted_class_list = [] # Stores the class per row. WIll be masked to Series
        for index, row in X_test.iterrows():
            leaf = None
            node_temp_name = 'root'
            # For each row, the while loop ends when it reaches the relevant lead node and receives its class
            while leaf == None:
                if self.nodes[node_temp_name].leaf_class == None: # If the node is not a leaf node
                    feature_for_split = self.nodes[node_temp_name].feature_for_split # Finds the feature for splitting
                    if row[feature_for_split] < self.nodes[node_temp_name].threshold:
                        tip_condition = "lower"
                    elif row[feature_for_split] >= self.nodes[node_temp_name].threshold:
                        tip_condition = "higher"
                    else:
                        logger.error("Value should be lower, equal or greater then threshold. Check data")
                    # The following loop checks which node is the next one "to traval" to according to the condition
                    for key, inner_dict in self.nodes[node_temp_name].children.items():
                        if inner_dict["condition"] == tip_condition:
                            node_temp_name = key
                else: # If the node is a leaf node
                    leaf = self.nodes[node_temp_name].leaf_class # leaf is the class
                    predicted_class_list.append(leaf)

        return pd.Series(predicted_class_list)


def gini_calculation(list_per_class):
    """
    Takes the number of samples from each class (category) and finds the gini impurity
    :param list_per_class: a list with the number of samples from each class (category)
    :return: a float with the gini impurity
    """
    component_list = []
    for item in list_per_class:
        p = item/sum(list_per_class)
        component_list.append((p*(1-p)))
    return sum(component_list)


def count_values_per_class(series, list_of_unique):
    """
    The function takes a series, and count the number of values from each class (category). The idea of this method
    is to prevent a situation where I have zero values from a certain class and as a result I will get a list the lacks
    this class, e.g. I want to receive [0,25] and not [25]
    :param series: a series with the dependent variable / classes (categories)
    :param list_of_unique: all the class vaues that were in the whole data (before tha splitting)
    :return: a list with the number of values from each class
    """
    count_per_class_list = []
    for item in list_of_unique:
        n_from_class = series[series == item].count()
        count_per_class_list.append(n_from_class)
    return count_per_class_list



data = pd.read_csv('weight.txt')
logger.info("File is read")
X = data.loc[:, "Age" : "Weight"]
y = data.loc[:, "Sex"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train_main, X_eval, y_train_main, y_eval = train_test_split(X_train, y_train, test_size = 0.25, random_state=0)
print(len(X_train_main))
print(len(X_eval))
print(len(X_test))


my_classifier = TreeClassifier(max_depth=4)
my_classifier.fit(X_train, y_train)
pred = my_classifier.predict(X_test)

cr = classification_report(y_test, pred)
print(cr)




# #This is the code of the sklearn algorithm to compare with my code
#
# # Now I am running the model on the the train and examining various
# # hyperparameters on the evaluation data
# min_sam_lf = 0
# min_sam_sp = 0
# max_dep = 0
# f1_best = 0
# for dep in range(1, 20):
#     for mss in range(2, 20):
#         for msl in range(1, 10):
#             dessert_dt = DecisionTreeClassifier(max_depth=dep,
#                                                min_samples_split=mss,
#                                                min_samples_leaf=msl)
#             # fitting the model
#             dessert_dt.fit(X_train_main, y_train_main)
#
#             # Predicting the model on the evaluaiton data
#             pred = dessert_dt.predict(X_eval)
#
#             # Evaluation
#             #cm = confusion_matrix(y_eval, pred)
#             #a_score = accuracy_score(y_eval, pred)
#             f1 = f1_score(y_true=y_eval, y_pred=pred,pos_label='m')
#             if f1 > f1_best:
#                 f1_best = f1
#                 min_sam_lf = msl
#                 min_sam_sp = mss
#                 max_dep = dep
# print("max_depth: ", max_dep)
# print("min_sample_split: ", min_sam_sp)
# print("min_sample_leaf: ", min_sam_lf)
# print("best_f1: ", f1_best)
# print()
# # After I found the best hyperparameters on the validation data, it's time to run the test data
# weight_dt = DecisionTreeClassifier(max_depth=7, min_samples_split=2,
#                                     min_samples_leaf=3)
#
# weight_dt.fit(X_train, y_train)
#
# pred1 = weight_dt.predict(X_test)
# # Evaluation
# cr1 = classification_report(y_test, pred1)
# print(cr1)

# export_graphviz(decision_tree=weight_dt,
#                 out_file='algorithm_dev.dot',
#                 feature_names=X.columns,
#                 class_names=dessert_dt.classes_,
#                 leaves_parallel=True,
#                 filled=True,
#                 rotate=False,
#                 rounded=True)
# =============================================================================
