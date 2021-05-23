# Author: Ziga Trojer, zt0006@student.uni-lj.si
import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt

dataset = pd.read_csv('housing3.csv')
Y, X = dataset.Class, dataset.iloc[:, 0:13]
col_names = list(dataset.columns)
unique_class = list(Y.unique())
data_values, class_values = X.values, Y.values

def split_data(X, Y):
    """splits data into 80 % training and 20 % test data."""
    return X[:400], X[400:], Y[:400], Y[400:]

def get_classes(labels):
    """transforms data classes to {0,1} class"""
    classes = []
    for j in range(len(labels)):
        value = labels[j]
        for i, element in enumerate(unique_class):
            if value == element:
                classes.append(i)
    return np.array(classes)

X_train, X_test, Y_train, Y_test = split_data(data_values, class_values)
Y_train, Y_test = get_classes(Y_train), get_classes(Y_test)

class Node:
    def __init__(self, boundary, column_idx):
        self.right = None
        self.left = None
        self.prediction = None
        self.node_type = 'Node'
        self.boundary = boundary
        self.column_idx = column_idx

    def build_left(self, left):
        """Builds left tree"""
        self.left = left

    def build_right(self, right):
        """Builds right tree"""
        self.right = right

    def define_leaf(self, prediction):
        """Transforms node into leaf and saves predictions"""
        self.node_type = 'Leaf'
        self.prediction = prediction

    def predict_single(self, x):
        """Predicts the class of a single data point"""
        if self.node_type != 'Node':
            return self.prediction
        else:
            if x[self.column_idx] <= self.boundary:
                return self.left.predict_single(x)
            if x[self.column_idx] > self.boundary:
                return self.right.predict_single(x)

    def predict(self, all_data):
        """Makes a prediction for all data points"""
        prediction_list = []
        for i, x in enumerate(all_data):
            prediction_list.append(self.predict_single(x))
        return prediction_list

class Tree:
    def __init__(self, rand, get_candidate_columns, min_samples):
        self.rand = rand
        self.get_candidate_columns = get_candidate_columns
        self.min_samples = min_samples

    def build_tree(self, x_data, y_data, row_position, begin_classes):
        """Builds a tree and returns a model as an object. It uses Gini Index for splitting"""
        # get the splitting boundary, column index for splitting and gini index
        # for the splitting, consider only columns that are defined with the get_candidate_columns function
        splitting_boundary, column_idx, gini = \
            self.get_optimal_column_and_value(x_data, y_data, self.get_candidate_columns(x_data, self.rand))
        # make sure it is numpy array
        x_data, y_data = np.array(x_data), np.array(y_data)
        # separate data points by the splitting boundary
        left_idx, right_idx = self.seperate_data(x_data, column_idx, splitting_boundary)
        # save splitting boundary and column into a node
        current_node = Node(splitting_boundary, column_idx)
        # indexes of data points on left
        end_class_left = row_position[left_idx[0]]
        # if there is only one class in left node, we will make that node a leaf
        only_one_class_left, left_prediction = self.find_majority_class(begin_classes[end_class_left])
        # indexes of data points on right
        end_class_right = row_position[right_idx[0]]
        # if there is only one class in right node, we will make that node a leaf
        only_one_class_right, right_prediction = self.find_majority_class(begin_classes[end_class_right])
        # making a left node a leaf if:
        # 1) in node less data points than min_samples
        # OR 2) there is only one class left
        if ((len(left_idx[0]) < self.min_samples) and (len(left_idx[0]) > 0)) | only_one_class_left:
            left_node = Node(None, None)
            left_node.define_leaf(left_prediction)
            current_node.build_left(left_node)
        # if left node not a leaf, build a tree from a left node
        else:
            current_node.build_left(
                self.build_tree(x_data[left_idx[0]], y_data[left_idx[0]], row_position[left_idx[0]], begin_classes))
        # making a right node a leaf if:
        # 1) in node less data points than min_samples
        # OR 2) there is only one class right
        if ((len(right_idx[0]) < self.min_samples) and (len(right_idx[0]) > 0)) | only_one_class_right:
            right_node = Node(None, None)
            right_node.define_leaf(right_prediction)
            current_node.build_right(right_node)
        # if right node not a leaf, build a tree from a right node
        else:
            current_node.build_right(
                self.build_tree(x_data[right_idx[0]], y_data[right_idx[0]], row_position[right_idx[0]], begin_classes))
        return current_node

    def build(self, x_data, y_data):
        """Builds a tree and satisfies unit test"""
        return self.build_tree(x_data, y_data, np.array(list(range(len(x_data)))), y_data)

    def get_column_values_and_set(self, data, col):
        """For each data point, returns its values and middle points for boundary decisions"""
        column, boundary_values = list(), list()
        for i in range(len(data)):
            column.append(data[i][col])
        boundary_values_tmp = sorted(set(column))
        for j in range(len(boundary_values_tmp)):
            try:
                boundary_values.append((boundary_values_tmp[j + 1] + boundary_values_tmp[j]) / 2)
            except:
                continue
        return column, sorted(boundary_values)

    def get_gini_index(self, column, boundary, classes):
        """returns gini index for selected data point and boundary"""
        freq_c1, freq_c2, freq_c3, freq_c4 = 0, 0, 0, 0
        row_number_c1, row_number_c2 = 0, 0
        for i, elem in enumerate(column):
            if elem <= boundary:
                row_number_c1 += 1
                if classes[i] == 0:
                    freq_c1 += 1
                else:
                    freq_c3 += 1
            else:
                row_number_c2 += 1
                if classes[i] == 1:
                    freq_c2 += 1
                else:
                    freq_c4 += 1
        if row_number_c1 == 0:
            g1 = 0
        else:
            p1 = freq_c1 / row_number_c1
            p2 = freq_c3 / row_number_c1
            g1 = p1 * (1-p1) + p2 * (1 - p2)
        if row_number_c2 == 0:
            g2 = 0
        else:
            p3 = freq_c2 / row_number_c2
            p4 = freq_c4 / row_number_c2
            g2 = p3 * (1-p3) + p4 * (1-p4)
        return (row_number_c1 * g1 + row_number_c2 * g2) / (row_number_c1 + row_number_c2)

    def get_optimal_split(self, column, column_set, data_y):
        """Returns the feature that splits the data best"""
        gini_index_g = 2
        col_set = [0]
        for position, bound in enumerate(column_set):
            gini_index_l = self.get_gini_index(column, bound, data_y)
            if gini_index_l < gini_index_g:
                gini_index_g = gini_index_l
                position_g = position
                col_set = column_set[position_g]
        return col_set, gini_index_g

    def get_optimal_column_and_value(self, data_x, data_y, column_indexes):
        """Searches for the optimal column and boundary that splits the data best"""
        global_boundary = 0
        global_gini_index = 2
        position = -1
        for i in range(len(data_x[1])):
            if i in column_indexes:
                column_i, column_set_i = self.get_column_values_and_set(data_x, i)
                parameter_position, gini_index = self.get_optimal_split(column_i, column_set_i, data_y)
                if gini_index <= global_gini_index:
                    global_gini_index = gini_index
                    global_boundary = parameter_position
                    position = i
        return global_boundary, position, global_gini_index

    def get_row_index(self, x, y, list_boolean):
        """Returns indexes of the row"""
        lst_x, lst_y = list(), list()
        y = list(y)
        if len(x) == 0:
            return [], []
        else:
            for j, elem in enumerate(list_boolean):
                if list_boolean == 1:
                    lst_x.append(x[j])
                    lst_y.append(y[j])
            return lst_x, lst_y

    def seperate_data(self, data, column_idx, boundary):
        """Separates data to the left and right side by the selected boundary"""
        some = data[:, column_idx] <= boundary
        other = data[:, column_idx] > boundary
        left_dataset = np.where(some)
        right_dataset = np.where(other)
        return left_dataset, right_dataset

    def find_majority_class(self, lst):
        """Returns majority class. Randomly if 50/50"""
        first_class, second_class = 0, 0
        for elem in lst:
            if elem == 0:
                first_class += 1
            elif elem == 1:
                second_class += 1
        if first_class == 0:
            return True, 1
        elif second_class == 0:
            return True, 0
        elif first_class > second_class:
            return False, 0
        elif second_class > first_class:
            return False, 1
        else:
            return False, self.rand.choice(lst)

class EndModel:
    def __init__(self,  list_of_trees):
        self.list_of_trees = list_of_trees

    def predict(self, x_data):
        """Predicts the most common class"""
        tree_prediction = list()
        for tree in self.list_of_trees:
            tree_prediction.append(tree.predict(x_data))
        return self.find_majority_classes(tree_prediction)

    def predict_k(self, x_data, k):
        """Predicts the most common class for first k < len(list_of_trees) trees"""
        tree_prediction = list()
        n = 0
        for tree in self.list_of_trees:
            if n < k:
                tree_prediction.append(tree.predict(x_data))
                n += 1
        return self.find_majority_classes(tree_prediction)

    def find_majority_classes(self, classes_list):
        """Finds the most common class in list of classes"""
        end_classes = [0] * len(classes_list[0])
        for class_list in classes_list:
            end_classes = [sum(cls) for cls in zip(end_classes, class_list)]
        return [int(round(x / len(classes_list))) for x in end_classes]

class Bagging:
    def __init__(self, rand, tree_builder, n):
        self.rand = rand
        self.tree_builder = tree_builder
        self.n = n

    def build(self, x_data, y_data):
        """Builds trees and returns an object"""
        all_trees = list()
        tree_object = self.tree_builder
        for t in range(self.n):
            bootstrap_sample_X, bootstrap_sample_Y = self.resample_data(x_data, y_data)
            bootstrap_tree = tree_object.build(bootstrap_sample_X, bootstrap_sample_Y)
            all_trees.append(bootstrap_tree)
        return EndModel(all_trees)

    def resample_data(self, X_data, Y_data):
        """Resample data with replacement"""
        sample_X, sample_Y = list(), list()
        n_sample = round(len(X_data))
        while len(sample_X) < n_sample:
            index = self.rand.randrange(len(X_data))
            sample_X.append(X_data[index].tolist())
            sample_Y.append(Y_data[index].tolist())
        return np.array(sample_X), np.array(sample_Y)

class RandomForest:
    def __init__(self, rand, n, min_samples):
        self.rand = rand
        self.n = n
        self.min_samples = min_samples

    def build(self, x_data, y_data):
        t = Tree(rand=self.rand,
                 get_candidate_columns=self.get_random_feature,
                 min_samples=self.min_samples)
        b = Bagging(rand=self.rand, tree_builder=t, n=self.n)
        return b.build(x_data, y_data)

    def get_random_feature(self, x_data, rnd):
        n_sample = len(x_data[0])
        sqrt_sample = round(math.sqrt(n_sample))
        lst = rnd.sample(range(n_sample), sqrt_sample)
        return sorted(lst)

def misclassification_rate_tree(predicted_class_list, true_class_list):
    """Calculates the misclassification rate!"""
    wrong_classification = 0
    for pred, act in zip(predicted_class_list, true_class_list):
        if pred != act:
            wrong_classification += 1
    return round(wrong_classification / len(predicted_class_list), 4)

def return_same_idx(X, rand):
    """Returns indexes of all columns"""
    return list(range(len(X[0])))

def hw_tree_full(train, test):
    """Calculates misclassification rates for tree"""
    print('---CLASSIFICATION TREE---')
    Xtrain, Ytrain = train
    Xtest, Ytest = test
    print('Constructing classification tree!')
    t = Tree(rand=random.Random(1), get_candidate_columns=return_same_idx, min_samples=2)
    print('Building a model')
    b = t.build(Xtrain, Ytrain)
    print('Predicting train classes')
    train_prediction = b.predict(Xtrain)
    print('Predicted! ... Predicting test classes')
    test_prediction = b.predict(Xtest)
    train_misclassification = misclassification_rate_tree(train_prediction, list(Ytrain))
    test_misclassification = misclassification_rate_tree(test_prediction, list(Ytest))
    print('(Train error, Test error) = ')
    return train_misclassification, test_misclassification

def split_index(x_data, k):
    """Splits data into k folds"""
    folds = list()
    indexes = list(range(len(x_data)))
    for j in range(k):
        fold = random.Random(0).sample(indexes, round(len(x_data) / k))
        folds.append(fold)
        for element in fold:
            indexes.remove(element)
    return folds, list(range(len(x_data)))

def get_cross_validation_data(x_data, y_data, k):
    """Returns training and testing folds of x_data and y_data"""
    train_x, train_y = list(), list()
    test_x, test_y = list(), list()
    indexes, all_index = split_index(x_data, k)
    for test_index in indexes:
        test_y.append(y_data[test_index])
        test_x.append(x_data[test_index])
        train_index = [i for i in all_index if i not in test_index]
        train_x.append(x_data[train_index])
        train_y.append(y_data[train_index])
    return train_x, train_y, test_x, test_y

def hw_cv_min_samples_advanced(train, test, return_plot_results):
    """Calculates misclassification rate different values of min_samples and returns the best.
    If return_plot_results = True, it returns all data needed for plotting misclassification rates
    If return_plot_results = False, it returns training and test misclassification and best min_samples
    """
    print('---CLASSIFICATION TREE---')
    Xtrain, Ytrain = train
    Xtest, Ytest = test
    print('Splitting data into folds')
    k, k_is_ok = 5, False
    average_train_error, average_test_error = list(), list()
    while not k_is_ok:
        try:
            cross_train_x, cross_train_y, cross_test_x, cross_test_y = get_cross_validation_data(Xtrain, Ytrain, k)
            k_is_ok = True
        except:
            print('K is too big! New k is ' + str(k-1))
            k = k - 1
            continue
    print('New k is ' + str(k))
    min_samples_list = list(range(1, 51))
    global_train_miss, global_test_miss, best_split = 1, 1, 1000
    for min_sample in min_samples_list:
        print('Min sample:' + str(min_sample))
        cross_train_misclassification = 0
        cross_test_misclassification = 0
        for X, Y, Z, W in zip(cross_train_x, cross_train_y, cross_test_x, cross_test_y):
            t = Tree(rand=random.Random(1), get_candidate_columns=return_same_idx, min_samples=min_sample)
            b = t.build(X, Y)
            train_prediction = b.predict(X)
            test_prediction = b.predict(Z)
            cross_train_misclassification += misclassification_rate_tree(train_prediction, list(Y))
            cross_test_misclassification += misclassification_rate_tree(test_prediction, list(W))
        average_cross_train_misclassification = cross_train_misclassification / k
        print('Average misclassification train: ' + str(average_cross_train_misclassification))
        average_cross_test_misclassification = cross_test_misclassification / k
        print('Average misclassification test: ' + str(average_cross_test_misclassification))
        average_train_error.append(average_cross_train_misclassification)
        average_test_error.append(average_cross_test_misclassification)
        if average_cross_train_misclassification < global_train_miss:
            global_train_miss = average_cross_train_misclassification
        if average_cross_test_misclassification < global_test_miss:
            global_test_miss = average_cross_test_misclassification
            best_split = min_sample
    t = Tree(rand=random.Random(1), get_candidate_columns=return_same_idx, min_samples=best_split)
    b = t.build(Xtrain, Ytrain)
    print('Predicting train classes')
    train_prediction = b.predict(Xtrain)
    print('Predicting test classes')
    test_prediction = b.predict(Xtest)
    train_misclassification = misclassification_rate_tree(train_prediction, list(Ytrain))
    test_misclassification = misclassification_rate_tree(test_prediction, list(Ytest))
    print('(Train error, Test error) = ')
    if return_plot_results:
        return train_misclassification, test_misclassification, \
               best_split, average_train_error, average_test_error, min_samples_list
    else:
        return train_misclassification, test_misclassification, best_split

def hw_cv_min_samples(train, test):
    """Function that satisfies unit tests"""
    return hw_cv_min_samples_advanced(train, test, False)

def hw_bagging_advanced(train, test, return_plot_results):
    """Function that is used to calculate rates for bagging"""
    if return_plot_results:
        possible_number_of_trees = list(range(1, 150))
        possible_seeds = list(range(3))
        Xtrain, Ytrain = train
        Xtest, Ytest = test
        seed_list_misclassification = list()
        for sid in possible_seeds:
            print('Building new Bagging for seed ' + str(sid))
            test_misclassification_list = list()
            t = Tree(rand=random.Random(sid), get_candidate_columns=return_same_idx, min_samples=2)
            b = Bagging(rand=random.Random(sid), tree_builder=t, n=max(possible_number_of_trees))
            p = b.build(Xtrain, Ytrain)
            for num in possible_number_of_trees:
                print('Predicting on first ' + str(num) + ' trees!')
                test_prediction = p.predict_k(Xtest, num)
                test_misclassification = misclassification_rate_tree(test_prediction, list(Ytest))
                test_misclassification_list.append(test_misclassification)
            seed_list_misclassification.append(test_misclassification_list)
        return seed_list_misclassification, possible_number_of_trees, possible_seeds
    else:
        print('---------BAGGING---------')
        Xtrain, Ytrain = train
        Xtest, Ytest = test
        print('Constructing classification tree and bagging!')
        t = Tree(rand=random.Random(1), get_candidate_columns=return_same_idx, min_samples=2)
        b = Bagging(rand=random.Random(0), tree_builder=t, n=50)
        print('Building a model')
        p = b.build(Xtrain, Ytrain)
        print('Predicting train classes')
        train_prediction = p.predict(Xtrain)
        print('Predicting test classes')
        test_prediction = p.predict(Xtest)
        train_misclassification = misclassification_rate_tree(train_prediction, list(Ytrain))
        test_misclassification = misclassification_rate_tree(test_prediction, list(Ytest))
        print('(Train error, Test error) = ')
        return train_misclassification, test_misclassification

def hw_bagging(train, test):
    """Function that satisfies unit tests"""
    return hw_bagging_advanced(train, test, False)

def hw_randomforests_advanced(train, test, return_plot_results):
    """Function that is used to calculate rates for random forest"""
    if return_plot_results:
        possible_number_of_trees = list(range(1, 150))
        possible_seeds = list(range(3))
        Xtrain, Ytrain = train
        Xtest, Ytest = test
        seed_list_misclassification = list()
        for sid in possible_seeds:
            print('Building new Forest for seed ' + str(sid))
            test_misclassification_list = list()
            rf = RandomForest(rand=random.Random(sid), n=max(possible_number_of_trees), min_samples=2)
            p = rf.build(Xtrain, Ytrain)
            for num in possible_number_of_trees:
                print('Predicting on first ' + str(num) + ' trees!')
                test_prediction = p.predict_k(Xtest, num)
                test_misclassification = misclassification_rate_tree(test_prediction, list(Ytest))
                test_misclassification_list.append(test_misclassification)
            seed_list_misclassification.append(test_misclassification_list)
        return seed_list_misclassification, possible_number_of_trees, possible_seeds
    else:
        print('------RANDOM FOREST------')
        Xtrain, Ytrain = train
        Xtest, Ytest = test
        print('Constructing random forest!')
        rf = RandomForest(rand=random.Random(0), n=50, min_samples=2)
        print('Building a model')
        p = rf.build(Xtrain, Ytrain)
        print('Predicting train classes')
        train_prediction = p.predict(Xtrain)
        print('Predicting test classes')
        test_prediction = p.predict(Xtest)
        train_misclassification = misclassification_rate_tree(train_prediction, list(Ytrain))
        test_misclassification = misclassification_rate_tree(test_prediction, list(Ytest))
        print('(Train error, Test error) = ')
        return train_misclassification, test_misclassification

def hw_randomforests(train, test):
    """Function that satisfies unit tests"""
    return hw_randomforests_advanced(train, test, False)

def draw_plots(train, test):
    """Function that draws plots"""
    _, _, best_split, cross_validate_train, cross_validate_test, min_samples_list = hw_cv_min_samples_advanced(train, test, True)
    plt.plot(min_samples_list,cross_validate_test)
    plt.plot(min_samples_list,cross_validate_train)
    plt.axvline(x=best_split, color='k', linestyle='--')
    plt.legend(['Test misclassification', 'Train misclassification'], loc='lower right')
    plt.title('Misclassification rates versus min samples')
    plt.xlabel('min samples')
    plt.ylabel('misclassification rate')
    plt.show()

    plotting_results, number_of_trees, seeds = hw_bagging_advanced(train, test, True)
    for test in plotting_results:
        plt.plot(number_of_trees, test)
    plt.legend(['Test with seed ' + str(seed) for seed in seeds])
    plt.title('Misclassification rate versus the number of trees (bagging)')
    plt.xlabel('number of trees')
    plt.ylabel('misclassification rate')
    plt.show()

    plotting_results, number_of_trees, seeds = hw_randomforests_advanced(train, test, True)
    for test in plotting_results:
        plt.plot(number_of_trees, test)
    plt.legend(['Test with seed ' + str(seed) for seed in seeds])
    plt.title('Misclassification rate versus the number of trees (random forest)')
    plt.xlabel('number of trees')
    plt.ylabel('misclassification rate')
    plt.show()

train = (X_train, Y_train)
test = (X_test, Y_test)

if __name__ == "__main__":

    print(hw_tree_full(train, test))
    #print(hw_cv_min_samples(train, test))
    #print(hw_bagging(train, test))
    #print(hw_randomforests(train, test))
    #draw_plots(train, test)