# Author: Ziga Trojer, zt0006@student.uni-lj.si
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
import sys

sys.path.append('../homework2')
sys.path.append('../homework3')

from hw_svr import SVR, RBF
from log_reg import MultinomialLogReg

class ANNRegression:
    def __init__(self, units, lambda_):
        self.units = units
        self.lambda_ = lambda_

    def fit(self, X, y):
        call_fit = NeuralNetwork(self.units, self.lambda_, None, None, X, y, 'linear')
        initial_weights = call_fit.generate_random_initial_weights(X, y, 'linear')
        optimized_weights = call_fit.heart_of_ann(X, y, initial_weights, 'linear')
        return NeuralNetwork(self.units, self.lambda_, optimized_weights, np.array(list(range(len(np.unique(y))))), X, y, 'linear')

class ANNClassification:
    def __init__(self, units, lambda_):
        self.units = units
        self.lambda_ = lambda_

    def fit(self, X, y):
        call_fit = NeuralNetwork(self.units, self.lambda_, None, None, X, y, 'softmax')
        initial_weights = call_fit.generate_random_initial_weights(X, y)
        optimized_weights = call_fit.heart_of_ann(X, y, initial_weights)
        return NeuralNetwork(self.units, self.lambda_, optimized_weights, np.array(list(range(len(np.unique(y))))), X, y, 'softmax')

class NeuralNetwork:
    def __init__(self, units, lambda_, optimized_weights, empty_y, X, y, final_layer):
        self.units = units
        self.lambda_ = lambda_
        self.optimized_weights = optimized_weights
        self.empty_y = empty_y
        self.final_layer = final_layer
        self.X = X
        self.X_shape = X.shape[1]
        self.y = y

    def predict(self, X):
        self.X = X
        return np.squeeze(self.forward(self.optimized_weights, X, self.empty_y, self.final_layer, False))

    def weights(self):
        return self.structure_input_back(self.optimized_weights, self.units, self.X_shape, self.y, self.final_layer)

    @staticmethod
    def sigmoid(x):
        """
        :param x:
        :return: sigmoid activation
        """
        return 1. / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        """
        :param x:
        :return: softmax activation
        """
        s = np.exp(x)
        d = np.sum(s, axis=1).reshape((s.shape[0], 1))
        return s / d

    @staticmethod
    def linear_output(x):
        """
        :param x: x
        :return: x
        """
        return x

    @staticmethod
    def structure_input_back(inputs, units, X_shape, y,  name='softmax'):
        """
        :param inputs: weights + bias
        :param units: hidden layers
        :param X: input data
        :param y: classes
        :param name: choose softmax / mse
        :return: list of numpy array in the right structure
        """
        if name == 'softmax':
            lay = [X_shape] + units + [len(np.unique(y))]
        else:
            lay = [X_shape] + units + [1]
        indexes, sizes, new_input = list(), list(), list()
        prev_idx = 0
        for k in range(len(lay) - 1):
            idx = ((lay[k] + 1) * lay[k + 1]) + prev_idx
            sizes.append((lay[k] + 1, lay[k + 1]))
            indexes.append(idx)
            prev_idx = idx
        indexes = indexes[:-1]
        splitted = np.array_split(inputs, indexes)
        for splt, sz in zip(splitted, sizes):
            new_input.append(splt.reshape(sz))
        return new_input

    def generate_random_initial_weights(self, X, y, name='softmax'):
        """
        :param X: input data
        :param y: classes
        :param name: choose softmax / mse
        :return: randomly initial weights
        """
        lay = [X.shape[1]] + self.units + [len(np.unique(y))] if name == 'softmax' else [X.shape[1]] + self.units + [1]
        weights_list = list()
        for i in range(len(lay) - 1):
            w = 2 * np.random.random((lay[i] + 1, lay[i + 1])) - 1
            weights_list.append(w)
        return weights_list

    def regularization(self, weights, lambda_):
        regulariz = 0
        for w in weights:
            without_bias = w[:-1, ]
            squares = np.sum(pow(without_bias, 2), axis = None)
            regulariz += squares
        return lambda_ / 2 * regulariz

    def forward(self, inputs, X, y, name='softmax', optimizator = True):
        """
        :param inputs: weights + bias
        :param X: input data
        :param y: classes
        :param name: choose softmax / mse
        :param optimizator: choose True when feeding into optimizator
        :return: loss to minimize
        """
        inputs = self.structure_input_back(inputs, self.units, self.X_shape, y, name)
        a = X
        for input_ in inputs[:-1]:
            w, b = input_[:-1, :], input_[-1:, :]
            z = np.dot(a, w) + b
            a = self.sigmoid(z)
        last_input = inputs[-1]
        w, b = last_input[:-1, :], last_input[-1:, :]
        z = np.dot(a, w) + b
        a = self.softmax(z) if name == 'softmax' else self.linear_output(z)
        if optimizator:
            loss_ = log_loss(a, y) if name == 'softmax' else mean_square_error(a, y)
            r_ = self.regularization(inputs, self.lambda_)
            return loss_ + r_
        else:
            return a

    def backprop(self, inputs, X, y, name='softmax'):
        """
        :param inputs: weights + bias
        :param X: input data
        :param y: classes
        :param name: choose softmax / mse
        :return: vector of gradients
        """
        inputs = self.structure_input_back(inputs, self.units, self.X_shape, y, name)
        activations, a = [X], X
        regularization = list()
        for input_ in inputs[:-1]:
            w, b = input_[:-1, :], input_[-1:, :]
            regularization.append((np.vstack((self.lambda_ * w, b * 0.))).flatten())
            z = np.dot(a, w) + b
            a = self.sigmoid(z)
            activations.append(a)
        last_input = inputs[-1]
        w, b = last_input[:-1, :], last_input[-1:, :]
        regularization.append((np.vstack((self.lambda_ * w, b * 0.))).flatten())
        z = np.dot(a, w) + b
        a = self.softmax(z) if name == 'softmax' else self.linear_output(z)
        activations.append(a)
        nabla_w = list()
        if name == 'softmax':
            gradd, grad_calc_w, grad_calc_b = self.calculate_gradient(activations, y)
        else:
            gradd, grad_calc_w, grad_calc_b = self.calculate_gradient_mse(activations, y)
        nabla_w.append(gradd.flatten())
        activations, prev_activations, inputs = activations[:-1], activations[:-2], inputs[1:]
        for current_activation, prev_activation, current_input in zip(activations[::-1],
                                                                      prev_activations[::-1],
                                                                      inputs[::-1]):
            grad_both = np.multiply(current_activation, (1 - current_activation))
            grad_w, grad_b = grad_both[:-1, :], grad_both[-1:, :]
            current_w, current_b = current_input[:-1, :], current_input[-1:, :]
            part_w = np.dot(grad_calc_w, current_w.T)
            part_b = np.dot(grad_calc_b, current_w.T)
            grad_calc_w = np.multiply(grad_w, part_w)
            grad_calc_b = np.multiply(grad_b, part_b)
            stacked = np.vstack((grad_calc_w, grad_calc_b))
            gradd = np.vstack((np.dot(prev_activation.T, stacked), np.sum(stacked, axis=0)))
            nabla_w.append(gradd.flatten())
        nabla_w = nabla_w[::-1]
        abc = np.concatenate(nabla_w, axis=None) + np.concatenate(regularization, axis=None)
        return abc

    def calculate_gradient(self, forward_activation, y):
        """
        :param forward_activation: vector of activations
        :param y: classes
        :return: gradient of softmax
        """
        act_layer_top, act_layer_prev = forward_activation[-1], forward_activation[-2]
        grad_calc = act_layer_top
        grad_calc[range(len(y)), y] = (grad_calc[range(len(y)), y] - 1)
        grad_calc = grad_calc / len(y)
        grad_calc_w, grad_calc_b = grad_calc[:-1, :], grad_calc[-1:, :]
        g_bias = np.sum(grad_calc, axis=0)
        g_weight = np.dot(act_layer_prev.T, grad_calc)
        return np.vstack((g_weight, g_bias)), grad_calc_w, grad_calc_b

    def calculate_gradient_mse(self, forward_activation, y):
        """
        :param forward_activation: vector of activations
        :param y: true values
        :return: gradient mse
        """
        act_layer_top, act_layer_prev = forward_activation[-1], forward_activation[-2]
        grad_calc = 2 * (act_layer_top - np.expand_dims(y, axis=0).T) / len(y)
        grad_calc_w, grad_calc_b = grad_calc[:-1, :], grad_calc[-1:, :]
        g_bias = np.sum(grad_calc, axis=0)
        g_weight = np.dot(act_layer_prev.T, grad_calc)
        return np.vstack((g_weight, g_bias)), grad_calc_w, grad_calc_b

    def gradient_numerical(self, X, y, weights, tolerance, epsilon, name):
        """
        :param X: input data
        :param y: classes
        :param weights: weights
        :param tolerance: tolerance for checking difference
        :param epsilon: small change in value to aprox. gradient
        :param name: choose softmax / mse
        :return:
        """
        weights = np.concatenate(weights, axis=None)
        length_weights = len(weights)
        fails = []
        for k in range(10):
            random_weights = 2 * np.random.random(length_weights) - 1
            my_gradient = self.backprop(random_weights, X, y, name).flatten()
            for j in range(length_weights):
                change = np.zeros(length_weights)
                change[j] = epsilon
                change = change + random_weights
                aprox = self.forward(change, X, y, name, True) - self.forward(random_weights, X, y, name, True)
                difference = abs(my_gradient[j] - aprox / epsilon)
                if difference > tolerance:
                    fails.append(True)
                else:
                    fails.append(False)
        if sum(fails) > 0:
            return False
        else:
            return True

    def heart_of_ann(self, X, y, inputs, name='softmax'):
        """
        :param X: input
        :param y: true value
        :param inputs: weights + bias
        :param name: choose softmax / mse
        :return: optimal values of weights, biases
        """
        print('Fitting ANN!')
        inputs = np.concatenate(inputs, axis=None)
        correct_gradients = self.gradient_numerical(X, y, inputs, 1e-02, 1e-06, name)
        try_to_be_funny = False
        if correct_gradients:
            print('Gradients OK!')
            if try_to_be_funny:
                print('Division by zero! System is shutting down in')
                time.sleep(1)
                print('3')
                time.sleep(1)
                print('2')
                time.sleep(1)
                print('1')
                time.sleep(1)
                print('Just joking, everything fine! Continue fitting...')
        else:
            print('ABORT! Ziga can not calculate gradients correctly!')
            raise
        # set maxiter to 5000 when fitting the huge dataset!!!
        weights_opt, _, _ = fmin_l_bfgs_b(self.forward, inputs, fprime=self.backprop, args=(X, y, name)) #, maxiter=5000)
        return weights_opt

def mean_square_error(a, y):
    return np.mean(pow(a - np.expand_dims(y, axis=0).T, 2))

def log_loss(a, y):
    class_y = np.zeros((y.size, y.max() + 1))
    class_y[np.arange(y.size), y] = 1
    log_loss_ = -np.sum(class_y * np.log(a)) / len(y)
    return log_loss_

def scale_data(X):
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler.transform(X)

def split_index(x_data, k):
    """Splits data into k folds"""
    folds = list()
    indexes = list(range(len(x_data)))
    for j in range(k):
        fold = random.Random(42).sample(indexes, round(len(x_data) / k))
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
        test_y.append(list(np.array(y_data)[test_index]))
        test_x.append(x_data[test_index])
        train_index = [i for i in all_index if i not in test_index]
        train_x.append(x_data[train_index])
        train_y.append(list(np.array(y_data)[train_index]))
    return train_x, train_y, test_x, test_y

def test_with_cv(X_train, y_train, k, name):
    testing_units = [[], [10], [20, 20], [10, 20, 10], [3, 9, 18, 3]]
    testing_lambdas = [0.00001, 0.0001, 0.001, 0.1, 1]
    train_x, train_y, test_x, test_y = get_cross_validation_data(X_train, y_train, k)
    results = []
    for unit in testing_units:
        for lam in testing_lambdas:
            print(f'Testing unit {unit} and lam {lam}')
            middle_results = []
            for X, Y, Z, W in zip(train_x, train_y, test_x, test_y):
                if name == 'regression':
                    fitter = ANNRegression(units=unit, lambda_=lam)
                else:
                    fitter = ANNClassification(units=unit, lambda_=lam)
                m = fitter.fit(scale_data(X), np.array(Y))
                pred = m.predict(scale_data(Z))
                if name == 'regression':
                    res = np.sum((W - pred.flatten()) ** 2) / len(W)
                else:
                    res = log_loss(pred, np.array(W))
                middle_results.append(res)
            average_res = (sum(middle_results) / len(middle_results))
            results.append((unit, lam, average_res))
            print(f'Those are results: {results[-1]}')
    best_parameters = min(results, key=lambda t: t[2])
    return best_parameters

def compare_results_for_regression(y_train, y_test, x_train, x_test, unitts, lammba_):
    # we got best parameters with CV
    fitter = ANNRegression(units=unitts, lambda_=lammba_)
    m = fitter.fit(scale_data(x_train), y_train)
    log_reg_fitter = SVR(kernel=RBF(sigma=2), lambda_=1, epsilon=2)
    m2 = log_reg_fitter.fit(scale_data(x_train), y_train)
    fold_idx, all_idx = split_index(x_test, 5)
    mse_all_ann, mse_all_svr = list(), list()
    for test_index in fold_idx:
        current_x_test = x_test[test_index]
        current_y_test = list(np.array(y_test)[test_index])
        pred = m.predict(scale_data(current_x_test))
        pred2 = m2.predict(scale_data(current_x_test))
        mse_current = np.sum((current_y_test - pred.flatten()) ** 2) / len(current_y_test)
        mse_current_SVR = np.sum((current_y_test - pred2.flatten()) ** 2) / len(current_y_test)
        mse_all_ann.append(mse_current)
        mse_all_svr.append(mse_current_SVR)
    mse_all_ann = np.array(mse_all_ann)
    mse_all_svr = np.array(mse_all_svr)
    print(f'Mean square error for ANN with best parameters: {np.mean(mse_all_ann)}, SD {2 * np.std(mse_all_ann)}. '
          f'95CI: {[max(0, np.mean(mse_all_ann) - 2 * np.std(mse_all_ann)), np.mean(mse_all_ann) + 2 * np.std(mse_all_ann)]}')
    print(f'Mean square error for SVR with best parameters: {np.mean(mse_all_svr)}, SD {2 * np.std(mse_all_svr)}. '
          f'95CI: {[max(0, np.mean(mse_all_svr) - 2 * np.std(mse_all_svr)), np.mean(mse_all_svr) + 2 * np.std(mse_all_svr)]}')

def compare_results_for_classification(y_train, y_test, x_train, x_test, unitts, lammba_):
    # we got best parameters with CV
    fitter = ANNClassification(units=unitts, lambda_=lammba_)
    m = fitter.fit(scale_data(x_train), y_train)
    multinomial_model = MultinomialLogReg()
    multinomial_model_build = multinomial_model.build(scale_data(x_train), y_train)
    fold_idx, all_idx = split_index(x_test, 5)
    logloss_all_ann, logloss_all_mr = list(), list()
    for test_index in fold_idx:
        current_x_test = x_test[test_index]
        current_y_test = list(np.array(y_test)[test_index])
        pred = m.predict(scale_data(current_x_test))
        log_loss_ann = log_loss(pred, np.array(current_y_test))
        log_loss_m = multinomial_model_build.log_loss(scale_data(current_x_test), current_y_test)
        logloss_all_ann.append(log_loss_ann)
        logloss_all_mr.append(log_loss_m)
    logloss_all_ann = np.array(logloss_all_ann)
    logloss_all_mr = np.array(logloss_all_mr)
    print(f'Mean square error for ANN with best parameters: {np.mean(logloss_all_ann)}, SD {2 * np.std(logloss_all_ann)}. '
          f'95CI: {[max(0, np.mean(logloss_all_ann) - 2 * np.std(logloss_all_ann)), np.mean(logloss_all_ann) + 2 * np.std(logloss_all_ann)]}')
    print(f'Mean square error for SVR with best parameters: {np.mean(logloss_all_mr)}, SD {2 * np.std(logloss_all_mr)}. '
          f'95CI: {[max(0, np.mean(logloss_all_mr) - 2 * np.std(logloss_all_mr)), np.mean(logloss_all_mr) + 2 * np.std(logloss_all_mr)]}')

def create_final_predictions():
    train_data = pd.read_csv('train.csv.gz', sep=',')
    test_data = pd.read_csv('test.csv.gz', sep=',')
    class_map = {'Class_1': int(0),
                 'Class_2': int(1),
                 'Class_3': int(2),
                 'Class_4': int(3),
                 'Class_5': int(4),
                 'Class_6': int(5),
                 'Class_7': int(6),
                 'Class_8': int(7),
                 'Class_9': int(8)}
    train_data = train_data.replace({'target': class_map}).values
    test_data = test_data.replace({'target': class_map}).values
    y_train, x_train, x_test = (train_data[:, -1]).astype(int), train_data[:, 1:-1], test_data[:, 1:]
    begin_cv = time.time()
    u, lam, min_log_loss = test_with_cv(x_train, y_train, 5, 'classification')
    end_cv = time.time()
    print(f'Minimum log-loss for huge dataset: {min_log_loss}, with parameter: units: {u}, lambda {lam}. Time: {end_cv - begin_cv} seconds')
    # u = [20, 20]
    # lam = 0.0001
    fitter2 = ANNClassification(u, lam)
    m2 = fitter2.fit(scale_data(x_train), np.array(y_train))
    # pred = m2.predict(scale_data(x_train))
    # print(log_loss(pred, np.array(y_train)))
    pred = m2.predict(scale_data(x_test))
    predictions = np.array(pred)
    prediction_time = time.time()
    with open('final.txt', 'w+') as file:
        file.write(','.join(['id', 'Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']) + '\n')
        for k in range(predictions.shape[0]):
            file.write(str(k + 1) + ',' + ','.join([str(x) for x in predictions[k, :]]) + '\n')
        file.close()
    write_time = time.time()
    print(f'Completed! Time for prediction {prediction_time - end_cv}, time for write: {write_time - prediction_time}')
    return pred

if __name__ == '__main__':
    housing_regression = True
    housing_classification = True
    large_dataset = False
    perform_CV = False
    if housing_regression:
        np.random.seed(42)
        housing2r = pd.read_csv('housing2r.csv', sep=',').values
        np.random.shuffle(housing2r)
        y_train, y_test, x_train, x_test = housing2r[:160, 5], housing2r[160:, 5], housing2r[:160, :5], housing2r[160:, :5]
        if perform_CV:
            u, lam, min_mse = test_with_cv(x_train, y_train, 5, 'regression')
            print(f'Minimum mse: {min_mse}, with parameter: units: {u}, lambda {lam}')
            compare_results_for_regression(y_train, y_test, x_train, x_test, u, lam)
        else:
            u, lam = [10], 0.1
            compare_results_for_regression(y_train, y_test, x_train, x_test, u, lam)

    if housing_classification:
        np.random.seed(42)
        housing3 = pd.read_csv('housing3.csv', sep=',')
        class_map = {'C1': int(0), 'C2': int(1)}
        housing3 = housing3.replace({'Class': class_map}).values
        np.random.shuffle(housing3)
        y_train, y_test, x_train, x_test = (housing3[:400, 13]).astype(int), (housing3[400:, 13]).astype(int), housing3[:400, :13], housing3[400:, :13]
        if perform_CV:
            u, lam, min_log_loss = test_with_cv(x_train, y_train, 5, 'classification')
            print(f'Minimum log-loss: {min_log_loss}, with parameter: units: {u}, lambda {lam}')
            compare_results_for_classification(y_train, y_test, x_train, x_test, u, lam)
        else:
            u, lam = [10], 0.001
            compare_results_for_classification(y_train, y_test, x_train, x_test, u, lam)

    if large_dataset:
        create_final_predictions()
