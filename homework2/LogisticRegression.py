# Author: Ziga Trojer, zt0006@student.uni-lj.si
import pandas as pd
import numpy as np
import math
import scipy
from scipy import optimize
from sklearn.preprocessing import StandardScaler
import random
import scipy.stats as st

def p_object_j_observation_i(current_class, X, B):
    """
    :param current_class: chosen class
    :param X: input data
    :param B: matrix of weights
    :return: probabilities for class current_class
    """
    denominator = 0
    for c in range(len(classes) - 1):
        denominator += pow(math.e, np.dot(B[c, 1:], np.transpose(X)) + B[c, :1])
    if current_class < len(classes) - 1:
        return pow(math.e, np.dot(B[current_class, 1:], np.transpose(X)) + B[current_class, :1]) / (1 + denominator)
    else:
        return 1 / (1 + denominator)

def softmax(X, B):
    """
    :param X: input data
    :param B: matrix of weight
    :return: softmax array of size (number of data points) x (number of classes)
    """
    softmax_array = np.zeros((X.shape[0], B.shape[0] + 1))
    for c in range(len(classes)):
        softmax_array[:, c] = np.array(p_object_j_observation_i(c, X, B))[:]
    return softmax_array

def log_likelihood(B, X, y, shape):
    """
    :param B: matrix of weights
    :param X: input data
    :param y: input response
    :param shape: shape of B
    :return: log-likelihood
    """
    B = np.reshape(B, shape)
    temp_likelihood = 0
    softmax_vector = softmax(X, B)
    softmax_vector_log = np.log(softmax_vector)
    for k in range(len(y)):
        true_class = y[k]
        temp_likelihood += softmax_vector_log[k, true_class]
    return -temp_likelihood

def standard_logistic_distribution(x):
    """
    :param x: x
    :return: value between 0 and 1
    """
    return 1/2 + 1/2 * np.tanh(x/2)

def ordinal_probabilities(X, W, T):
    """
    :param X: input data
    :param W: vector of weights
    :param T: vector of boundaries
    :return: probabilities for ordinal logistic regression
    """
    u = np.zeros((X.shape[0], 1))
    p = np.zeros((X.shape[0], len(T)-1))
    for i in range(X.shape[0]):
        u[i] = np.dot(W[1:], X[i]) + W[0]
        for j in range(0, len(T)-1):
            p[i, j] = standard_logistic_distribution(T[j+1]-u[i][0]) - standard_logistic_distribution(T[j]-u[i][0])
    return p

def log_likelihood_ordinal(weights_W_T, X, y):
    """
    :param weights_W_T: combined weights and boundaries
    :param X: input data
    :param y: input response
    :return: log_likelihood value
    """
    bound = X.shape[1]
    W = weights_W_T[:bound+1]
    diff = weights_W_T[bound+1:]
    temp_likelihood = 0
    T = np.zeros((len(set(y))))
    T[0] = -np.Inf
    T = np.append(T, np.Inf)
    for k in range(2, len(T)-1):
        T[k] = T[k - 1] + diff[k - 1]
    probab = ordinal_probabilities(X, W, T)
    for k in range(len(y)):
        true_class = y[k]
        temp_likelihood += math.log(probab[k, true_class])
    return -temp_likelihood

def get_accuracy(predictions, real_values):
    """
    :param predictions: vector of predictions
    :param real_values: true values
    :return: accuracy
    """
    acc = 0
    for first, second in zip(predictions, real_values):
        if first == second:
            acc += 1
    return acc / len(predictions)

class ModelMultinomial:
    def __init__(self):
        self.weights = None

    def update_weights(self, new_weights):
        """
        :param new_weights: array of new weights
        """
        self.weights = new_weights

    def predict(self, X):
        """
        :param X: input data
        :return: vector of predictions
        """
        predicted_class = list()
        probabilities = softmax(X, self.weights)
        prediction_len = len(probabilities)
        for k in range(prediction_len):
            pred = list(probabilities[k]).index(max(list(probabilities[k])))
            predicted_class.append(pred)
        return predicted_class

    def log_loss(self, X, y):
        """
        :param X: input data
        :param y: true values
        :return: log-loss value
        """
        log_loss_value = 0
        probabilities = softmax(X, self.weights)
        for i in range(len(y)):
            true_class = y[i]
            prob = probabilities[i, true_class]
            log_loss_value += math.log(prob)
        return - log_loss_value / len(y)

class ModelLogistic:
    def __init__(self):
        self.weights = None
        self.boundary = None

    def update_weights(self, new_weights):
        """
        :param new_weights: vector of new weights
        """
        self.weights = new_weights

    def update_boundary(self, new_boundary):
        """
        :param new_boundary: vector of new boundaries
        """
        self.boundary = new_boundary

    def predict(self, X):
        """
        :param X: input data
        :return: vector of predictions
        """
        predicted_class = list()
        probabilities = ordinal_probabilities(X, self.weights, self.boundary)
        prediction_len = len(probabilities)
        for k in range(prediction_len):
            pred = list(probabilities[k]).index(max(list(probabilities[k])))
            predicted_class.append(pred)
        return predicted_class

    def log_loss(self, X, y):
        """
        :param X: input data
        :param y: true values
        :return: log-loss value
        """
        log_loss_value = 0
        probabilities = ordinal_probabilities(X, self.weights, self.boundary)
        for i in range(len(y)):
            true_class = y[i]
            prob = probabilities[i, true_class]
            log_loss_value += math.log(prob)
        return - log_loss_value / len(y)

class MultinomialLogReg:
    def __init__(self):
        self.initial_size = 2

    def build(self, X, y):
        """
        :param X: input data
        :param y: input response
        :return: model as an object ModelMultinomial
        """
        model = ModelMultinomial()
        shape = (len(list(set(y))) - 1, X.shape[1]+1)
        initial_guess = np.ones(shape) / self.initial_size
        c = scipy.optimize.fmin_l_bfgs_b(log_likelihood, x0=initial_guess, args=[X, y, shape], approx_grad=True)
        model.update_weights(c[0].reshape(shape))
        if c[2]['warnflag'] == 0:
            print('Optimization algorithm converged. Multinomial Logistic Regression successfully fitted!')
        elif c[2]['warnflag']:
            print('Too many function evaluations or too many iterations!')
        else:
            print(f"Stopped for the reason: {c[2]['task']}")
        return model

class OrdinalLogReg:
    def __init__(self):
        self.initial_size = 15

    def build(self, X, y):
        """
        :param X: input data
        :param y: input response
        :return: model as an object ModelLogistic
        """
        model = ModelLogistic()
        shape_diff = (1, len(set(y))-1)
        shape = (1, X.shape[1]+1)
        initial_guess = np.ones(shape) / self.initial_size
        initial_guess = np.append(initial_guess, np.full(shape_diff, 1e-10))
        bounds = [(None, None)] * shape[1]
        bounds += [(1e-10, None)] * shape_diff[1]
        c = scipy.optimize.fmin_l_bfgs_b(log_likelihood_ordinal,
                                         x0=initial_guess,
                                         args=[X, y],
                                         bounds=bounds,
                                         approx_grad=True)
        model.update_weights(c[0][:shape[1]])
        # write coefficients into csv
        print(c[0])
        write = False
        if write:
            keys = ['intersection', 'age', 'sex', 'year', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'delta1', 'delta2', 'delta3']
            values = c[0]
            res = {keys[i]: values[i] for i in range(len(keys))}
            df = pd.DataFrame(res, index=[0])
            df.to_csv('weights_logistic.csv', index=False)

        if c[2]['warnflag'] == 0:
            print('Optimization algorithm converged. Ordinal Logistic Regression successfully fitted!')
        elif c[2]['warnflag']:
            print('Too many function evaluations or too many iterations!')
        else:
            print(f"Stopped for the reason: {c[2]['task']}")
        T = np.zeros((len(set(y))))
        T[0] = -np.Inf
        T = np.append(T, np.Inf)
        for k in range(2, len(T)-1):
            T[k] = T[k-1] + c[0][shape[1]+k-1]
        model.update_boundary(T)
        print(T)
        return model

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

def naive_model_log_loss(y):
    log_loss_value = 0
    p = [0.15, 0.1, 0.05, 0.4, 0.3]
    for t in range(len(y)):
        log_loss_value += math.log(p[y[t]])
    return -log_loss_value / len(y)

def scale_data(X):
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler.transform(X)

def log_loss_cv(X, y, k):
    cross_train_x, cross_train_y, cross_test_x, cross_test_y = get_cross_validation_data(X, y, k)
    multinomial_list = list()
    ordinal_list = list()
    naive_list = list()
    for X, Y, Z, W in zip(cross_train_x, cross_train_y, cross_test_x, cross_test_y):
        print('Next fold!')
        X = scale_data(X)
        Z = scale_data(Z)
        multinomial_mdl = MultinomialLogReg()
        multinomial_mdl_build = multinomial_mdl.build(X, Y)
        multinomial_list.append(multinomial_mdl_build.log_loss(Z, W))
        ordinal_mdl = OrdinalLogReg()
        ordinal_mdl_build = ordinal_mdl.build(X, Y)
        ordinal_list.append(ordinal_mdl_build.log_loss(Z, W))
        naive_list.append(naive_model_log_loss(W))
    multinomial_ci = st.t.interval(0.95, len(multinomial_list) - 1, np.mean(multinomial_list), st.sem(multinomial_list))
    ordinal_ci = st.t.interval(0.95, len(ordinal_list) - 1, np.mean(ordinal_list), st.sem(ordinal_list))
    naive_ci = st.t.interval(0.95, len(naive_list) - 1, np.mean(naive_list), st.sem(naive_list))
    return multinomial_ci, np.mean(multinomial_list),  ordinal_ci, np.mean(ordinal_list), naive_ci, np.mean(naive_list)



if __name__ == "__main__":
    # load data
    dataset = pd.read_csv('dataset.csv', sep=';')
    # correlation = dataset.corr(method='pearson')
    # correlation.to_csv('correlation.csv', index=False)
    # separate to response
    Y_data, X_data = dataset.response, dataset.iloc[:, 1:]
    # define classes
    classes_ordinal = Y_data.unique()
    class_order = [4, 1, 5, 3, 2]
    classes_ordinal = [x for _, x in sorted(zip(class_order, classes_ordinal))]
    classes = [y-1 for y, _ in sorted(zip(class_order, classes_ordinal))]
    d = dict([(y, x) for x, y in enumerate(classes_ordinal)])
    # transform feature sex into is_female
    sex_dict = {'M': 0, 'F': 1}
    X_data.sex = X_data.sex.map(sex_dict)
    new_Y = list(Y_data.map(d))
    new_X = X_data.values

    # set this value to True, if want to check CV
    cross_valid = True

    if cross_valid:
        print(log_loss_cv(new_X, new_Y, 10))

    new_X = scale_data(new_X)

    test_own_dataset = False
    if test_own_dataset:
        dataset2_train = pd.read_csv('multinomial_bad_ordinal_good_train.csv', sep=',')
        dataset2_test = pd.read_csv('multinomial_bad_ordinal_good_test.csv', sep=',')
        Y_data_train, X_data_train = dataset2_train.variance, dataset2_train.iloc[:, 1:]
        Y_data_test, X_data_test = dataset2_test.variance, dataset2_test.iloc[:, 1:]
        X_data_train = X_data_train.values
        X_data_test = X_data_test.values
        classes_ordinal_test = Y_data_train.unique()
        class_order_test = [3, 4, 2, 1]
        classes = [x for _, x in sorted(zip(class_order_test, classes_ordinal_test))]
        d_test = dict([(y, x) for x, y in enumerate(classes_ordinal_test)])
        new_Y_train = list(Y_data_train.map(d_test))
        new_Y_test = list(Y_data_test.map(d_test))
        X_data_train = scale_data(X_data_train)
        X_data_test = scale_data(X_data_test)
        multinomial_model = MultinomialLogReg()
        multinomial_model_build = multinomial_model.build(X_data_train, new_Y_train)
        multinomial_model_predictions = multinomial_model_build.predict(X_data_test)
        multinomial_loss = multinomial_model_build.log_loss(X_data_test, new_Y_test)
        ordinal_model = OrdinalLogReg()
        ordinal_model_build = ordinal_model.build(X_data_train, new_Y_train)
        ordinal_model_predictions = ordinal_model_build.predict(X_data_test)
        ordinal_loss = ordinal_model_build.log_loss(X_data_test, new_Y_test)
        print(f'Accuracy for multinomial logistic regression is '
              f'{get_accuracy(multinomial_model_predictions, new_Y_test)}. 'f'Log-loss is: {multinomial_loss}')
        print(f'Accuracy for ordinal logistic regression is '
              f'{get_accuracy(ordinal_model_predictions, new_Y_test)}. 'f'Log-loss is: {ordinal_loss}')
