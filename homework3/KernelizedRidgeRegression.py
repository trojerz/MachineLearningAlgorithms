# Author: Ziga Trojer, zt0006@student.uni-lj.si
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import StandardScaler

def scale_data(X):
    """
    :param X: Input data
    :return: Scaled data
    """
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler.transform(X)

class Polynomial:
    """Polynomial kernel."""

    def __init__(self, M):
        self.M = M

    def __call__(self, x1, x2):
        try:
            x1.shape[1]
        except IndexError:
            x1 = x1.reshape(1, x1.shape[0])
        try:
            x2.shape[1]
        except IndexError:
            x2 = x2.reshape(1, x2.shape[0])
        return pow(1 + x1.dot(x2.T), self.M).squeeze()

class RBF:
    """RBF kernel."""

    def __init__(self, sigma):
        self.sigma = sigma

    @staticmethod
    def dist(a, b):
        return np.sum(np.multiply(a, b), axis=1)

    def __call__(self, x1, x2):
        try:
            x1.shape[1]
        except IndexError:
            x1 = x1.reshape(1, x1.shape[0])
        try:
            x2.shape[1]
        except IndexError:
            x2 = x2.reshape(1, x2.shape[0])
        norm_x1 = self.dist(x1, x1)
        norm_x2 = self.dist(x2, x2)
        matrix_norm_x1 = np.tile(norm_x1, (x2.shape[0], 1)).T
        matrix_norm_x2 = np.tile(norm_x2, (x1.shape[0], 1))
        dot_product = - 2 * x1.dot(x2.T)
        matrix_norm = matrix_norm_x1 + matrix_norm_x2 + dot_product
        return np.exp(-matrix_norm / (2 * pow(self.sigma, 2))).squeeze()

class Model:
    """Model on which we predict"""

    def __init__(self, X, kernel):
        self.alpha = None
        self.X = X
        self.kernel = kernel

    def update(self, alpha):
        self.alpha = alpha

    def predict(self, Y):
        krnl = self.kernel(Y, self.X)
        return np.dot(self.alpha, krnl.T)

class KernelizedRidgeRegression:
    """KernelizedRidgeRegression with fit method"""

    def __init__(self, kernel, lambda_):
        self.kernel = kernel
        self.lambda_ = lambda_

    def fit(self, X, y):
        model = Model(X, self.kernel)
        ident = np.identity(X.shape[0]) * self.lambda_
        alpha = np.dot(np.linalg.pinv(self.kernel(X, X) + ident).T, y).T
        model.update(alpha)
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

if __name__ == "__main__":
    show_sine = False
    show_house = True
    show_polynomial = True
    show_RBF = False

    if show_sine:
        sine = pd.read_csv('sine.csv', sep=',')
        sine_x = sine['x'].values
        sine_y = sine['y'].values

        fig, ax = plt.subplots(1)
        ax.plot(sine_x, sine_y, 'bo', label='Original data')

        new_data = np.arange(1, 20, step=0.2)

        sine_x = sine_x.reshape((sine_x.shape[0], 1))
        sine_y = sine_y.reshape((sine_y.shape[0], 1))
        new_x = new_data.reshape((new_data.shape[0], 1))

        fitter = KernelizedRidgeRegression(kernel=Polynomial(M=11), lambda_=0.001)
        m = fitter.fit(scale_data(sine_x), sine_y)
        pred = m.predict(scale_data(new_x))
        ax.plot(new_data, pred.reshape((len(new_data), 1)), '-', label='Polynomial M=11')

        fitter = KernelizedRidgeRegression(kernel=RBF(sigma=0.3), lambda_=0.001)
        m = fitter.fit(scale_data(sine_x), sine_y)
        pred = m.predict(scale_data(new_x))
        ax.plot(new_data, pred.reshape((len(new_data), 1)), '-', label='RBF sigma=0.3')
        plt.legend()
        plt.title('Fit sinus function using kernels')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    if show_house:
        house = pd.read_csv('housing2r.csv', sep=',').values
        X_train, X_test = house[:160, :5], house[160:, :5]
        y_train, y_test = house[:160, 5], house[160:, 5]
        fig, ax = plt.subplots(1)

        # Here you set lambdas
        lam = np.arange(1, 5, step=1)
        train_x, train_y, test_x, test_y = get_cross_validation_data(X_train, y_train, 5)

        if show_polynomial:
            ALL_RMSE = list()
            for m in range(1, 11):
                print(m)
                AVERAGE_RMSE = list()
                for lamb in lam:
                    RMSE_CV = list()
                    for X, Y, Z, W in zip(train_x, train_y, test_x, test_y):
                        fitter = KernelizedRidgeRegression(kernel=Polynomial(M=m), lambda_=lamb)
                        mod = fitter.fit(scale_data(X), Y)
                        pred = mod.predict(scale_data(Z))
                        prediction_list = list()
                        for x, y in zip(W, pred):
                            prediction_list.append(pow(x - y, 2))
                        # scalar
                        RMSE_CV.append(np.sqrt(np.sum(np.array(prediction_list)) / len(pred)))
                    AVERAGE_RMSE.append((np.mean(RMSE_CV), lamb))  # skalarji za cross validation
                ALL_RMSE.append(AVERAGE_RMSE)  # list skalarjev za vsak m
            best_lambdas = list()

            for j in range(1, 11):
                current_m = ALL_RMSE[j - 1]
                best_lambda = min(current_m, key=lambda t: t[0])[1]
                best_lambdas.append(best_lambda)
            print(best_lambdas)
            print(f'Those are best lambdas: {best_lambdas}')
            RMSE_best = list()
            for m in range(1, 11):
                fitter = KernelizedRidgeRegression(kernel=Polynomial(M=m), lambda_=best_lambdas[m - 1])
                m = fitter.fit(scale_data(X_train), y_train)
                pred = m.predict(scale_data(X_test))
                prediction_list = list()
                for x, y in zip(y_test, pred):
                    prediction_list.append(pow(x - y, 2))
                RMSE_best.append(np.sqrt(np.sum(np.array(prediction_list)) / len(pred)))
            ax.plot(list(range(1, 11)), RMSE_best, label='Polynomial lambda best')

            RMSE_fix = list()
            for m in range(1, 11):
                fitter = KernelizedRidgeRegression(kernel=Polynomial(M=m), lambda_=1)
                m = fitter.fit(scale_data(X_train), y_train)
                pred = m.predict(scale_data(X_test))
                prediction_list = list()
                for x, y in zip(y_test, pred):
                    prediction_list.append(pow(x - y, 2))
                RMSE_fix.append(np.sqrt(np.sum(np.array(prediction_list)) / len(pred)))
            ax.plot(list(range(1, 11)), RMSE_fix, label='Polynomial lambda=1')
            plt.legend()
            plt.title('RMSE depending on kernel parameter M')
            plt.xlabel('parameter M')
            plt.ylabel('RMSE')
            plt.show()

        ALL_RMSE = list()
        if show_RBF:
            # Here you set the parameters for sigma
            stepp = 0.5
            min_val = 0.5
            max_val = 18
            for m in np.arange(min_val, max_val, step=stepp):
                print(m)
                AVERAGE_RMSE = list()
                for lamb in lam:
                    RMSE_CV = list()
                    for X, Y, Z, W in zip(train_x, train_y, test_x, test_y):
                        fitter = KernelizedRidgeRegression(kernel=RBF(sigma=m), lambda_=lamb)
                        mod = fitter.fit(scale_data(X), Y)
                        pred = mod.predict(scale_data(Z))
                        prediction_list = list()
                        for x, y in zip(W, pred):
                            prediction_list.append(pow(x - y, 2))
                        # scalar
                        RMSE_CV.append(np.sqrt(np.sum(np.array(prediction_list)) / len(pred)))
                    AVERAGE_RMSE.append((np.mean(RMSE_CV), lamb))
                ALL_RMSE.append(AVERAGE_RMSE)
            best_lambdas = list()
            print(len(ALL_RMSE))
            for j in range(1, len(np.arange(min_val, max_val, step=stepp)) + 1):
                current_m = ALL_RMSE[j - 1]
                best_lambda = min(current_m, key=lambda t: t[0])[1]
                best_lambdas.append(best_lambda)
            print(f'Those are best lambdas: {best_lambdas}')
            RMSE_best = list()
            for m in range(1, len(np.arange(min_val, max_val, step=stepp)) + 1):
                fitter = KernelizedRidgeRegression(kernel=RBF(sigma=m), lambda_=best_lambdas[m - 1])
                mod = fitter.fit(scale_data(X_train), y_train)
                pred = mod.predict(scale_data(X_test))
                prediction_list = list()
                for x, y in zip(y_test, pred):
                    prediction_list.append(pow(x - y, 2))
                RMSE_best.append(np.sqrt(np.sum(np.array(prediction_list)) / len(pred)))
            ax.plot(list(np.arange(min_val, max_val, step=stepp)), RMSE_best, label='RBF lambda best')
            RMSE_fix = list()
            for m in np.arange(min_val, max_val, step=stepp):
                fitter = KernelizedRidgeRegression(kernel=RBF(sigma=m), lambda_=1)
                m = fitter.fit(scale_data(X_train), y_train)
                pred = m.predict(scale_data(X_test))
                prediction_list = list()
                for x, y in zip(y_test, pred):
                    prediction_list.append(pow(x - y, 2))
                RMSE_fix.append(np.sqrt(np.sum(np.array(prediction_list)) / len(pred)))
            ax.plot(list(np.arange(min_val, max_val, step=stepp)), RMSE_fix, label='RBF lambda=1')
            plt.legend()
            plt.title('RMSE depending on kernel parameter sigma')
            plt.xlabel('parameter sigma')
            plt.ylabel('RMSE')
            plt.show()
