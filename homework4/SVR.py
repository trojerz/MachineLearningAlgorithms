# Author: Ziga Trojer, zt0006@student.uni-lj.si
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import StandardScaler
from cvxopt import solvers
from cvxopt import matrix

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
        self.b = None
        self.X = X
        self.kernel = kernel

    def update(self, alpha, all_alpha, b, support_vectors):
        self.alpha = alpha
        self.b = b
        self.all_alpha = all_alpha
        self.support_vectors = support_vectors

    def get_alpha(self):
        return self.all_alpha

    def get_b(self):
        return self.b

    def get_support_vectors(self):
        return self.support_vectors

    def predict(self, Y):
        krnl = self.kernel(Y, self.X)
        return (np.dot(self.alpha, krnl.T) + self.b).squeeze()

class SVR:
    def __init__(self, kernel, lambda_, epsilon):
        self.kernel = kernel
        self.lambda_ = lambda_
        self.z = np.array([[1, -1], [-1, 1]])
        self.eps = epsilon
        self.small_eps = 1e-06

    def fit(self, X, y):
        model = Model(X, self.kernel)
        C = (1 / self.lambda_)
        krnl = self.kernel(X, X)
        # constructing matrix P - signs needs to be alternating.
        # also, matrix needs to be PD, so it is probably symmetric too
        p_dash = np.repeat(krnl, 2).reshape((krnl.shape[1], krnl.shape[0] * 2))
        p_dash = np.repeat(p_dash, 2, axis=0)
        z = np.tile(self.z, (krnl.shape[0], krnl.shape[1]))
        P = np.multiply(p_dash, z)
        # constructing vector q
        vec_ones = np.ones(P.shape[0])
        y_vec = np.repeat(y, 2).reshape(1, y.shape[0] * 2)
        y_vec = np.multiply(y_vec, np.tile(self.z[0], (1, y.shape[0])))
        q = + self.eps * vec_ones - y_vec
        # we need to construct other matrices & vectors too
        h = np.ones(krnl.shape[0] * 2) * C
        h = np.append(h, np.zeros(krnl.shape[0]*2))
        h = h.reshape((h.shape[0], 1))
        G = np.identity(krnl.shape[0] * 2)
        G = np.vstack([G, -G])
        A = np.tile(self.z[0], (1, y.shape[0]))
        # transforming all numpy objects into new matrix type
        # needed for optimization
        P = matrix(P, tc='d')
        q = matrix(q.T, tc='d')
        G = matrix(G, tc='d')
        h = matrix(h, tc='d')
        A = matrix(A, tc='d')
        b = matrix(np.zeros((1, 1)))
        solvers.options['show_progress'] = False
        # optimizing to get alphas
        solution = solvers.qp(P, q, G, h, A, b)
        alphas = np.array(solution['x'])
        # calculating alpha - alpha*
        alpha_diff = np.ediff1d(list(reversed(alphas)))
        alpha_diff = np.array(list(reversed(alpha_diff[::2])))
        alpha_diff = alpha_diff.reshape((1, alpha_diff.shape[0]))
        # checking conditions for calculating b
        ALPHA = alphas[::2]
        ALPHA_STAR = alphas[1::2]
        # only indexes that satisfy the condition
        lower_idx = np.array(list(set(np.union1d(np.where(ALPHA < C-self.small_eps)[0],
                                                 np.where(ALPHA_STAR > 0+self.small_eps)[0]))))
        upper_idx = np.array(list(set(np.union1d(np.where(ALPHA > 0 + self.small_eps)[0],
                                                 np.where(ALPHA_STAR < C - self.small_eps)[0]))))
        # calculating weights w
        weight = np.dot(alpha_diff, krnl).T
        # calculating lower and upper bound and filtering it by index, calculated before
        lower = y - self.eps - weight
        try:
            lower = lower[lower_idx]
        except IndexError:
            pass
        upper = y + self.eps - weight
        try:
            upper = upper[upper_idx]
        except IndexError:
            pass
        # max and min of lower and upper almost always coincide, but we handle the case when
        # they do not by calculating their mean.
        b = (np.min(upper) + np.max(lower)) / 2
        #alpha_diff[abs(alpha_diff) <= self.small_eps] = 0

        # saving which indexes are support vectors
        support_vectors = np.where(abs(alpha_diff) >= self.small_eps)[1]
        # updating the model
        # we like already calculated alphas, so we save all alphas for the unit tests
        model.update(np.array(alpha_diff), alphas.reshape((X.shape[0], 2)), np.array(b), support_vectors)
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
    show_polynomial = False
    show_RBF = True

    if show_sine:
        sine = pd.read_csv('sine.csv', sep=',')
        sine_x = sine['x'].values
        sine_y = sine['y'].values
        fig, ax = plt.subplots(1)
        ax.plot(sine_x, sine_y, 'ko', alpha = 0.2, label='Original data')
        new_data = np.arange(0, 20, step=0.1)
        sine_x = sine_x.reshape((sine_x.shape[0], 1))
        sine_y = sine_y.reshape((sine_y.shape[0], 1))
        new_x = new_data.reshape((new_data.shape[0], 1))
        # set the epsilon for sine
        epsilon = 0.5

        fitter = SVR(kernel=Polynomial(M=11), lambda_=0.1, epsilon=epsilon)
        m = fitter.fit(scale_data(sine_x), sine_y)
        pred = m.predict(scale_data(new_x))
        ax.plot(sine_x[m.get_support_vectors()], (sine_y[m.get_support_vectors()]), '2r',
                label='Support vectors Polynomial')
        pred2 = pred
        print(len(m.get_support_vectors()))
        fitter = SVR(kernel=RBF(sigma=0.3), lambda_=0.1, epsilon=epsilon)
        m = fitter.fit(scale_data(sine_x), sine_y)
        pred = m.predict(scale_data(new_x))
        ax.plot(sine_x[m.get_support_vectors()], (sine_y[m.get_support_vectors()]), '1',
                label='Support vectors RBF')
        print(len(m.get_support_vectors()))
        ax.plot(new_data, pred2.reshape((len(new_data), 1)), '-', label='Polynomial M=11')
        ax.plot(new_data, pred.reshape((len(new_data), 1)), '-', label='RBF sigma=0.3')
        plt.legend()
        plt.title('Fit sinus function using SVR')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    if show_house:
        house = pd.read_csv('housing2r.csv', sep=',').values
        X_train, X_test = house[:160, :5], house[160:, :5]
        y_train, y_test = house[:160, 5], house[160:, 5]
        fig, ax = plt.subplots(2)

        # Here you set lambdas
        lam = np.array([0.001, 0.01, 0.1, 1, 5, 10, 25, 50, 100])
        train_x, train_y, test_x, test_y = get_cross_validation_data(X_train, y_train, 5)
        # set the parameter epsilon
        epsilon = 4
        if show_polynomial:
            ALL_RMSE = list()

            for m in range(1, 11):
                AVERAGE_RMSE = list()
                for lamb in lam:
                    RMSE_CV = list()
                    for X, Y, Z, W in zip(train_x, train_y, test_x, test_y):
                        fitter = SVR(kernel=Polynomial(M=m), lambda_=lamb, epsilon=epsilon)

                        mod = fitter.fit(scale_data(X), np.array(Y))
                        pred = mod.predict(scale_data(Z))
                        print(f'Number of support vectors: {len(mod.get_support_vectors())}')
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
            number_support_vectors_best = list()
            for m in range(1, 11):
                fitter = SVR(kernel=Polynomial(M=m), lambda_=best_lambdas[m - 1], epsilon=epsilon)
                m = fitter.fit(scale_data(X_train), y_train)
                pred = m.predict(scale_data(X_test))
                number_support_vectors_best.append(len(m.get_support_vectors()))
                prediction_list = list()
                for x, y in zip(y_test, pred):
                    prediction_list.append(pow(x - y, 2))
                RMSE_best.append(np.sqrt(np.sum(np.array(prediction_list) / len(pred))))
            ax[0].plot(list(range(1, 11)), RMSE_best, label='Polynomial lambda best')
            ax[1].plot(list(range(1, 11)), number_support_vectors_best, label='Polynomial lambda best')
            RMSE_fix = list()
            number_support_vectors_fix = list()
            for m in range(1, 11):
                fitter = SVR(kernel=Polynomial(M=m), lambda_=1, epsilon=epsilon)
                m = fitter.fit(scale_data(X_train), y_train)
                pred = m.predict(scale_data(X_test))
                number_support_vectors_fix.append(len(m.get_support_vectors()))
                prediction_list = list()
                for x, y in zip(y_test, pred):
                    prediction_list.append(pow(x - y, 2))
                RMSE_fix.append(np.sqrt(np.sum(np.array(prediction_list)) / len(pred)))
            ax[0].plot(list(range(1, 11)), RMSE_fix, label='Polynomial lambda=1')
            plt.setp(ax[0], ylabel='RMSE')
            ax[1].plot(list(range(1, 11)), number_support_vectors_fix, label='Polynomial lambda=1')
            ax[0].set_title('RMSE depending on M')
            plt.legend()
            plt.title('Number of support vectors depending on M')
            #plt.title('RMSE depending on kernel parameter M')
            plt.ylabel('# support vectors')
            plt.xlabel('parameter M')
            handles, labels = ax[0].get_legend_handles_labels()
            plt.show()

        ALL_RMSE = list()
        if show_RBF:
            # Here you set the parameters for sigma
            sigmas = np.array([0.05, 0.5, 1, 2, 5, 10])
            for m in sigmas:
                print(m)
                AVERAGE_RMSE = list()
                for lamb in lam:
                    RMSE_CV = list()
                    for X, Y, Z, W in zip(train_x, train_y, test_x, test_y):
                        fitter = SVR(kernel=RBF(sigma=m), lambda_=lamb, epsilon=epsilon)
                        mod = fitter.fit(scale_data(X), np.array(Y))
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
            for j in range(1, len(sigmas) + 1):
                current_m = ALL_RMSE[j - 1]
                best_lambda = min(current_m, key=lambda t: t[0])[1]
                best_lambdas.append(best_lambda)
            print(f'Those are best lambdas: {best_lambdas}')
            RMSE_best = list()
            number_support_vectors_best = list()
            for m in range(1, len(sigmas) + 1):
                fitter = SVR(kernel=RBF(sigma=m), lambda_=best_lambdas[m - 1], epsilon=epsilon)
                mod = fitter.fit(scale_data(X_train), y_train)
                pred = mod.predict(scale_data(X_test))
                number_support_vectors_best.append(len(mod.get_support_vectors()))
                prediction_list = list()
                for x, y in zip(y_test, pred):
                    prediction_list.append(pow(x - y, 2))
                RMSE_best.append(np.sqrt(np.sum(np.array(prediction_list)) / len(pred)))
            ax[0].plot(list(sigmas), RMSE_best, label='RBF lambda best')
            ax[1].plot(list(sigmas), number_support_vectors_best, label='RBF lambda best')
            RMSE_fix = list()
            number_support_vectors_fix = list()
            for m in sigmas:
                fitter = SVR(kernel=RBF(sigma=m), lambda_=1, epsilon=epsilon)
                m = fitter.fit(scale_data(X_train), y_train)
                pred = m.predict(scale_data(X_test))
                number_support_vectors_fix.append(len(m.get_support_vectors()))
                prediction_list = list()
                for x, y in zip(y_test, pred):
                    prediction_list.append(pow(x - y, 2))
                RMSE_fix.append(np.sqrt(np.sum(np.array(prediction_list)) / len(pred)))
            ax[0].plot(list(sigmas), RMSE_fix, label='RBF sigma=1')
            plt.setp(ax[0], ylabel = 'RMSE')
            ax[0].set_title('RMSE depending on sigma')
            plt.title('Number of support vectors depending on sigma')
            ax[1].plot(list(sigmas), number_support_vectors_fix, label='RBF lambda=1')
            plt.legend()
            plt.xlabel('parameter sigma')
            plt.ylabel('# support vectors')
            plt.show()