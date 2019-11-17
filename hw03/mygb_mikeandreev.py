#coding=utf-8

from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd


# Ваш email, который вы укажете в форме для сдачи
AUTHOR_EMAIL = 'mikeandreev@gmail.com'
# Параметрами с которыми вы хотите обучать деревья
TREE_PARAMS_DICT = {'max_depth': 7} # 'criterion': 'mse'}
# Параметр tau (learning_rate) для вашего GB
TAU = 0.05

class SimpleGB(BaseEstimator):
    def __init__(self, tree_params_dict, iters, tau):
        self.tree_params_dict = tree_params_dict
        self.iters = iters
        self.tau = tau
        self.change_false_val = True
        if self.change_false_val:
            self.predict_bound = 0.
        else:
            self.predict_bound = 0.5
        
    def fit(self, X_data, y_data):
        if self.change_false_val:
            y_data = y_data.copy()
            y_data[ y_data == 0 ] = -1
        self.base_algo = DecisionTreeRegressor(**self.tree_params_dict).fit(X_data, y_data)
        self.estimators = []
        curr_pred = self.base_algo.predict(X_data)
        acc = accuracy_score(y_data>self.predict_bound, curr_pred>self.predict_bound)
        for iter_num in range(self.iters):
            # Нужно посчитать градиент функции потер
            # Нужно обучить DecisionTreeRegressor предсказывать антиградиент
            if self.change_false_val:
              #anti_grad = y_data * np.exp( -y_data @ curr_pred )
              anti_grad = y_data / ( 1 + np.exp( - y_data @ curr_pred ) )
            else:
              anti_grad = y_data - curr_pred
            algo = DecisionTreeRegressor(**self.tree_params_dict).fit(X_data, anti_grad)
            next_pred = algo.predict(X_data)
            self.estimators.append(algo)
            # Обновите предсказания в каждой точке
            curr_pred += (self.tau * next_pred)
        acc = accuracy_score(y_data>self.predict_bound, curr_pred>self.predict_bound)
        return self
    
    def predict(self, X_data):
        # Предсказание на данных
        res = self.base_algo.predict(X_data)
        for estimator in self.estimators:
            res += self.tau * estimator.predict(X_data)
        # Задача классификации, поэтому надо отдавать 0 и 1
        return res > self.predict_bound
