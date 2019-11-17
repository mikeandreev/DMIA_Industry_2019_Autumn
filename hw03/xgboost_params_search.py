#!/usr/bin/env python
#coding=utf-8

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import pandas
import numpy as np
import signal
import os
import json
import sys
import traceback


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def signal_handler(signum, frame):
    raise Exception("Timed out!")


class Checker(object):
    def __init__(self, data_path=SCRIPT_DIR + '/HR.csv'):
        df = pandas.read_csv(data_path)
        target = 'left'
        features = [c for c in df if c != target]
        self.target = np.array(df[target])
        self.data = np.array(df[features])

    @staticmethod
    def get_estimator():
        return  xgb.XGBClassifier()

    def search(self, json_path):
        grid_search = GridSearchCV(
                self.get_estimator(),
                {
                    "learning_rate": [0.01 , 0.015, 0.02, 0.025, 0.03, 0.035],# 0.2, 0.3],
                    "max_depth": [2, 3, 5],# 10],
                    "n_estimators": [100, 250, 500, 1000],# 2000],# 5000],
                    "min_child_weight": [1, 3, 5, 10],
                    "scale_pos_weight": [0.5, 1, 2],
                    "seed": [42]
                },
                scoring='accuracy',
                cv=3
                )
        grid_search.fit(self.data, self.target)
        params = grid_search.best_params_
        print(params)
        params['author_email'] = 'mikeandreev@gmail.com'
        with open(json_path, 'w') as f:
            json.dump(params, f, indent=4)

    def check(self, params_path):
        author_email = None
        try:
            with open(params_path, 'r') as f:
                params = json.load(f)
                signal.signal(signal.SIGALRM, signal_handler)
                # Time limit на эту задачу 2 минуты
                signal.alarm(120)
                author_email = params.pop('author_email')
                estimator = self.get_estimator()
                estimator.set_params(**params)
                score = np.mean(cross_val_score(
                    estimator, self.data, self.target,
                    scoring='accuracy', 
                    cv=3
                ))

        except:
            traceback.print_exc()
            score = None
        
        return author_email, score


if __name__ == '__main__':
    json_path = SCRIPT_DIR + '/xgboost_params_search.json'
    checker = Checker()
    checker.search(json_path)
    print(checker.check(json_path))

