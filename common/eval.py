"""
Evaluation functions to help with sklearn
"""
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error


def mse_scorer():
    'Create a MSE scorer to use on Cross Validation'
    return make_scorer(mean_squared_error, greater_is_better=False)


def cross_val_rmse(estimator, X, y, cv=10):
    "Run cross validation using RMSE as scorer and return a np.ndarray"
    return np.sqrt(-cross_val_score(estimator, X, y, cv=cv, scoring=mse_scorer()))
