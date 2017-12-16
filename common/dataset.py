"""
Load datasets useful for this project, playing with pandas wherever possible.
"""

import pandas as pd
import requests


def load_abalones():
    'Loads the abalone dataset into a pandas dataframe'
    return pd.read_csv(
        'http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data',
        names=['sex', 'length', 'diameter', 'height',
               'whole_weight', 'shucked_weight', 'viscera_weight',
               'shell_weight', 'rings'],
        header=None)


def load_info():
    'Show the information about the dataset'
    with requests.get(
            'http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.names') as req:
        print('\n', req.text, '\n')
