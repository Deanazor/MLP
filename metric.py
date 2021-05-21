import numpy as np

def accuracy(actual, preds):
    return np.mean(preds == actual)