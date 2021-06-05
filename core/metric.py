import numpy as np

def accuracy(actual, preds):
    if len(actual.shape) == 2:
        actual = np.argmax(actual, axis=1)
    return np.mean(preds == actual)